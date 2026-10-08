#!/usr/bin/env python3
"""ねじ知識表を引くだけの、OpenAI互換 chat completions サーバー。

MoshiRAG の検索バックエンドは OpenAI 互換エンドポイントとして呼ばれるだけで、
中身が LLM である必要はない（moshi/moshi/llm/client.py が openai SDK で
`chat.completions.create` を叩き、`choices[0].message.content` を参照テキスト
として使う）。ここではその顔だけを借りて、決定的な表引きを返す。

狙い:
    技術値を重みに覚えさせない。参照として渡し、表を差し替えれば答えが変わる
    状態にする。表に無ければ空文字を返す。moshi-rag 側は空文字を [RET_FAILED]
    として扱うので、「表に無い」がモデルの内部状態ではなく観測可能な事実になる。

MoshiRAG が学習時に見ている参照の形に合わせる:
    Reference: <事実>     1行、50語以内、プレーンテキスト、改行なし
    (moshi/moshi/llm/reference_prompt_template_simplified.txt)

使い方:
    python screw_poc/scripts/reference_server.py --port 8010          # 英語表
    python screw_poc/scripts/reference_server.py --lang ja --port 8010  # 日本語表

    export MOSHI_RETRIEVAL_LLMS_JSON='[{"id":"screw-table",
        "base_url":"http://localhost:8010/v1","model":"table","default":true}]'

    表を差し替えるだけで別の工場になる:
    python screw_poc/scripts/reference_server.py --knowledge other_factory.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

POC_ROOT = Path(__file__).resolve().parents[1]
# MoshiRAG の重みは英語 (Moshika) なので、既定は英語表。日本語表は
# --lang ja で選ぶ。表の差し替えが再学習なしで効くことの確認も兼ねている。
TABLES = {
    "en": (POC_ROOT / "knowledge" / "screw_knowledge_en.csv",
           POC_ROOT / "knowledge" / "screw_glossary_en.csv"),
    "ja": (POC_ROOT / "knowledge" / "screw_knowledge.csv",
           POC_ROOT / "knowledge" / "screw_glossary.csv"),
}
DEFAULT_POLICY = POC_ROOT / "config" / "dialogue_policy.yaml"

# 用語を訊いていると分かる言い回し。技術値の照会より先に判定する。
TERM_QUESTION = re.compile(
    r"って何|とは|どういう意味|what is|what are|what do you mean|what does .{1,40} mean"
    r"|i do not (?:know|follow) what|which (?:one|part|bit|dimension) is|where is the",
    re.IGNORECASE,
)

# 表に match_keywords 列が無いときの保険。行ごとではなく category 単位なので、
# 同じ category の行を区別できない。英語表は行ごとの match_keywords を持つ。
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pitch": ("ピッチ", "並目", "細目", "ねじ山", "山の間隔", "かける"),
    "torque": ("トルク", "ニュートン", "締付", "締め", "締ま", "軸力", "座金", "潤滑", "油", "アルミ", "樹脂"),
    "pilot_hole": ("下穴", "タップ", "板厚", "引抜"),
    "identification": ("工具", "レンチ", "ソケット", "刻印", "強度区分", "互換", "使え", "組み合", "種類", "違い", "インチ"),
    "trouble": ("固い", "かじり", "空回り", "回らな", "動かな", "再使用", "もう一度", "滑る", "焼き付"),
}

# 学習時の参照は50語以内。超えた分は落ちるより、こちらで削って形を保つ。
MAX_WORDS = 50

logger = logging.getLogger("screw_reference")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def split_aliases(value: str) -> list[str]:
    return [item.strip() for item in value.split("||") if item.strip()]


def load_readings(path: Path) -> dict[str, str]:
    """合成用の読み表を逆に引く。規格表示と品番だけに限る。

    数値や単位の読み（イチ→1 など）まで戻すと、無関係な語を壊す。
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available, skipping reading normalisation")
        return {}
    if not path.exists():
        return {}
    policy = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    readings: dict[str, str] = {}
    for entry in policy.get("tts_pronunciations") or []:
        written = str(entry.get("written", ""))
        spoken = str(entry.get("spoken", ""))
        if not written or not spoken:
            continue
        if re.fullmatch(r"M\d+", written) or "SSH-" in written:
            readings[spoken] = written
    # 長い読みから先に戻す。エムジュウニ が エムジュウ に食われないように。
    return dict(sorted(readings.items(), key=lambda kv: len(kv[0]), reverse=True))


class ScrewTable:
    """会話テキストから、引ける行を決定的に選ぶ。

    渡ってくる context は整形済みのクエリではなく、ASR が起こしたユーザー発話と
    moshi の inner monologue を繋いだ生の会話。ここから鍵になる語を拾う。
    安全側の表なので、曖昧なら引かない（空を返す）方に倒す。
    """

    def __init__(
        self,
        knowledge: list[dict[str, str]],
        glossary: list[dict[str, str]],
        readings: dict[str, str] | None = None,
        source_label: str = "Source",
    ) -> None:
        self.knowledge = knowledge
        self.glossary = glossary
        self.source_label = source_label
        # 「エムジュウニ」のような読み上げ表記で来ることがある。合成に使っている
        # 読みの表をそのまま逆に引いて、表記へ戻してから照合する。
        self.readings = readings or {}
        # 長い表記から先に当てる。M1 が M10 や M12 を食わないように。
        self.diameters = sorted(
            {
                value
                for row in knowledge
                for value in re.findall(r"M\d+", row.get("slot_values", ""))
            },
            key=len,
            reverse=True,
        )
        self.part_numbers = sorted(
            {
                value
                for row in knowledge
                for value in re.findall(r"[A-Z]{2,}-M\d+-[A-Z]+-[A-Z]+", row.get("slot_values", ""))
            },
            key=len,
            reverse=True,
        )

    # -- 用語の問い合わせ ---------------------------------------------------
    def lookup_term(self, context: str) -> str | None:
        """用語集から引く。知識表が何も返せなかったときの後段。

        英語では "what is ..." が値の照会にも用語の質問にも使われるので、
        質問の形では区別できない。技術値を持つ知識表を先に試し、そこが
        引けなかったときだけ用語を返す、という優先順で分ける。
        """
        folded = context.lower()
        best: tuple[int, dict[str, str]] | None = None
        for row in self.glossary:
            for surface in [row.get("term", ""), *split_aliases(row.get("aliases", ""))]:
                if surface and surface.lower() in folded:
                    if best is None or len(surface) > best[0]:
                        best = (len(surface), row)
        if best is None:
            return None
        row = best[1]
        return f"{row['what_text']} {row['where_text']}"

    # -- 技術値の照会 -------------------------------------------------------
    def lookup_knowledge(self, context: str) -> str | None:
        hits = [(self._score(row, context), row) for row in self.knowledge]
        hits = [(score, row) for score, row in hits if score > 0]
        if not hits:
            return None
        hits.sort(key=lambda pair: pair[0], reverse=True)
        top = hits[0][0]
        # 同点が並ぶときは特定できていない。推測で1行選ぶより、引かない。
        if sum(1 for score, _ in hits if score == top) > 1:
            logger.info("ambiguous match (score=%s), returning no reference", top)
            return None
        row = hits[0][1]
        parts = [row.get("answer_text", ""), row.get("caution_text", "")]
        source = row.get("source_title", "")
        if source:
            parts.append(f"{self.source_label} {source}.")
        return " ".join(part for part in parts if part)

    def normalize(self, context: str) -> str:
        text = context
        for spoken, written in self.readings.items():
            text = text.replace(spoken, written)
        return text

    def _keywords(self, row: dict[str, str]) -> tuple[tuple[str, ...], bool]:
        """行ごとの識別語。無ければ category 単位の保険に落ちる。"""
        explicit = split_aliases(row.get("match_keywords", ""))
        if explicit:
            return tuple(explicit), True
        return CATEGORY_KEYWORDS.get(row.get("category", ""), ()), False

    def _score(self, row: dict[str, str], context: str) -> int:
        folded = context.lower()
        score = 0
        slot_values = row.get("slot_values", "")
        for part_number in self.part_numbers:
            if part_number in slot_values and part_number.lower() in folded:
                score += 5
        for diameter in self.diameters:
            if diameter in slot_values and re.search(rf"{diameter}(?!\d)", context, re.IGNORECASE):
                score += 3
                break
        # 何の話かが合っていない行は、径が一致していても引かない。
        keywords, per_row = self._keywords(row)
        hits = sum(1 for word in keywords if word.lower() in folded)
        if hits:
            # 行ごとの語で当たったほうが、category 単位より強い根拠になる。
            score += (3 if per_row else 2) + hits
        elif score:
            # 径だけ当たって意図が読めていない状態。単独では引かせない。
            score -= 2
        return score

    def mentions_unknown_diameter(self, context: str) -> str | None:
        """表に無い呼び径を名指しされていないか。

        M14 のようにドメイン内で表に無い値は、いちばん危ない入力。近い行を
        返すと補間したことになるので、何も返さないために先に弾く。
        """
        known = set(self.diameters)
        for token in re.findall(r"\bM(\d+)\b", context, re.IGNORECASE):
            if f"M{token}" not in known:
                return f"M{token}"
        return None

    def reference_for(self, raw_context: str) -> str:
        context = self.normalize(raw_context)
        unknown = self.mentions_unknown_diameter(context)
        if unknown is not None:
            logger.info("%s is not in the table, returning no reference", unknown)
            return ""
        # 技術値を持つ知識表が先。引けなければ用語集に落ちる。
        text = self.lookup_knowledge(context) or self.lookup_term(context)
        if not text:
            return ""
        return clip_words(collapse(text), MAX_WORDS)


def collapse(text: str) -> str:
    """1行、プレーンテキストに均す。改行はプロンプト規約で禁じられている。"""
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def clip_words(text: str, limit: int) -> str:
    """50語で切る。日本語は空白で割れないので、文単位で落とす。"""
    if len(text.split()) <= limit and len(text) <= 240:
        return text
    sentences = re.split(r"(?<=[。.!?])", text)
    out = ""
    for sentence in sentences:
        candidate = (out + sentence).strip()
        if len(candidate.split()) > limit or len(candidate) > 240:
            break
        out = candidate
    return out or text[:240]


def context_from_messages(payload: dict[str, Any]) -> str:
    """OpenAI 形式の messages から会話テキストを取り出す。

    moshi-rag は system + user の2通で送ってくる。user 側の末尾に、
    プロンプト雛形に続けて会話履歴が連結されている。
    """
    chunks: list[str] = []
    for message in payload.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
    return "\n".join(chunks)


class Handler(BaseHTTPRequestHandler):
    table: ScrewTable
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler の規約)
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": "table", "object": "model"}]})
        else:
            self._send(404, {"error": {"message": f"not found: {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": f"not found: {self.path}"}})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            self._send(400, {"error": {"message": f"invalid json: {error}"}})
            return

        context = context_from_messages(payload)
        reference = self.table.reference_for(context)
        # 引けなければ空文字。moshi-rag 側は [RET_FAILED] として扱う。
        content = f"Reference: {reference}" if reference else ""
        logger.info("context=%d chars -> %s", len(context), reference or "(no hit)")

        self._send(
            200,
            {
                "id": f"screwtable-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model") or "table",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(fmt, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--lang", choices=sorted(TABLES), default="en",
                        help="既定の表。MoshiRAG の重みは英語なので en")
    parser.add_argument("--knowledge", type=Path, default=None)
    parser.add_argument("--glossary", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY,
                        help="読み表（tts_pronunciations）の取得元")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    default_knowledge, default_glossary = TABLES[args.lang]
    knowledge_path = args.knowledge or default_knowledge
    glossary_path = args.glossary or default_glossary
    table = ScrewTable(
        load_rows(knowledge_path),
        load_rows(glossary_path),
        load_readings(args.policy),
        source_label="出典" if args.lang == "ja" else "Source",
    )
    logger.info(
        "loaded %d knowledge rows, %d glossary terms from %s",
        len(table.knowledge),
        len(table.glossary),
        knowledge_path,
    )
    Handler.table = table
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("listening on http://%s:%d/v1/chat/completions", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
