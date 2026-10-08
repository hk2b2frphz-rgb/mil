#!/usr/bin/env python3
"""Paired real-v2 ablations. Standard library only; no synthetic AI fallback."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import urllib.request

ARMS = ("real_v2", "traditional_overlap", "ai_placement")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_config(path, config):
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError(f"Configuration changed; use a new comparison directory: {path}")
    write_json(path, config)


def is_backchannel(turn):
    return turn.get("speaker") == "moshi" and turn.get("event") == "model_backchannel"


def fixed_content(row):
    """Ignore only backchannels and user segmentation; retain silence and probes."""
    out = []
    for turn in row["turns"]:
        if is_backchannel(turn):
            continue
        turn = {k: v for k, v in turn.items() if v is not None}
        if turn.get("speaker") == "user":
            if out and out[-1].get("speaker") == "user":
                out[-1]["text"] += turn.get("text", "")
            else:
                out.append({"speaker": "user", "text": turn.get("text", "")})
        else:
            out.append(turn)
    return out


def freeze(source, out, count, require_continuous=False):
    # Use the existing greeting detector, exactly as the bank rendering route does.
    from rewrite_aizuchi_vocab import is_greeting
    rows = read_rows(source)[:count]
    if len(rows) != count or len({r["id"] for r in rows}) != count:
        raise ValueError(f"Need {count} unique source dialogues")
    for row in rows:
        if require_continuous:
            protocol = row.get("listener_protocol") or {}
            if (protocol.get("version") != "continuous_listener_v1"
                    or protocol.get("greeting") is not False
                    or protocol.get("ending") != "ongoing_excerpt"
                    or protocol.get("min_blocks", 0) >= protocol.get("max_blocks", 0)):
                raise ValueError(f"Need continuous listener data without greetings/endings and with variable lengths: {row['id']}")
        if not re.fullmatch(r"density=(?:0(?:\.\d+)?|1(?:\.0+)?)", row.get("aizuchi_frequency_label", "")):
            raise ValueError(f"Missing real-v2 density label: {row['id']}")
        row["turns"] = [t for t in row["turns"] if not (
            t.get("speaker") == "moshi" and is_greeting(t.get("text", "")))]
    config = {
        "source": str(source.resolve()), "sha256": digest(source), "count": count,
        "drop_greeting": True, "version": 1,
    }
    if require_continuous:
        config["require_continuous"] = True
        config["listener_protocol"] = rows[0]["listener_protocol"]
        if any(row["listener_protocol"] != config["listener_protocol"] for row in rows):
            raise ValueError("Mixed listener generation protocols in source")
    lock_config(out.parent / "source_config.json", config)
    write_rows(out, rows)


def split_after(text, pattern):
    pieces, start = [], 0
    for match in re.finditer(pattern, text):
        pieces.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def clauses(text):
    # Keep every character. Punctuation boundaries correspond to the source
    # text, rather than guessed acoustic timestamps or offsets in generated text.
    return split_after(text, r"[、，。!！?？]+|(?<!\d)[,.](?!\d)")


def sentence_units(turns, minimum_chars=24):
    """Recover sentences; merge short adjacent sentences to avoid rapid replies."""
    def split_run(text):
        sentences = split_after(text, r"[。!！?？]+|(?<!\d)\.(?!\d)")
        units, part = [], ""
        for sentence in sentences:
            part += sentence
            if len(part) >= minimum_chars:
                units.append(part)
                part = ""
        if part:
            if units:
                units[-1] += part
            else:
                units.append(part)
        return units

    pending = ""
    for index, turn in enumerate(turns):
        if is_backchannel(turn):
            continue
        # A user confirmation probe and its existing answer are a fixed pair,
        # not a monologue sentence on which to add another listener response.
        if (turn.get("speaker") == "user" and index + 1 < len(turns)
                and turns[index + 1].get("speaker") == "moshi"
                and not is_backchannel(turns[index + 1])):
            if pending:
                yield from split_run(pending)
                pending = ""
            yield deepcopy(turn)
            continue
        if turn.get("speaker") != "user":
            if pending:
                yield from split_run(pending)
                pending = ""
            yield deepcopy(turn)
            continue
        pending += turn.get("text", "")
    if pending:
        yield from split_run(pending)


def reaction_cap(density, parts, maximum):
    return 0 if density == 0 else min(len(parts), maximum, max(1, math.ceil(density * maximum)))


def validate_reactions(raw, parts, vocab, cap, min_chars, previous_words, previous_reaction=False):
    try:
        # Accept a fenced JSON response, but never substitute random placements.
        data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        items = data["reactions"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Return a JSON object with reactions") from exc
    if not isinstance(items, list) or not 1 <= len(items) <= cap:
        raise ValueError(f"Use 1..{cap} reactions")
    out = []
    recent = list(previous_words[-2:])
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each reaction must be an object")
        index, word = item.get("after_clause"), item.get("text")
        if type(index) is not int or not 1 <= index <= len(parts) or word not in vocab:
            raise ValueError("Use an in-range integer boundary and an exact vocabulary entry")
        if out and index <= out[-1]["after_clause"]:
            raise ValueError("Positions must increase strictly; no duplicate positions")
        if word in recent[-2:]:
            raise ValueError("Do not repeat the last two backchannel phrases")
        if not out and previous_reaction and len("".join(parts[:index])) < min_chars:
            raise ValueError(f"Leave at least {min_chars} characters after the previous sentence's reaction")
        if out and len("".join(parts[out[-1]["after_clause"]:index])) < min_chars:
            raise ValueError(f"Leave at least {min_chars} text characters between reactions")
        out.append({"after_clause": index, "text": word})
        recent.append(word)
    if out[-1]["after_clause"] != len(parts):
        raise ValueError("Include a reaction at the end of the sentence")
    return out


def prompt_for(text, parts, vocab, density, cap, min_chars, context):
    return (
        "あなたは相手の話を聞く相槌AIです。発話の内容と会話の流れから、自然に受け止める句の切れ目を選んでください。"
        "質問・助言・言い換えはせず、指定語彙だけを使ってください。\n"
        f"相槌密度の条件は {density:.2f}。この一文は1〜{cap}個を上限とし、上限まで埋める必要はありません。"
        f"最後の句 {len(parts)} には必ず1個置いてください。"
        f"位置を重複させず、相槌の間に少なくとも{min_chars}文字分の話を挟みます。直近2回の相槌と同じ語を使わないでください。\n"
        f"直前の会話（今回より後の発話は含みません）: {json.dumps(context[-12:], ensure_ascii=False)}\n"
        f"今回の一文: {text}\n句番号:\n" + "\n".join(f"{i+1}: {s}" for i, s in enumerate(parts))
        + f"\n許可語彙: {json.dumps(vocab, ensure_ascii=False)}\n"
        + 'JSONだけ返してください: {"reactions":[{"after_clause":1,"text":"許可語彙の一つ"}]}'
    )


def request_llm(prompt):
    base = os.environ.get("LLM_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
    body = {
        "model": os.environ.get("LLM_MODEL", "Qwen/Qwen3.6-27B"),
        "messages": [{"role": "system", "content": "自然な日本語の傾聴相槌をJSONで返す。"},
                     {"role": "user", "content": prompt}],
        "temperature": 0.2, "max_tokens": 800,
        "seed": int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:4], "big") % (2**31),
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {
        "Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("LLM_API_KEY", "EMPTY")})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def reposition(row, vocab, maximum=3, min_chars=24, caller=request_llm):
    out = deepcopy(row)
    out["turns"] = []
    density = float(row["aizuchi_frequency_label"].split("=", 1)[1])
    trace = []
    recent = []
    previous_reaction = False
    for unit in sentence_units(row["turns"], min_chars):
        if isinstance(unit, dict):
            out["turns"].append(unit)
            previous_reaction = False
            continue
        parts = clauses(unit)
        cap = reaction_cap(density, parts, maximum)
        if cap == 0:
            out["turns"].append({"speaker": "user", "text": unit})
            continue
        prompt = prompt_for(unit, parts, vocab, density, cap, min_chars, out["turns"])
        attempts = []
        for attempt in range(3):
            raw = caller(prompt)
            attempts.append({"raw": raw})
            try:
                reactions = validate_reactions(raw, parts, vocab, cap, min_chars, recent, previous_reaction)
                break
            except ValueError as exc:
                attempts[-1]["error"] = str(exc)
                prompt += f"\n前の回答は制約違反でした: {exc}。文末1個だけでもよいので、制約を満たす回答を返してください。"
        else:
            raise ValueError(f"AI placement failed three times: {row['id']}: {unit}: {attempts}")
        start = 0
        for reaction in reactions:
            index = reaction["after_clause"]
            out["turns"].append({"speaker": "user", "text": "".join(parts[start:index])})
            out["turns"].append({"speaker": "moshi", "text": reaction["text"], "event": "model_backchannel"})
            recent.append(reaction["text"])
            start = index
        trace.append({"sentence": unit, "cap": cap, "reactions": reactions, "attempts": attempts})
        previous_reaction = True
    if fixed_content(row) != fixed_content(out):
        raise ValueError(f"AI changed the source conversation: {row['id']}")
    return out, {"id": row["id"], "sentences": trace}


def run_ai(args):
    rows = read_rows(args.source)
    vocab = [line.split("\t")[0].strip() for line in args.vocab.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    if len(set(vocab)) < 3:
        raise ValueError("At least three distinct vocabulary entries are required")
    lock_config(args.out.with_suffix(".config.json"), {
        "source_sha256": digest(args.source), "vocab_sha256": digest(args.vocab),
        "maximum": args.max_per_sentence, "min_chars": args.min_chars,
        "model": os.environ.get("LLM_MODEL", "Qwen/Qwen3.6-27B"), "version": 1,
    })
    done = {r["id"]: r for r in read_rows(args.out)} if args.out.exists() else {}
    source_ids = {r["id"] for r in rows}
    if len(source_ids) != len(rows) or not set(done).issubset(source_ids):
        raise ValueError("Duplicate or foreign dialogue IDs")
    for row in rows:
        if row["id"] in done and fixed_content(row) != fixed_content(done[row["id"]]):
            raise ValueError("Resumed output changed source content")
    pending = [r for r in rows if r["id"] not in done]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for out, trace in pool.map(lambda r: reposition(r, vocab, args.max_per_sentence, args.min_chars), pending):
            # Persist a successful dialogue even if a later request fails.
            with args.out.open("a", encoding="utf-8") as file:
                file.write(json.dumps(out, ensure_ascii=False) + "\n")
            with args.out.with_suffix(".trace.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(trace, ensure_ascii=False) + "\n")
            done[out["id"]] = out
            print(f"[ai-placement] {len(done)}/{len(rows)} {out['id']}", flush=True)
    write_rows(args.out, [done[r["id"]] for r in rows])


def index_manifest(path):
    indexed = {}
    for row in read_rows(path):
        wav = (path.parent / row["path"]).resolve()
        if not wav.is_file():
            raise ValueError(f"Missing audio: {wav}")
        sidecar = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        ident = sidecar["metadata"]["dialogue"]["id"]
        if ident in indexed:
            raise ValueError(f"Duplicate rendered dialogue: {ident}")
        indexed[ident] = {**row, "path": str(wav)}
    return indexed


def assemble(args):
    rows = read_rows(args.source)
    ids = [r["id"] for r in rows]
    manifests = {
        "real_v2": args.root / "real_v2/tts/placement_bank/shard_000_conditioned/training_set/synthetic_moshi_train.jsonl",
        "traditional_overlap": args.root / "traditional_overlap/tts/merged_conditioned/training_set/synthetic_moshi_train.jsonl",
        "ai_placement": args.root / "ai_placement/tts/merged_conditioned/training_set/synthetic_moshi_train.jsonl",
    }
    indexed = {arm: index_manifest(path) for arm, path in manifests.items()}
    # Fail instead of silently selecting different successful dialogues per arm.
    for arm, items in indexed.items():
        if set(items) != set(ids):
            raise ValueError(f"{arm}: source/render ID mismatch, missing={len(set(ids)-set(items))}, extra={len(set(items)-set(ids))}")
    config = {"ids": ids, "manifests": {a: digest(p) for a, p in manifests.items()}, "version": 1}
    lock_config(args.root / "paired_config.json", config)
    summary = {"samples_per_arm": len(ids), "source_sha256": digest(args.source), "arms": {}}
    for arm in ARMS:
        # Identical ordering plus the trainer's shared seed gives identical
        # dialogue-level train/eval membership, independent of render durations.
        target = args.root / arm / "paired/training_set/synthetic_moshi_train.jsonl"
        write_rows(target, [indexed[arm][ident] for ident in ids])
        summary["arms"][arm] = {"manifest": str(target), "hours": sum(float(r["duration"]) for r in indexed[arm].values())/3600}
    write_json(args.root / "paired_summary.json", summary)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--require-continuous", action="store_true")
    p = sub.add_parser("ai")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-per-sentence", type=int, default=3)
    p.add_argument("--min-chars", type=int, default=24)
    p = sub.add_parser("assemble")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.count < 2:
            parser.error("count must be at least 2")
        freeze(args.source, args.out, args.count, args.require_continuous)
    elif args.stage == "ai":
        if args.concurrency < 1 or args.max_per_sentence < 1 or args.min_chars < 1:
            parser.error("AI limits must be positive")
        run_ai(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
