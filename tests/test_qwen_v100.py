from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from scripts.check_qwen_v100 import runtime_errors, validate_chat_response


REPO = Path(__file__).resolve().parents[1]
BASH = Path(r"C:\Program Files\Git\bin\bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing")


def fake_torch(count=4, memory=32, cuda_version="12.6", arch="sm_70", name="Tesla V100-SXM2-32GB"):
    return SimpleNamespace(version=SimpleNamespace(cuda=cuda_version), cuda=SimpleNamespace(
        get_arch_list=lambda: [arch], is_available=lambda: True, device_count=lambda: count,
        get_device_properties=lambda i: SimpleNamespace(name=name, major=7, minor=0,
                                                        total_memory=memory * 1024**3)))


def chat_response(content='{"message":"こんにちは"}', finish="stop", **kwargs):
    return {"choices": [{"finish_reason": finish, "message": {"content": content, **kwargs}}]}


class PreflightTests(unittest.TestCase):
    def test_four_v100_32gb_pass(self):
        self.assertEqual(runtime_errors(fake_torch(), "0.21.0+cu126"), [])

    def test_wrong_cuda_and_missing_volta_kernels_rejected(self):
        errors = runtime_errors(fake_torch(cuda_version="12.9", arch="sm_80"), "0.21.0")
        self.assertTrue(any("CUDA" in e for e in errors))
        self.assertTrue(any("sm_70" in e for e in errors))

    def test_gpu_allocation_and_capacity_rejected(self):
        errors = runtime_errors(fake_torch(count=1, memory=16), "0.21.0")
        self.assertTrue(any("4 visible" in e for e in errors))
        self.assertTrue(any("32GB" in e for e in errors))

    def test_non_v100_and_new_vllm_rejected(self):
        errors = runtime_errors(fake_torch(name="A100"), "0.27.0")
        self.assertTrue(any("must be a V100" in e for e in errors))
        self.assertTrue(any("vLLM must" in e for e in errors))

    def test_japanese_json_passes(self):
        validate_chat_response(chat_response())

    def test_truncation_reasoning_and_bad_output_rejected(self):
        for response in (chat_response(finish="length"), chat_response(reasoning_content="thinking"),
                         chat_response(content="<think>reason</think>"),
                         chat_response(content='{"message":"hello"}')):
            with self.subTest(response=response), self.assertRaises((ValueError, json.JSONDecodeError)):
                validate_chat_response(response)


@unittest.skipUnless(BASH.is_file(), "bash unavailable")
class RunnerTests(unittest.TestCase):
    def write(self, root, relative, text):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def shell(self, root, command, **overrides):
        env = {**os.environ, "PBS_JOBID": "test.0", **overrides}
        env.pop("PBS_O_WORKDIR", None)
        return subprocess.run([str(BASH), "-c", command], cwd=root, env=env,
                              capture_output=True, text=True, encoding="utf-8", timeout=25)

    def test_profile_uses_separate_environment_and_four_gpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, "scripts/run_dialogues_qwen_v100_4gpu.pbs",
                       (REPO / "scripts/run_dialogues_qwen_v100_4gpu.pbs").read_text(encoding="utf-8"))
            self.write(root, "scripts/setup_vllm_qwen_v100_env.sh", '#!/bin/bash\nprintf "setup\\n" >> order\n')
            self.write(root, "scripts/run_dialogues_qwen_3000.pbs", '''#!/bin/bash
printf '%s\n' "$VLLM_ENV_DIR" "$VLLM_DTYPE" "$VLLM_TENSOR_PARALLEL_SIZE" "$CUDA_VISIBLE_DEVICES" "$VLLM_SKIP_INSTALL" "$BATCH_ID" "$VLLM_ATTENTION_BACKEND" "$VLLM_QWEN_CHAT_TEMPLATE_KWARGS" > profile
printf 'runner\n' >> order
''')
            result = self.shell(root, "bash scripts/run_dialogues_qwen_v100_4gpu.pbs", NUM_CASES="20",
                                CUDA_VISIBLE_DEVICES="0", VLLM_ENV_DIR="/a100-env")
            self.assertEqual(result.returncode, 0, result.stderr)
            profile = (root / "profile").read_text(encoding="utf-8").splitlines()
            self.assertTrue(profile[0].endswith("/.venv-vllm-qwen-v100"))
            self.assertEqual(profile[1:5], ["float16", "4", "0,1,2,3", "1"])
            self.assertEqual(profile[5], "qwen_dialogues_v100_4gpu_20_v1")
            self.assertEqual(profile[6], "TRITON_ATTN")
            self.assertFalse(json.loads(profile[7])["enable_thinking"])
            self.assertEqual((root / "order").read_text().splitlines(), ["setup", "runner"])

    def prepare_runner(self, root):
        for relative in ("scripts/run_dialogues_qwen_3000.pbs", "scripts/run_full_duplex_training_data.sh"):
            self.write(root, relative, (REPO / relative).read_text(encoding="utf-8"))
        self.write(root, "scripts/run_id_utils.sh", "#!/bin/bash\n")
        self.write(root, "runtime/bin/python", '''#!/bin/bash
printf '%s\n' "$*" >> checks
if [[ "$*" == *" runtime" && "${FAIL_RUNTIME:-0}" == 1 ]]; then exit 7; fi
if [[ "$2" == chat && "${FAIL_CHAT:-0}" == 1 ]]; then exit 8; fi
''')
        self.write(root, "runtime/bin/vllm", '''#!/bin/bash
printf '%s\n' "$@" > server.argv
exec sleep 60
''')
        self.write(root, "mocks/curl", '#!/bin/bash\nprintf \'{"data":[{"id":"Qwen/Qwen3.6-27B"}]}\\n\'\n')
        self.write(root, "mocks/uv", '''#!/bin/bash
if [[ "$1" != run ]]; then
    if [[ "${ALLOW_INSTALL:-0}" != 1 ]]; then echo 'unexpected install' >&2; exit 9; fi
    printf '%s\n' "$*" >> installs
    exit 0
fi
printf '%s\n' "$*" >> generation
mkdir -p "$OUT_ROOT/llm_dialogues"
printf '{}\n' > "$OUT_ROOT/llm_dialogues/dialogues.jsonl"
''')

    def run_common(self, root, **overrides):
        return self.shell(root, 'export PATH="$PWD/mocks:$PATH"; bash scripts/run_dialogues_qwen_3000.pbs',
                          VLLM_ENV_DIR="runtime", VLLM_SKIP_INSTALL="1", QWEN_V100_PREFLIGHT="1",
                          VLLM_TENSOR_PARALLEL_SIZE="4", VLLM_DTYPE="float16",
                          VLLM_ATTENTION_BACKEND="TRITON_ATTN", VLLM_LANGUAGE_MODEL_ONLY="1",
                          VLLM_ENABLE_CHUNKED_PREFILL="1", VLLM_MAX_NUM_SEQS="8",
                          VLLM_MAX_NUM_BATCHED_TOKENS="1024", VLLM_ENFORCE_EAGER="1",
                          LLM_OPENAI_ENDPOINT="chat", BATCH_ID="test_v100", NUM_CASES="2",
                          PORT="18001", VLLM_CUDA_MODULE="", **overrides)

    def test_common_runner_reuses_runtime_and_passes_gpu_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare_runner(root)
            result = self.run_common(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            argv = (root / "server.argv").read_text().splitlines()
            self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "4")
            self.assertEqual(argv[argv.index("--dtype") + 1], "float16")
            self.assertEqual(argv[argv.index("--attention-backend") + 1], "TRITON_ATTN")
            self.assertIn("--language-model-only", argv)
            self.assertIn("--enable-chunked-prefill", argv)
            self.assertIn("--enforce-eager", argv)
            checks = (root / "checks").read_text().splitlines()
            self.assertTrue(checks[0].endswith(" runtime"))
            self.assertIn(" chat ", checks[1])
            self.assertTrue((root / "generation").exists())

    def test_a100_defaults_still_install_and_use_one_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare_runner(root)
            result = self.shell(root, 'export PATH="$PWD/mocks:$PATH"; bash scripts/run_dialogues_qwen_3000.pbs',
                                VLLM_ENV_DIR="runtime", VLLM_CUDA_MODULE="", ALLOW_INSTALL="1",
                                BATCH_ID="test_a100", LLM_OPENAI_ENDPOINT="chat", PORT="18002")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            argv = (root / "server.argv").read_text().splitlines()
            self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "1")
            self.assertEqual(argv[argv.index("--dtype") + 1], "auto")
            self.assertNotIn("--attention-backend", argv)
            self.assertNotIn("--language-model-only", argv)
            self.assertFalse((root / "checks").exists())
            self.assertEqual(len((root / "installs").read_text().splitlines()), 2)

    def test_failed_runtime_never_starts_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare_runner(root)
            result = self.run_common(root, FAIL_RUNTIME="1")
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            self.assertFalse((root / "server.argv").exists())
            self.assertFalse((root / "generation").exists())

    def test_failed_json_probe_never_generates_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare_runner(root)
            result = self.run_common(root, FAIL_CHAT="1")
            self.assertEqual(result.returncode, 8, result.stdout + result.stderr)
            self.assertFalse((root / "generation").exists())


if __name__ == "__main__":
    unittest.main()
