# ImageCLEF 2026 Multimodal Reasoning — Visual OpenQA (Qwen3-VL-8B)

MarsadLab's system for the **Visual Open Question Answering** subtask of the
ImageCLEF 2026 Multimodal Reasoning task. Each item is an image containing
an exam question (with an optional diagram, table, or graph) in one of six
languages; the system produces a short free-form textual answer in the
question's language.

Task description and rules from the organizers:
- <https://mbzuai-nlp.github.io/ImageCLEF-MultimodalReasoning/2026/>
- <https://www.imageclef.org/2026/multimodalreasoning>

- **Model:** [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (9B params, BF16)
- **Inference engine:** vLLM (zero-shot / self-consistency prompting, no fine-tuning)
- **Languages:** Bulgarian, Chinese, Croatian, English, Italian, Serbian
- **Hardware used:** 2× NVIDIA A100-80GB (single GPU also supported)
- **Dataset:** [`SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual`](https://huggingface.co/datasets/SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual) (`train` / `dev` / `test` splits)

## Approach

Three decoding strategies are compared on the labeled `dev` split, and the
best one is used for the final `test` submission:

1. **Zero-shot CoT** — single sample, `T=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`.
2. **Self-consistency** — 5 sampled chains, majority vote over normalized answers.
3. **Subject-specialized prompts** — zero-shot CoT plus a one-line hint about the expected answer form (e.g. "chemical formula" for Chemistry).

Answers are extracted from `<answer>...</answer>` tags (with regex and
last-line fallbacks) and normalized (LaTeX folding, case, punctuation)
before scoring or voting.

## Notebook structure

| Phase | What it does |
|---|---|
| 1–2 | GPU check, paths, install dependencies |
| 3 | Download `train` / `dev` / `test` from the HF repo |
| 4 | Data exploration (language / subject / graphical-type distributions) |
| 5 | Answer normalization and `<answer>` extraction utilities |
| 6 | Native-language prompts (system + user) for all six languages, subject hints |
| 7 | Load Qwen3-VL-8B-Instruct via vLLM |
| 8 | Zero-shot CoT baseline on `dev` |
| 9 | Evaluation — EM, norm-EM, token-F1, "contains", optional BERTScore; reported overall and per language / subject / graphical type |
| 10 | Self-consistency (n=5) on `dev` |
| 11 | Subject-specialized prompts on `dev` |
| 12 | Run best strategy on `test`, validate, write submission JSON |

## Setup

```bash
pip install "vllm>=0.11.0" "transformers>=4.57.0" accelerate datasets \
    huggingface-hub "qwen-vl-utils[decord]>=0.0.10" pillow pyarrow pandas \
    numpy tqdm bert-score regex openai
```

Requires `transformers>=4.57` for `Qwen3VLForConditionalGeneration` and
`vllm>=0.10` for Qwen3-VL support. If you run a different stack than what's
pinned above (this has also been verified against `transformers 5.12.1` /
`vllm 0.24.0`), update the pins to match what you actually used.

## Loading the model

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,   # 2 if ≥2 GPUs available, else 1
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=16384,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1},
    mm_processor_kwargs={
        "min_pixels": 256 * 32 * 32,
        "max_pixels": 1280 * 32 * 32,   # raise for denser text, e.g. Cyrillic pages
    },
)
```

## Submission format

`{question_id, answers: [str], language}` per item, validated for no
duplicate ids, full test-set coverage, and non-empty `answers` (falls back
to `[""]`), written to `outputs/submission_test.json`.

## Known gotchas

- **`GLIBCXX_3.4.29 not found` on model load** — a `pyzmq` / system
  `libstdc++` mismatch in vLLM's subprocess, not a notebook bug. Try
  `pip install --force-reinstall --no-cache-dir pyzmq` and restart the
  kernel; otherwise update `libstdc++` via `conda-forge`.
- **Engine core fails to start with TP=2** — usually an NCCL/peer-to-peer
  issue between GPUs. Retry with `tensor_parallel_size=1` to isolate; an 8B
  BF16 model fits on a single A100-80GB.
- **Duplicate `pip install` cells** — an early cell pins `vllm==0.7.3`
  against `cu121`; a later cell reinstalls `vllm>=0.11.0` against `cu128`.
  Only run the later one on a fresh environment.
