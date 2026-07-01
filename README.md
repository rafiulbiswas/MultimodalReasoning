# MultimodalReasoning
ImageCLEF 2026 — Multimodal Reasoning OpenQA Visual Model: Qwen3-VL-8B-Instruct (9B params, BF16) Inference: vLLM Hardware: NUQ JupyterLab, 2× A100-80GB Task: Image contains an exam question. Generate a free-form answer in the question's language.

Why Qwen3-VL-8B?
Native 256K context, 32-language OCR (covers all 6 dataset languages well)
Improved STEM/math reasoning — directly helps the math-heavy dev split (22.5%)
DeepStack ViT fusion — sharper layout/diagram understanding than Qwen2.5-VL
9B params in BF16 ≈ 18 GB — fits on 1× A100-80GB, with TP=2 you have plenty of headroom
Phase map
GPU check & paths
Install dependencies
Download dataset
Data exploration
Utilities (normalization, answer extraction)
Multilingual prompts
Load Qwen3-VL-8B with vLLM
Inference — zero-shot CoT baseline
Evaluate on dev (multi-metric, per-slice)
Self-consistency
Subject-specialized prompts
Build final submission
