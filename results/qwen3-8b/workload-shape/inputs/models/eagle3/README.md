---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- neuralmagic
- redhat
- speculators
- eagle3
---

# Qwen3-8B-speculator.eagle3

## Model Overview
- **Verifier:** Qwen/Qwen3-8B
- **Speculative Decoding Algorithm:** EAGLE-3
- **Model Architecture:** Eagle3Speculator
- **Release Date:** 07/27/2025
- **Version:** 1.0
- **Model Developers:** RedHat

This is a speculator model designed for use with [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), based on the [EAGLE-3](https://arxiv.org/abs/2503.01840) speculative decoding algorithm.
It was trained using the [speculators](https://github.com/neuralmagic/speculators) library on a combination of the [Aeala/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered) and the [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) datasets.
The model was trained with thinking turned disabled.
This model should be used with the [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) chat template, specifically through the `/chat/completions` endpoint.

## Use with vLLM

```bash
vllm serve Qwen/Qwen3-8B \
  -tp 1 \
  --speculative-config '{
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3,
    "method": "eagle3"
  }'
```

## Evaluations

<h3>Use cases</h3>
<table>
  <thead>
    <tr>
      <th>Use Case</th>
      <th>Dataset</th>
      <th>Number of Samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Coding</td>
      <td>HumanEval</td>
      <td>168</td>
    </tr>
    <tr>
      <td>Math Reasoning</td>
      <td>gsm8k</td>
      <td>80</td>
    </tr>
    <tr>
      <td>Text Summarization</td>
      <td>CNN/Daily Mail</td>
      <td>80</td>
    </tr>
  </tbody>
</table>

<h3>Acceptance lengths</h3>
<table>
  <thead>
    <tr>
      <th>Use Case</th>
      <th>k=1</th>
      <th>k=2</th>
      <th>k=3</th>
      <th>k=4</th>
      <th>k=5</th>
      <th>k=6</th>
      <th>k=7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Coding</td>
      <td>1.72</td>
      <td>2.17</td>
      <td>2.39</td>
      <td>2.59</td>
      <td>2.60</td>
      <td>2.59</td>
      <td>2.69</td>
    </tr>
    <tr>
      <td>Math Reasoning</td>
      <td>1.73</td>
      <td>2.20</td>
      <td>2.48</td>
      <td>2.63</td>
      <td>2.72</td>
      <td>2.79</td>
      <td>2.81</td>
    </tr>
    <tr>
      <td>Text Summarization</td>
      <td>1.62</td>
      <td>1.96</td>
      <td>2.13</td>
      <td>2.24</td>
      <td>2.25</td>
      <td>2.29</td>
      <td>2.30</td>
    </tr>
  </tbody>
</table>

<h3>Performance benchmarking (1xA100)</h3>
<div style="display: flex; justify-content: center; gap: 20px;">

  <figure style="text-align: center;">
    <img src="assets/Qwen3-8B-HumanEval.png" alt="Coding" width="100%">
  </figure>

  <figure style="text-align: center;">
    <img src="assets/Qwen3-8B-math_reasoning.png" alt="Math Reasoning" width="100%">
  </figure>

  <figure style="text-align: center;">
    <img src="assets/Qwen3-8B-summarization.png" alt="Text Summarization" width="100%">
  </figure>
</div>

<details> <summary>Details</summary>
<strong>Configuration</strong>

- temperature: 0.6
- top_p: 0.95
- top_k: 20
- repetitions: 3
- time per experiment: 10min
- hardware: 1xA100
- vLLM version: 0.11.0
- GuideLLM version: 0.3.0

<strong>Command</strong>
```bash
GUIDELLM__PREFERRED_ROUTE="chat_completions" \
guidellm benchmark \
  --target "http://localhost:8000/v1" \
  --data "RedHatAI/speculator_benchmarks" \
  --data-args '{"data_files": "HumanEval.jsonl"}' \
  --rate-type sweep \
  --max-seconds 600 \
  --output-path "Qwen3-8B-HumanEval.json" \
  --backend-args '{"extra_body": {"chat_completions": {"temperature":0.6, "top_p":0.95, "top_k":20}}}'
```
GuideLLM interface changed, so for compatibility with the latest version (v0.6.0), please use the following command:
```bash
GUIDELLM__PREFERRED_ROUTE="chat_completions" \
guidellm benchmark \
  --target "http://localhost:8000/v1" \
  --data "RedHatAI/speculator_benchmarks" \
  --data-args '{"data_files": "HumanEval.jsonl"}' \
  --profile sweep \
  --max-seconds 1800 \
  --output-path "my_output.json" \
  --backend-args '{"extras": {"body": {"temperature":0.6, "top_p":0.95, "top_k":20}}}'
```
</details>