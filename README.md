# Qwen3-0.6-MS

**English** | [中文](README_cn.md)

### Project Overview

Qwen3-0.6-MS is a Chinese role-play QA (Minglan) fine-tuning and inference example project based on Qwen/Qwen3-0.6B, including:

- Data processing and full-parameter fine-tuning script: `train.py`
- LoRA-based efficient fine-tuning script: `train_lora.py`
- Flask multi-turn chat Web app: `app.py` (chat with Minglan in the browser, with history and clear)

### Requirements

- OS: Linux / Windows / macOS
- Python ≥ 3.10
- GPU (recommended, CUDA-capable), or CPU (training will be very slow)

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `transformers`, `modelscope`, `peft`, `pandas`.

Training scripts additionally need `datasets` (not included in requirements.txt, install separately):

```bash
pip install datasets
```

The Web app additionally needs (recommended):

```bash
pip install flask
```

### Directory Structure (key files)

- `train.py`: full-parameter fine-tuning (saves to `./output/final_model`)
- `train_lora.py`: LoRA fine-tuning (saves to `./output_lora/final_model`)
- `app.py`: loads base model + LoRA adapter, provides Web inference
- `templates/`: Web templates (`index.html`)
- `data/minglan_qa_clean.json`: training data (JSON array, 1154 items, each with `问题` and `明兰` fields)
- `data/minglan_qa_clean.csv`: same content as CSV source (training reads the JSON)
- `model/`: base model cache directory (auto-downloaded from ModelScope on first run)
- After training: `train.json` / `eval.json` (split data), `train_format.json` / `eval_format.json` (instruction format), `./processed_train_dataset` / `./processed_eval_dataset` (tokenize cache), `./output/` / `./output_lora/` (training outputs)

### Dataset

The training scripts load data from `data/minglan_qa_clean.json` (1154 items), shuffle with a fixed random seed (`random.seed(50)`), split 9:1 into train/validation, and convert to instruction pairs:

```python
with open(data_file, 'r', encoding='utf-8') as f:
    ds = json.load(f)
```

Field mapping (in `dataset_jsonl_transfer`):

```python
{
    'instruction': '你是盛明兰，以明兰的口吻和性格回答问题',
    'input': data['问题'],
    'output': data['明兰'],
}
```

To replace the dataset: prepare a JSON array file with `问题` / `明兰` fields, and modify `data_file` and the field names in `train.py` / `train_lora.py`.

### Fine-tuning Workflow

#### A. Full-parameter fine-tuning (train.py)

1. Run:

```bash
python train.py
```

2. Main flow:

- Auto-downloads base model `Qwen/Qwen3-0.6B` to `./model/` cache
- Converts raw data to instruction format and tokenizes to `MAX_LENGTH=4096`
- Trains with `Trainer`
- Saves automatically to `./output/final_model`

3. Key hyperparameters (editable in the script):

- Learning rate: `5e-5`
- Batch: `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`
- Epoch: `num_train_epochs=1` (example value)
- Eval/save steps: `eval_steps=100`, `save_steps=400`

#### B. LoRA fine-tuning (train_lora.py)

1. Run:

```bash
python train_lora.py
```

2. Main flow:

- Same download and loading of `Qwen/Qwen3-0.6B`
- Injects LoRA:
  - `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
  - `r=8, lora_alpha=32, lora_dropout=0.1`
- Trains the LoRA adapter with `Trainer`
- Saves to `./output_lora/final_model`
- Eval/save steps: `eval_steps=100`, `save_steps=100`

3. At inference, the adapter is loaded onto the base model via `PeftModel.from_pretrained` (see next section).

### Inference & Web App

`app.py` provides a browser chat application for talking with Minglan (multi-turn).

Run `python train_lora.py` first to generate `./output_lora/final_model`, otherwise startup reports "模型加载失败" (model load failure) and cannot reply.

1. Path configuration:
   In `app.py`'s `load_model()`, default paths are aligned with the training scripts:

```python
model_path = "./model/Qwen/Qwen3-0.6B"       # adjust to your actual cache directory
lora_path = "./output_lora/final_model"      # aligned with train_lora.py
```

2. Run the Web app:

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in a browser.

3. Features:

- Chat page for multi-turn conversation with Minglan (Enter to send, full history kept)
- `/chat` API returns Minglan's reply, `/clear` resets the history
- Model load status shown at the top
- Page copy and system prompt both follow the *Zhifou* Minglan character

4. Inference details:

- Uses `tokenizer.apply_chat_template` to build the conversation input (`enable_thinking=False`, disabling Qwen3 thinking mode)
- System prompt identical to training: `你是盛明兰，以明兰的口吻和性格回答问题`
- Generation parameters: `max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True`; only newly generated tokens are decoded
- LoRA adapter loaded via `PeftModel.from_pretrained`

### FAQ

- Model or LoRA path mismatch: adjust `model_path`/`lora_path` in `app.py` to your actual download/save paths. Note that different ModelScope versions use different cache directory names (new: `./model/Qwen/Qwen3-0.6B`, old: `./model/Qwen/Qwen3-0___6B`).
- `app.py` reports "模型加载失败": usually the LoRA hasn't been trained (missing `./output_lora/final_model`) or paths are wrong — run `python train_lora.py` first and check `model_path`/`lora_path`.
- No GPU: inference works on CPU but is slow; training strongly requires a GPU.
- Data processing cache: tokenized data is cached in `./processed_train_dataset` and `./processed_eval_dataset`; if you change the dataset or fields, delete these two cache directories and rerun (`train_format.json` / `eval_format.json` regenerate automatically).

### Extensions

#### 1. How to use the merged (full fine-tuned) model

`./output/final_model` currently holds the complete model weights (because this is full-parameter fine-tuning, not LoRA).
To export it as a standard model directly usable with vLLM or transformers inference, simply run:

```python
# Load and save is all you need
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./output/final_model")
tokenizer = AutoTokenizer.from_pretrained("./output/final_model")
model.save_pretrained("./exported_model")
tokenizer.save_pretrained("./exported_model")
```

#### 2. Limitations of the 0.6B model (manage expectations)

The model's tone is right now, but as you may have noticed, given a long background description (the first question), it only replied "侯爷可真是见到了真金不怕火炼啊！" — this shows the 0.6B model's context understanding is very weak. It is like "an actor who only memorizes lines." If future business requires complex reasoning or long multi-turn conversations, consider switching to Qwen2.5-7B-Instruct. Your current code structure (format alignment, anti-crash parameters, inference logic) can migrate seamlessly to the 7B model, and the results will take off.

#### 3. Record your training milestones

You can now record the parameters that worked — this is a major milestone for this project:

- Model: Qwen3-0.6B
- Fine-tuning method: Full Fine-tuning
- Anti-crash trio: FP32 loading + lr=1e-5 + max_grad_norm=1.0
- Data format: apply_chat_template + EOS (end-of-sequence token)
- Hardware: single RTX 3090, about 3.8 minutes / 3 epochs

#### 4. Three typical problems and their root causes

**(1) Becoming a "repeater" in an infinite loop**

Symptom: the model outputs "北疆的兵种都齐了" a dozen times, or endlessly repeats the user's question.

Root causes:
- Wrong inference parameters: using `max_length=2048` (which counts the input) leaves too little room for generation and forces truncation; `repetition_penalty` is missing.
- Missing EOS token: the training data concatenated the answer without `tokenizer.eos_token_id`, so the model never learns when to stop.
- Prompt format mismatch: hand-written `<|im_start|>` strings during training vs. the official `apply_chat_template` during inference — tiny differences crash a small model.

**(2) Gradient explosion during full fine-tuning (Loss becomes NaN)**

Symptom: `grad_norm: nan` in the logs, followed by `loss: 0.0`, and finally `CUDA error: device-side assert triggered` at inference.

Root causes:
- Learning rate too high: for full-parameter updates on a 0.6B model, `5e-5` or `1e-4` is too aggressive and causes explosion.
- Precision overflow: fp16 mixed precision on a 3090 — extremes of a small model easily exceed the FP16 range (max 65504) and become inf/nan.
- No gradient clipping: without `max_grad_norm=1.0`, abnormal gradients destroy the weights.
- Dirty data: empty answers or extremely long garbled text produce huge instantaneous losses.

**(3) Dried-up answers (like a soulless machine)**

Symptom: you send a long paragraph, and the model replies with one sentence like "那可真是难办啊！".

Root causes:
- Not enough diversity at inference: `temperature=0.3` is too conservative and `repetition_penalty=1.1` over-penalizes.
- Training data itself is short: if your qa.csv answers are all short, the model never learns long sentences.
- Weak System Prompt: you never explicitly required it to "say more".

#### 5. Lessons learned after the pitfalls

- Beware "format mismatches": for small models (like 0.6B), tolerance is extremely low. Training-time data concatenation must match inference exactly — always use the official `tokenizer.apply_chat_template()`.
- The "gradient checkpointing" trap: `gradient_checkpointing=True` saves memory but easily breaks requires_grad flow. For a tiny model like 0.6B, turn it off (set to False) for ~30% faster training and higher stability.
- Don't let caches fool you: after changing `process_fun` or the Prompt format, always delete the old `processed_dataset` cache folders, otherwise the framework silently reads stale data.
- Full fine-tuning is far more fragile than LoRA: LoRA only touches a few parameters and is more robust; full fine-tuning exposes every parameter to gradients, so tread carefully.

#### 6. Hyperparameter SOP ("stay rock-solid" recipe)

**Anti-NaN trio**
- Precision: if the GPU supports it (3090/4090/A100), prefer `bf16=True, fp16=False`. bf16 has a huge dynamic range and almost never overflows. Otherwise train in `torch.float32` (if memory allows) — absolutely safe.
- Learning rate: start at `1e-5` or `2e-5` for full fine-tuning; `1e-4` or `2e-4` for LoRA. When in doubt, go lower.
- Gradient clipping: always add `max_grad_norm=1.0` — the fuse that saves you.

**Three locks against the "repeater"**
- Training: append `[tokenizer.eos_token_id]` to the end of labels.
- Inference: never use `max_length`; use `max_new_tokens=256`.
- Sampling: add `repetition_penalty=1.1` together with `no_repeat_ngram_size=3`.

**Two tricks to make answers "fuller"**
- Prompt constraint: hard-code in the System Prompt: "Please answer in detail in the voice of XX, at least 50 characters."
- Open up inference: if it sounds too stiff, raise temperature to `0.6–0.7` and top_p to `0.85–0.9` for more "human warmth".
- Data sets the ceiling: the model is a parrot. Over 30% of your dataset should be long-paragraph answers, otherwise it will only ever produce one-liners.

**Golden config for VRAM & speed (single RTX 3090 24GB)**

```python
# Golden parameter template
args = TrainingArguments(
    per_device_train_batch_size=4,   # 4 for full FT, 8 for LoRA
    gradient_accumulation_steps=2,   # accumulate to an equivalent batch
    num_train_epochs=3,              # 3 epochs is enough for ~1000 samples
    learning_rate=1e-5,              # full FT, anti-explosion
    lr_scheduler_type="cosine",      # cosine annealing for smooth convergence
    warmup_ratio=0.05,               # 5% warmup steps
    max_grad_norm=1.0,               # gradient clipping — the lifeline
    bf16=True,                       # must-enable on 3090
    gradient_checkpointing=False,    # off for small models, on for large ones
)
```

#### 7. Final thoughts

Model fine-tuning is not only a science but also a blend of "dark art" and engineering. Taking a 0.6B model from "gibberish" to the "Zhifou style" means hitting nearly every beginner pitfall at least once. With these lessons — Loss monitoring, NaN debugging, data-format alignment, inference constraints — the same underlying logic applies when you move on to 7B, 14B, or even 72B models.

### License & Disclaimer

- This project is for learning and research only; generated Minglan role-play content is for entertainment purposes only and is not affiliated with the original TV drama *The Story of Minglan*.
- Please follow the licenses of the upstream model (Qwen) and data sources.
