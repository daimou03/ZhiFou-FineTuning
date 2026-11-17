# Qwen3-0.6-MS

[English](README.md) | **中文（简体）**

### 项目简介

Qwen3-0.6-MS 是一个基于 Qwen/Qwen3-0.6B 的中文角色扮演问答（明兰）微调与推理示例项目，包含：

- 数据处理与全参数微调脚本：`train.py`
- 基于 LoRA 的高效微调脚本：`train_lora.py`
- Flask 多轮对话 Web 应用：`app.py`（在浏览器中与明兰聊天，支持历史记录与清空）

### 环境要求

- 操作系统：Linux / Windows / macOS
- Python ≥ 3.10
- GPU（推荐，支持 CUDA），或 CPU（训练会很慢）

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖：`torch`、`transformers`、`modelscope`、`peft`、`pandas`。

训练脚本还需要 `datasets`（未包含在 requirements.txt 中，需单独安装）：

```bash
pip install datasets
```

Web 应用还需要（推荐安装）：

```bash
pip install flask
```

### 目录结构（关键文件）

- `train.py`：全参数微调（保存到 `./output/final_model`）
- `train_lora.py`：LoRA 微调（保存到 `./output_lora/final_model`）
- `app.py`：加载基础模型 + LoRA 适配器，提供 Web 推理
- `templates/`：Web 模板（`index.html`）
- `data/minglan_qa_clean.json`：训练数据（JSON 数组，1154 条，每条含 `问题` 和 `明兰` 字段）
- `data/minglan_qa_clean.csv`：同内容的 CSV 源文件（训练读取的是 JSON）
- `model/`：基础模型缓存目录（首次运行时自动从 ModelScope 下载）
- 训练后生成：`train.json` / `eval.json`（切分数据）、`train_format.json` / `eval_format.json`（指令格式）、`./processed_train_dataset` / `./processed_eval_dataset`（tokenize 缓存）、`./output/` / `./output_lora/`（训练输出）

### 数据集

训练脚本从 `data/minglan_qa_clean.json`（1154 条）读取数据，以固定随机种子打乱（`random.seed(50)`），按 9:1 切分为训练集/验证集，并转换为指令对：

```python
with open(data_file, 'r', encoding='utf-8') as f:
    ds = json.load(f)
```

字段映射（在 `dataset_jsonl_transfer` 中）：

```python
{
    'instruction': '你是盛明兰，以明兰的口吻和性格回答问题',
    'input': data['问题'],
    'output': data['明兰'],
}
```

替换数据集：准备一个含 `问题` / `明兰` 字段的 JSON 数组文件，并修改 `train.py` / `train_lora.py` 中的 `data_file` 和字段名。

### 微调流程

#### A. 全参数微调（train.py）

1. 运行：

```bash
python train.py
```

2. 主流程：

- 自动下载基础模型 `Qwen/Qwen3-0.6B` 到 `./model/` 缓存
- 将原始数据转换为指令格式并 tokenize 到 `MAX_LENGTH=4096`
- 使用 `Trainer` 训练
- 自动保存到 `./output/final_model`

3. 关键超参数（可在脚本中修改）：

- 学习率：`5e-5`
- 批量：`per_device_train_batch_size=1`、`gradient_accumulation_steps=4`
- 轮数：`num_train_epochs=1`（示例值）
- 评估/保存步数：`eval_steps=100`、`save_steps=400`

#### B. LoRA 微调（train_lora.py）

1. 运行：

```bash
python train_lora.py
```

2. 主流程：

- 同样下载并加载 `Qwen/Qwen3-0.6B`
- 注入 LoRA：
  - `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
  - `r=8, lora_alpha=32, lora_dropout=0.1`
- 使用 `Trainer` 训练 LoRA 适配器
- 保存到 `./output_lora/final_model`
- 评估/保存步数：`eval_steps=100`、`save_steps=100`

3. 推理时通过 `PeftModel.from_pretrained` 将适配器加载到基础模型上（见下节）。

### 推理与 Web 应用

`app.py` 提供浏览器聊天应用，可与明兰进行多轮对话。

先运行 `python train_lora.py` 生成 `./output_lora/final_model`，否则启动时报"模型加载失败"且无法回复。

1. 路径配置：
   `app.py` 的 `load_model()` 中默认路径与训练脚本对齐：

```python
model_path = "./model/Qwen/Qwen3-0.6B"       # 按实际缓存目录调整
lora_path = "./output_lora/final_model"      # 与 train_lora.py 对齐
```

2. 启动 Web 应用：

```bash
python app.py
```

在浏览器访问 `http://127.0.0.1:5000/`。

3. 功能：

- 聊天页面支持与明兰多轮对话（Enter 发送，保留完整历史）
- `/chat` 接口返回明兰回复，`/clear` 重置历史
- 页面顶部显示模型加载状态
- 页面文案与 system prompt 均遵循《知否》明兰人设

4. 推理细节：

- 使用 `tokenizer.apply_chat_template` 构造对话输入（`enable_thinking=False`，关闭 Qwen3 思考模式）
- System prompt 与训练一致：`你是盛明兰，以明兰的口吻和性格回答问题`
- 生成参数：`max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True`；仅解码新生成的 token
- LoRA 适配器通过 `PeftModel.from_pretrained` 加载

### 常见问题（FAQ）

- 模型或 LoRA 路径不匹配：将 `app.py` 中的 `model_path`/`lora_path` 调整为实际下载/保存路径。注意不同 ModelScope 版本的缓存目录名不同（新版：`./model/Qwen/Qwen3-0.6B`，旧版：`./model/Qwen/Qwen3-0___6B`）。
- `app.py` 报"模型加载失败"：通常是尚未训练 LoRA（缺少 `./output_lora/final_model`）或路径错误——先运行 `python train_lora.py` 并检查 `model_path`/`lora_path`。
- 没有 GPU：CPU 可以推理但很慢；训练强烈建议使用 GPU。
- 数据处理缓存：tokenize 后的数据缓存在 `./processed_train_dataset` 和 `./processed_eval_dataset`；更换数据集或字段后请删除这两个缓存目录再重跑（`train_format.json` / `eval_format.json` 会自动重新生成）。

### 扩展

#### 1. 如何使用合并后的（全量微调）模型

`./output/final_model` 目前保存的是完整模型权重（因为这是全参数微调，而非 LoRA）。
要导出为可直接用于 vLLM 或 transformers 推理的标准模型，只需：

```python
# 加载再保存即可
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./output/final_model")
tokenizer = AutoTokenizer.from_pretrained("./output/final_model")
model.save_pretrained("./exported_model")
tokenizer.save_pretrained("./exported_model")
```

#### 2. 0.6B 模型的局限（管理预期）

模型的语气现在是对的，但你可能注意到，给出一段较长的背景描述（第一个问题），它只回了一句"侯爷可真是见到了真金不怕火炼啊！"——这说明 0.6B 模型的上下文理解能力非常弱，就像"一个只背台词的演员"。如果后续业务需要复杂推理或长多轮对话，可考虑换成 Qwen2.5-7B-Instruct。你现在的代码结构（格式对齐、防崩溃参数、推理逻辑）可以无缝迁移到 7B 模型，效果会大幅提升。

#### 3. 记录你的训练里程碑

现在可以记录下调通的参数了——这是本项目的重要里程碑：

- 模型：Qwen3-0.6B
- 微调方式：全量微调（Full Fine-tuning）
- 防崩溃三件套：FP32 加载 + lr=1e-5 + max_grad_norm=1.0
- 数据格式：apply_chat_template + EOS（结束符）
- 硬件：单卡 RTX 3090，约 3.8 分钟 / 3 epochs

#### 4. 三个典型问题及其根因

**(1) 变成"复读机"无限循环**

症状：模型把"北疆的兵种都齐了"输出十几遍，或不停重复用户的提问。

根因：
- 推理参数错误：使用 `max_length=2048`（会把输入算进去）导致生成空间不足被强制截断；缺少 `repetition_penalty`。
- 缺失 EOS token：训练数据拼接答案时没有拼 `tokenizer.eos_token_id`，模型永远学不会何时停止。
- Prompt 格式不匹配：训练时手写 `<|im_start|>` 字符串 vs 推理时用官方 `apply_chat_template`——小模型对细微差异零容忍。

**(2) 全量微调梯度爆炸（Loss 变 NaN）**

症状：日志中出现 `grad_norm: nan`，随后 `loss: 0.0`，最后推理时报 `CUDA error: device-side assert triggered`。

根因：
- 学习率过高：0.6B 模型全参数更新用 `5e-5` 或 `1e-4` 太激进，容易爆炸。
- 精度溢出：3090 上开 fp16 混合精度——小模型的极值容易超出 FP16 范围（最大 65504）变成 inf/nan。
- 没有梯度裁剪：缺少 `max_grad_norm=1.0`，异常梯度直接破坏权重。
- 脏数据：空答案或超长乱码会产生巨大的瞬时 loss。

**(3) 回答干瘪（像个没有灵魂的机器）**

症状：你发一大段话，模型只回一句"那可真是难办啊！"。

根因：
- 推理多样性不足：`temperature=0.3` 过于保守，`repetition_penalty=1.1` 惩罚过度。
- 训练数据本身短：如果 qa.csv 的答案都很短，模型永远学不会长句。
- System Prompt 太弱：从未明确要求它"多说点"。

#### 5. 踩坑之后的教训

- 警惕"格式不匹配"：小模型（如 0.6B）容错率极低，训练时的数据拼接必须与推理完全一致——务必使用官方 `tokenizer.apply_chat_template()`。
- "gradient checkpointing" 陷阱：`gradient_checkpointing=True` 省显存但容易打断 requires_grad 传递。对 0.6B 这种小模型关掉（设为 False）反而快约 30% 且更稳。
- 别让缓存骗了你：改动 `process_fun` 或 Prompt 格式后，务必删除旧的 `processed_dataset` 缓存目录，否则框架会悄悄读取旧数据。
- 全量微调远比脆弱：LoRA 只动少量参数更稳健；全量微调每个参数都暴露在梯度下，务必小心。

#### 6. 超参数 SOP（"稳如老狗"配方）

**防 NaN 三件套**
- 精度：GPU 支持的话（3090/4090/A100）优先 `bf16=True, fp16=False`。bf16 动态范围大几乎不溢出。否则用 `torch.float32` 训练（显存允许的话）——绝对安全。
- 学习率：全量微调从 `1e-5` 或 `2e-5` 起步；LoRA 用 `1e-4` 或 `2e-4`。拿不准就往低了调。
- 梯度裁剪：始终加上 `max_grad_norm=1.0`——这是救你命的保险丝。

**防"复读机"三道锁**
- 训练：labels 末尾追加 `[tokenizer.eos_token_id]`。
- 推理：永远不用 `max_length`，改用 `max_new_tokens=256`。
- 采样：加上 `repetition_penalty=1.1`，同时设 `no_repeat_ngram_size=3`。

**让回答"更饱满"的两招**
- Prompt 约束：在 System Prompt 里写死："请以 XX 的口吻详细回答，不少于 50 字。"
- 放开推理：如果听起来太僵硬，把 temperature 提到 `0.6–0.7`、top_p 提到 `0.85–0.9`，更有"人味"。
- 数据决定上限：模型是鹦鹉。数据集中超过 30% 应为长段落回答，否则它永远只会一句话回复。

**显存与速度黄金配置（单卡 RTX 3090 24GB）**

```python
# 黄金参数模板
args = TrainingArguments(
    per_device_train_batch_size=4,   # 全量微调 4，LoRA 8
    gradient_accumulation_steps=2,   # 梯度累积到等效批量
    num_train_epochs=3,              # 约 1000 条样本 3 轮足够
    learning_rate=1e-5,              # 全量微调，防爆炸
    lr_scheduler_type="cosine",      # 余弦退火平滑收敛
    warmup_ratio=0.05,               # 5% warmup 步数
    max_grad_norm=1.0,               # 梯度裁剪——生命线
    bf16=True,                       # 3090 必开
    gradient_checkpointing=False,    # 小模型关，大模型开
)
```

#### 7. 最后感想

模型微调既是一门科学，也是"玄学"与工程的结合。把一个 0.6B 模型从"乱码"调成"知否风格"，几乎把新手能踩的坑都踩了一遍。有了这些经验——Loss 监控、NaN 排查、数据格式对齐、推理约束——今后换到 7B、14B 甚至 72B 模型，底层逻辑完全相通。

### 许可与免责声明

- 本项目仅供学习与研究使用；生成的明兰角色扮演内容仅供娱乐，与原电视剧《知否知否应是绿肥红瘦》无关。
- 请遵循上游模型（Qwen）与数据来源的许可协议。
