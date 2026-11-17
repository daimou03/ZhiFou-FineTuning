### 项目简介

Qwen3-0.6-MS 是一个基于 Qwen/Qwen3-0.6B 的中文角色扮演问答（明兰）微调与推理示例项目，包含：

- 数据处理与全参数微调脚本：`train.py`
- 基于 LoRA 的高效微调脚本：`train_lora.py`
- Flask 推理与可视化 Web 应用：`app.py`（记录健康数据、生成建议与报告）

### 环境要求

- 操作系统：Linux / Windows / macOS
- Python ≥ 3.10
- GPU（推荐，支持 CUDA），或 CPU（训练会很慢）

安装依赖：

```bash
pip install -r requirements.txt
```

依赖主要包含：`torch`, `transformers`, `modelscope`, `peft`, `pandas`。

训练脚本额外依赖 `datasets`（requirements.txt 未包含，需单独安装）：

```bash
pip install datasets
```

Web 应用额外依赖（代码会用到，建议安装）：

```bash
pip install flask matplotlib
```

### 目录结构（关键文件）

- `train.py`：全参数微调（保存至 `./output/final_model`）
- `train_lora.py`：LoRA 微调（保存至 `./output_lora/final_model`）
- `app.py`：加载基础模型 + LoRA 适配器，提供 Web 推理
- `templates/`：网页模板（`index.html`, `report.html`）
- `data/minglan_qa_clean.json`：训练数据（JSON 数组，共 1154 条，每条含 `问题`、`明兰` 字段）
- `data/minglan_qa_clean.csv`：同内容的 CSV 源文件（训练实际读取 JSON）
- `model/`：基础模型缓存目录（首次运行时由 ModelScope 自动下载）
- 运行训练后还会生成：`train.json` / `eval.json`（切分数据）、`train_format.json` / `eval_format.json`（指令格式）、`./processed_train_dataset` / `./processed_eval_dataset`（tokenize 缓存）、`./output/` / `./output_lora/`（训练输出）

### 数据集说明

训练脚本从本地文件 `data/minglan_qa_clean.json` 加载数据（共 1154 条），固定随机种子（`random.seed(50)`）打乱后按 9:1 切分训练/验证集，并转换为指令对格式：

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

如需替换数据集：准备一个含 `问题`/`明兰` 两个字段的 JSON 数组文件，并修改 `train.py` / `train_lora.py` 中的 `data_file` 路径与字段名。

### 微调流程

#### A. 全参数微调（train.py）

1. 执行命令：

```bash
python train.py
```

2. 主要流程：

- 自动下载基础模型 `Qwen/Qwen3-0.6B` 至 `./model/` 缓存
- 将原始数据转为指令输出格式并 Tokenize 到 `MAX_LENGTH=4096`
- 使用 `Trainer` 训练
- 训练完成自动保存至：`./output/final_model`

3. 关键超参（可在脚本内修改）：

- 学习率：`5e-5`
- batch：`per_device_train_batch_size=1`，`gradient_accumulation_steps=4`
- epoch：`num_train_epochs=1`（示例值）
- 评估/保存步长：`eval_steps=100`，`save_steps=400`

#### B. LoRA 微调（train\_lora.py）

1. 执行命令：

```bash
python train_lora.py
```

2. 主要流程：

- 同样下载并加载 `Qwen/Qwen3-0.6B`
- 注入 LoRA：
  - `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
  - `r=8, lora_alpha=32, lora_dropout=0.1`
- 使用 `Trainer` 训练 LoRA 适配器
- 训练完成保存到：`./output_lora/final_model`
- 评估/保存步长：`eval_steps=100`，`save_steps=100`

3. 推理时通过 `PeftModel.from_pretrained` 在基础模型上加载 LoRA 适配器（见下一节）。

### 推理与 Web 应用

`app.py` 提供一个简单的健康数据记录与建议生成网页。

启动前请先运行 `python train_lora.py` 生成 `./output_lora/final_model`，否则启动时会提示“模型加载失败”，无法生成建议与报告。

1. 路径配置：
   在 `app.py` 的 `load_model()` 中，默认路径已与训练脚本对齐：

```python
model_path = "./model/Qwen/Qwen3-0.6B"       # 按实际缓存目录调整
lora_path = "./output_lora/final_model"      # 与 train_lora.py 对齐
```

2. 运行 Web：

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:5000/`。

3. 功能说明：

- 首页添加健康记录（日期/性别/身高/体重/血压）
- 写入 SQLite 本地库 `health_data.db`
- 基于模型生成个性化建议（含 BMI/血压评估等）
- 展示近 10 条记录的趋势图（`matplotlib`，默认中文字体配置）
- 一键生成综合报告（基于历史趋势 + 最新数据）

4. 推理细节：

- 使用 `tokenizer.apply_chat_template` 生成对话输入（`enable_thinking=False`，关闭 Qwen3 思考模式）
- system prompt 与训练一致：`你是盛明兰，以明兰的口吻和性格回答问题`
- 生成参数：建议生成 `max_new_tokens=400, temperature=0.7, do_sample=True`；综合报告 `max_new_tokens=2048`
- 通过 `PeftModel.from_pretrained` 加载 LoRA 适配器

### 常见问题

- 模型或 LoRA 路径不匹配：请根据本地实际下载/保存路径调整 `app.py` 中的 `model_path`/`lora_path`。注意不同 ModelScope 版本缓存目录名可能不同（新版为 `./model/Qwen/Qwen3-0.6B`，旧版可能为 `./model/Qwen/Qwen3-0___6B`）。
- 启动 `app.py` 提示“模型加载失败”：通常是尚未训练 LoRA（缺少 `./output_lora/final_model`）或路径不对，先运行 `python train_lora.py` 并检查 `model_path`/`lora_path`。
- 字体/中文显示异常：`matplotlib` 已加入常用中文字体名称，必要时在本机安装相应字体。
- 无 GPU：可在 CPU 上运行推理，但速度较慢；训练强烈建议使用 GPU。
- 数据处理缓存：脚本会将 tokenize 后的数据缓存到 `./processed_train_dataset`、`./processed_eval_dataset`；若更换了数据集或字段，请删除这两个缓存目录后重新运行（`train_format.json`、`eval_format.json` 会自动重新生成）。

### 拓展

#### 1. 如何使用合并后的模型

你当前保存的 `./output/final_model` 是完整的模型权重（因为是全量微调，不是 LoRA）。
如果你想把它导出成可以直接用 vLLM 或 transformers 推理的标准模型，直接运行：

```python
# 加载并保存即可
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./output/final_model")
tokenizer = AutoTokenizer.from_pretrained("./output/final_model")
model.save_pretrained("./exported_model")
tokenizer.save_pretrained("./exported_model")
```

#### 2. 关于 0.6B 模型的局限性（心理预期）

虽然模型现在的语气很对了，但你可能也注意到了，面对长段的背景描述（第一个问题），它只回了一句"侯爷可真是见到了真金不怕火炼啊！"，这说明 0.6B 模型的上下文理解能力极弱。它现在像是一个"只会背台词的演员"。如果未来的业务需要模型进行复杂的推理或长段对话，建议换成 Qwen2.5-7B-Instruct。你当前的这套代码结构（包括格式对齐、防炸参数、推理逻辑）可以直接无缝平移到 7B 模型上，效果会直接起飞。

#### 3. 记录你的训练成果

你现在可以把这次跑通的参数记录下来，这是你在这个项目里的重大里程碑：

- 模型：Qwen3-0.6B
- 微调方式：全量微调（Full Fine-tuning）
- 防炸三板斧：FP32 加载 + lr=1e-5 + max_grad_norm=1.0
- 数据格式：apply_chat_template + EOS（结束符）
- 硬件：单卡 RTX 3090，约耗时 3.8 分钟/3 epochs

### 许可与声明

- 本项目仅供学习与研究使用，生成内容不构成医疗建议，若有健康问题请咨询专业医生。
- 请遵循上游模型（Qwen）与数据来源的许可协议。
