### 项目简介

Qwen3-0.6-MS 是一个基于 Qwen/Qwen3-0.6B 的医疗问答微调与推理示例项目，包含：
- 数据处理与全参数微调脚本：`train.py`
- 基于 LoRA 的高效微调脚本：`train_lora.py`
- Flask 推理与可视化 Web 应用：`app.py`（记录健康数据、生成建议与报告）


### 环境要求
- 操作系统：Windows 10/11（PowerShell）或其它主流平台
- Python ≥ 3.10
- GPU（推荐，支持 CUDA），或 CPU（训练会很慢）

安装依赖：

```bash
pip install -r requirements.txt
```

依赖主要包含：`torch`, `transformers`, `modelscope`, `peft`, `pandas`, `swanlab`。


### 目录结构（关键文件）
- `train.py`：全参数微调（保存至 `./output/final_model`）
- `train_lora.py`：LoRA 微调（保存至 `./output_lora/final_model`）
- `app.py`：加载基础模型 + 可选 LoRA 适配器，提供 Web 推理
- `templates/`：网页模板（`index.html`, `report.html`）


### 数据集说明
训练脚本默认从 ModelScope 加载数据集 `krisfu/delicate_medical_r1_data`，并自动切分训练/验证集，转换为指令对格式，包含思考链标记：`<think>…</think>`。

如需替换数据集，请在 `train.py` / `train_lora.py` 中修改：
```python
MsDataset.load('krisfu/delicate_medical_r1_data', subset_name='default', split='train')
```


### 微调流程

#### A. 全参数微调（train.py）
1) 执行命令：
```bash
python train.py
```

2) 主要流程：
- 自动下载基础模型 `Qwen/Qwen3-0.6B` 至 `./model/` 缓存
- 将原始数据集转为指令输出格式并 Tokenize 到 `MAX_LENGTH=4096`
- 使用 `Trainer` 训练，并记录到 SwanLab（需有效 `api_key`）
- 训练完成自动保存至：`./output/final_model`

3) 关键超参（可在脚本内修改）：
- 学习率：`5e-5`
- batch：`per_device_train_batch_size=1`，`gradient_accumulation_steps=4`
- epoch：`num_train_epochs=1`（示例值）
- 评估/保存步长：`eval_steps=100`，`save_steps=400`


#### B. LoRA 微调（train_lora.py）
1) 执行命令：
```bash
python train_lora.py
```

2) 主要流程：
- 同样下载并加载 `Qwen/Qwen3-0.6B`
- 注入 LoRA：
  - `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
  - `r=8, lora_alpha=32, lora_dropout=0.1`
- 使用 `Trainer` 训练 LoRA 适配器
- 训练完成保存到：`./output_lora/final_model`

3) 推理时可在基础模型上合并/加载 LoRA 适配器（见下一节）。


### 推理与 Web 应用

`app.py` 提供一个简单的健康数据记录与建议生成网页：

1) 路径配置（重要）：
在 `app.py` 的 `load_model()` 中：
```python
model_path = "./model/Qwen/Qwen3-0___6B"
lora_path = "./outputLora/finalModel"
```
该示例路径与训练脚本默认保存路径存在差异。若使用本仓库提供的训练脚本：
- 基础模型通常缓存到 `./model/` 下，具体子目录以实际下载结果为准
- LoRA 适配器默认保存为 `./output_lora/final_model`

请将 `model_path` 与 `lora_path` 修改为你本地的真实路径，例如：
```python
model_path = "./model/Qwen/Qwen3-0.6B"       # 按实际缓存目录调整
lora_path = "./output_lora/final_model"      # 与 train_lora.py 对齐
```

2) 运行 Web：
```bash
python app.py
```
浏览器访问 `http://127.0.0.1:5000/`。

3) 功能说明：
- 首页添加健康记录（日期/性别/身高/体重/血压）
- 写入 SQLite 本地库 `health_data.db`
- 基于模型生成个性化建议（含 BMI/血压评估等）
- 展示近 10 条记录的趋势图（`matplotlib`，默认中文字体配置）
- 一键生成综合报告（基于历史趋势 + 最新数据）

4) 推理细节：
- 使用 `tokenizer.apply_chat_template` 生成对话输入
- 生成参数（示例）：`max_new_tokens=400, temperature=0.7, do_sample=True`
- 通过 `PeftModel.from_pretrained` 加载 LoRA 适配器


### 常见问题
- 模型或 LoRA 路径不匹配：请根据本地实际下载/保存路径调整 `app.py` 中的 `model_path`/`lora_path`。
- 字体/中文显示异常：`matplotlib` 已加入常用中文字体名称，必要时在本机安装相应字体。
- 无 GPU：可在 CPU 上运行推理，但速度较慢；训练强烈建议使用 GPU。
- SwanLab 未登录：将 `swanlab.login(api_key=...)` 中的 `api_key` 替换为你的有效密钥，或在脚本中移除上报。


### 许可与声明
- 本项目仅供学习与研究使用，生成内容不构成医疗建议，若有健康问题请咨询专业医生。

## 项目简介

基于 Qwen3-0.6B 的中文医疗场景微调与推理示例：
- `train.py`：全参数微调示例，使用 ModelScope 数据集与 Hugging Face Transformers 训练并接入 SwanLab 记录实验。
- `train_lora.py`：LoRA 微调示例（推荐资源有限场景）。
- `app.py`：Flask Web 应用，加载基础模型与 LoRA 适配器，为健康数据（身高、体重、血压等）生成建议与报告，并以 SQLite 本地存储历史记录及可视化趋势图。

---

## 环境准备

- Python：3.10 或 3.11
- OS：Windows 10/11（例如路径 `E:\ai项目\Qwen3-0.6-MS`）
- 安装依赖：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 额外依赖（代码会用到，建议安装）：
```bash
pip install flask matplotlib datasets
```
> 提示：`sqlite3` 为标准库；`torch==2.5.1` 已在 `requirements.txt` 指定，请到 PyTorch 官网根据你的 CUDA 版本安装兼容轮子。

---

## 目录结构

- `app.py`：Web 服务与推理逻辑（加载基础模型与 LoRA、SQLite 记录、图表、报告生成）
- `train.py`：全参数微调脚本
- `train_lora.py`：LoRA 微调脚本
- `templates/index.html`、`templates/report.html`：前端模板
- `requirements.txt`：依赖

---

## 数据与模型路径

- 基础模型加载路径（在 `app.py`）：`./model/Qwen/Qwen3-0___6B`
- LoRA 适配器路径（在 `app.py`）：`./outputLora/finalModel`

注意：`train_lora.py` 训练输出为 `./output_lora/final_model`，与 `app.py` 默认不一致。可以：
- 将 `./output_lora/final_model` 拷贝到 `./outputLora/finalModel`；或
- 修改 `app.py` 中 `lora_path` 指向你的实际 LoRA 目录。

---

## 运行 Web 应用

```bash
python app.py
```
浏览器打开：`http://127.0.0.1:5000/`

功能：
- 新增健康数据（日期、性别、身高、体重、血压）；
- 自动生成个性化健康建议；
- 查看最近记录趋势图（BMI、收缩压、舒张压等）；
- 一键生成综合健康报告。

> 若提示“模型未加载”，请检查 `model_path` 与 `lora_path` 是否存在且可读。

---

## 训练

### 全参数微调（`train.py`）
- 数据集：`krisfu/delicate_medical_r1_data`（ModelScope）
- 自动切分训练/验证集并转换为 SFT 格式；SwanLab 记录实验

运行：
```bash
python train.py
```
产物：
- 输出目录：`./output/Qwen3-0.6B`
- 最终模型：`./output/final_model`

### LoRA 微调（`train_lora.py`）
- 在基础模型上添加 LoRA，资源占用更低

运行：
```bash
python train_lora.py
```
产物：
- 输出目录：`./output_lora/Qwen3-0.6B`
- 最终 LoRA：`./output_lora/final_model`

---

## SwanLab 配置建议

代码示例里写死了 `api_key`。建议改为环境变量：
```python
import os
swanlab.login(api_key=os.getenv("SWANLAB_API_KEY"), save=True)
```
Windows 临时设置：
```bash
set SWANLAB_API_KEY=你的key
```

---

## 常见问题

- 模型加载失败：检查本地路径、显存是否足够，或改用 LoRA 推理。
- Torch/CUDA 不匹配：到 PyTorch 官网根据 CUDA 版本重装 `torch`。
- 中文字体：`matplotlib` 已设置常见中文字体；若乱码，安装 `SimHei` 或 `微软雅黑`。

---

## 许可

仅供研究学习使用，请遵循上游模型和数据集许可。
