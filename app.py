from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

app = Flask(__name__)

# 全局变量存储模型和tokenizer
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 初始化数据库
def init_db():
    conn = sqlite3.connect('health_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS health_records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT NOT NULL,
                  gender TEXT NOT NULL,
                  height REAL NOT NULL,
                  weight REAL NOT NULL,
                  systolic INTEGER NOT NULL,
                  diastolic INTEGER NOT NULL,
                  advice TEXT)''')
    conn.commit()
    conn.close()


# 加载模型
def load_model():
    global model, tokenizer
    try:
        model_path = "./model/Qwen/Qwen3-0___6B"
        lora_path = "./outputLora/finalModel"

        print("正在加载基础模型...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )

        print("正在加载LoRA适配器...")
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        print("模型加载成功!")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


# 使用模型生成建议
def generate_advice(gender, height, weight, systolic, diastolic, previous_data=None):
    if model is None or tokenizer is None:
        return "模型未加载，无法生成建议"

    try:
        # 计算BMI
        bmi = weight / ((height / 100) ** 2)

        # 构建提示词
        prompt = [
            {"role": "system",
             "content": "你是一个健康建议小助手，回答简洁不寒暄，分条列项。你需要根据用户情况进行如下建议：BMI评估/血压评估/具体健康状态/什么运动适合/锻炼时间安排/饮食安排等等。最后加一句：如有专业医疗需求请咨询医生"},
            {"role": "user",
             "content": f"用户信息：性别:{gender}, 身高:{height}cm, 体重:{weight}kg, 收缩压:{systolic}mmHg, 舒张压:{diastolic}mmHg, BMI:{bmi:.1f}"}
        ]

        # 如果有历史数据，添加趋势分析
        if previous_data:
            trend_info = "\n".join([
                f"{data['date']}: 身高{data['height']}cm, 体重{data['weight']}kg, 血压{data['systolic']}/{data['diastolic']}mmHg, BMI{data['bmi']:.1f}"
                for data in previous_data[-5:]  # 最近5条记录
            ])
            prompt.append({"role": "user",
                           "content": f"历史数据趋势:\n{trend_info}\n请根据这些历史数据分析用户的健康状况变化趋势，并给出针对性的建议。"})

        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors='pt').to(device)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True
        )

        advice = tokenizer.decode(outputs[0])
        advice = advice[len(text):-11]
        return advice
    except Exception as e:
        print(f"生成建议时出错: {e}")
        return "生成建议时出错，请稍后再试"


# 生成数值变化图表
def generate_trend_chart(records):
    if len(records) < 2:
        return None
    # 添加中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    dates = [datetime.strptime(r['date'], '%Y-%m-%d') for r in records]
    heights = [r['height'] for r in records]
    weights = [r['weight'] for r in records]
    bmis = [r['bmi'] for r in records]
    systolics = [r['systolic'] for r in records]
    diastolics = [r['diastolic'] for r in records]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.tight_layout(pad=3.0)

    # 身高趋势
    axes[0, 0].plot(dates, heights, marker='o', linestyle='-')
    axes[0, 0].set_title('身高变化趋势')
    axes[0, 0].set_ylabel('身高 (cm)')
    axes[0, 0].grid(True)

    # 体重趋势
    axes[0, 1].plot(dates, weights, marker='o', linestyle='-', color='orange')
    axes[0, 1].set_title('体重变化趋势')
    axes[0, 1].set_ylabel('体重 (kg)')
    axes[0, 1].grid(True)

    # BMI趋势
    axes[1, 0].plot(dates, bmis, marker='o', linestyle='-', color='green')
    axes[1, 0].axhline(y=18.5, color='blue', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=24, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('BMI变化趋势')
    axes[1, 0].set_ylabel('BMI')
    axes[1, 0].text(dates[0], 19, '偏瘦', fontsize=8)
    axes[1, 0].text(dates[0], 25, '超重', fontsize=8)
    axes[1, 0].grid(True)

    # 收缩压趋势
    axes[1, 1].plot(dates, systolics, marker='o', linestyle='-', color='purple')
    axes[1, 1].axhline(y=120, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=130, color='yellow', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=140, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('收缩压变化趋势')
    axes[1, 1].set_ylabel('收缩压 (mmHg)')
    axes[1, 1].text(dates[0], 122, '理想', fontsize=8)
    axes[1, 1].text(dates[0], 132, '偏高', fontsize=8)
    axes[1, 1].text(dates[0], 142, '高血压', fontsize=8)
    axes[1, 1].grid(True)

    # 舒张压趋势
    axes[2, 0].plot(dates, diastolics, marker='o', linestyle='-', color='brown')
    axes[2, 0].axhline(y=80, color='green', linestyle='--', alpha=0.5)
    axes[2, 0].axhline(y=85, color='yellow', linestyle='--', alpha=0.5)
    axes[2, 0].axhline(y=90, color='red', linestyle='--', alpha=0.5)
    axes[2, 0].set_title('舒张压变化趋势')
    axes[2, 0].set_ylabel('舒张压 (mmHg)')
    axes[2, 0].text(dates[0], 82, '理想', fontsize=8)
    axes[2, 0].text(dates[0], 87, '偏高', fontsize=8)
    axes[2, 0].text(dates[0], 92, '高血压', fontsize=8)
    axes[2, 0].grid(True)

    # 清空最后一个子图
    axes[2, 1].axis('off')

    # 调整布局
    plt.tight_layout()

    # 将图表转换为base64字符串
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return image_base64


# 首页路由
@app.route('/')
def index():
    conn = sqlite3.connect('health_data.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM health_records ORDER BY date DESC")
    records = c.fetchall()
    conn.close()

    # 准备数据用于图表生成
    recent_records = []
    for i, record in enumerate(records[:10]):  # 取最近10条记录
        recent_records.append({
            'date': record['date'],
            'height': record['height'],
            'weight': record['weight'],
            'systolic': record['systolic'],
            'diastolic': record['diastolic'],
            'bmi': record['weight'] / ((record['height'] / 100) ** 2)
        })

    chart_image = generate_trend_chart(recent_records) if len(recent_records) > 1 else None

    model_status = "模型已加载" if model is not None else "模型未加载"

    return render_template('index.html',
                           records=records,
                           model_status=model_status,
                           chart_image=chart_image)


# 添加记录路由
@app.route('/add', methods=['POST'])
def add_record():
    date = request.form['date']
    gender = request.form['gender']
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    systolic = int(request.form['systolic'])
    diastolic = int(request.form['diastolic'])

    # 生成建议
    advice = generate_advice(gender, height, weight, systolic, diastolic)

    # 保存到数据库
    conn = sqlite3.connect('health_data.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO health_records (date, gender, height, weight, systolic, diastolic, advice) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (date, gender, height, weight, systolic, diastolic, advice))
    conn.commit()
    conn.close()

    return redirect(url_for('index'))


# 删除记录路由
@app.route('/delete/<int:record_id>')
def delete_record(record_id):
    conn = sqlite3.connect('health_data.db')
    c = conn.cursor()
    c.execute("DELETE FROM health_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

    return redirect(url_for('index'))


# 生成报告路由
@app.route('/generate_report')
def generate_report():
    conn = sqlite3.connect('health_data.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM health_records ORDER BY date ASC")
    records = c.fetchall()
    conn.close()

    if not records:
        return "没有足够的数据生成报告"

    # 获取最新记录
    latest_record = records[-1]

    # 准备历史数据用于分析
    historical_data = []
    for record in records[:-1]:  # 排除最新一条
        historical_data.append({
            'date': record['date'],
            'height': record['height'],
            'weight': record['weight'],
            'systolic': record['systolic'],
            'diastolic': record['diastolic'],
            'bmi': record['weight'] / ((record['height'] / 100) ** 2)
        })

    # 生成综合报告
    report_prompt = [
        {"role": "system",
         "content": "你是一个健康专家，负责生成综合健康报告。请根据用户提供的历史数据和当前数据，生成一份详细的健康评估报告，包括：1. 整体健康趋势分析 2. 各项指标的变化情况 3. 可能的健康风险 4. 具体的改善建议。报告应该结构清晰，重点突出。"},
        {"role": "user",
         "content": f"最新健康数据：\n日期: {latest_record['date']}\n性别: {latest_record['gender']}\n身高: {latest_record['height']}cm\n体重: {latest_record['weight']}kg\n收缩压: {latest_record['systolic']}mmHg\n舒张压: {latest_record['diastolic']}mmHg\nBMI: {latest_record['weight'] / ((latest_record['height'] / 100) ** 2):.1f}\n\n历史数据趋势:\n" + "\n".join(
             [
                 f"{data['date']}: 身高{data['height']}cm, 体重{data['weight']}kg, 血压{data['systolic']}/{data['diastolic']}mmHg, BMI{data['bmi']:.1f}"
                 for data in historical_data[-10:]])}
    ]

    text = tokenizer.apply_chat_template(report_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors='pt').to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True
    )

    report = tokenizer.decode(outputs[0])
    report = report[len(text):-11]

    return render_template('report.html', report=report)


# 初始化
if __name__ == '__main__':
    init_db()
    load_model()  # 尝试加载模型
    app.run(debug=True)
