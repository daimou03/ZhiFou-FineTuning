import os
import json
import random
from datasets import Dataset, load_from_disk
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from modelscope.msdatasets import MsDataset
import pandas as pd
import torch
import swanlab

# 使用swanlab
swanlab.login(api_key="CXJ9tRSpL0A78jr9wE8oh", save=True)

MAX_LENGTH = 4096
eval_process_path = './processed_eval_dataset'
eval_dataset_path = 'eval_format.json'
train_process_path = './processed_train_dataset'
train_dataset_path = 'train_format.json'

run = swanlab.init(
    # 设置项目
    project="Qwen3-0.6B-MS",
    experiment_name="Medical",
    description="基于医疗数据集的Qwen3-0.6B模型的微调",
    # 跟踪超参数与实验元数据
    config={

        'model': 'Qwen/Qwen3-0.6B',
        'dataset': 'krisfu/delicate_medical_r1_data',
        'data_max_length': MAX_LENGTH,
        'learning_rate': 5e-5,
        'batch_size': 1,
        'epochs': 2,
        'gradient_accumulation_steps': 4,
    },
)
system_prompt = '你是一个医学专家，为用户解答问题，并进行思考后回答'

script_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_path, "dataset")
cache_dir = os.path.join(script_path, "model")

# 下载模型
print('---------------------')
print('开始下载模型...')
model_dir = snapshot_download('Qwen/Qwen3-0.6B', cache_dir=cache_dir, revision='master')
print('模型已下载完毕')
print('正在加载模型...')
model = AutoModelForCausalLM.from_pretrained(model_dir,device_map='auto', torch_dtype='auto')
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 用 eos_token 作为 pad_token
print('已加载完毕')
print('模型运行设备为：', next(model.parameters()).device)


# 处理数据集成需要的格式
def dataset_jsonl_transfer(origin_path, new_path):
    messages = []

    with open(origin_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            message = {
                'instruction': system_prompt,
                'input': data['question'],
                'output': '<think>' + data['think'] + '</think>' + data['answer'],
            }
            messages.append(message)

    with open(new_path, 'w', encoding='utf-8') as f:
        for line in messages:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def process_fun(example):
    input_all = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False)
    output_all = tokenizer(example['output'], add_special_tokens=False)

    input_ids = input_all['input_ids'] + output_all['input_ids']
    attention_mask = input_all['attention_mask'] + output_all['attention_mask']
    output_ids = [-100] * len(input_all['input_ids']) + output_all['input_ids']

    if  len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        output_ids = output_ids[:MAX_LENGTH]


    return {'input_ids': input_ids, "attention_mask": attention_mask, 'labels':  output_ids}


def predict(message, model, tokenizer):
    input = tokenizer.apply_chat_template(message,
                                          add_generation_prompt=True,
                                          tokenize=False, )

    input_ids = tokenizer(input, return_tensors='pt').to(model.device)

    generation_ids = model.generate(**input_ids,
                                    max_length=2048,
                                    do_sample=True,
                                    )

    generation_ids = generation_ids[0]  # 返回的是（batchsize，seqlen）的张量，输入的是一句话默认batchsize=1

    response = tokenizer.decode(generation_ids[len(input_ids['input_ids'][0]):], skip_special_tokens=True)
    return response


def get_processed_dataset(dataset_path, process_path, process_fun):
    print('开始处理数据集...')
    if os.path.exists(process_path):
        print('处理完毕的数据集已存在')
        dataset = load_from_disk(process_path)

    else:
        df = pd.read_json(dataset_path, lines=True)
        ds = Dataset.from_pandas(df)
        dataset = ds.map(process_fun, remove_columns=ds.column_names)
        dataset.save_to_disk(process_path)
        print(f'数据集已储存到当前目录{train_process_path}下')

    print('数据量为:', dataset.num_rows)
    return dataset

def check_dataset_quality(dataset, tokenizer, num_samples=3):
    """检查数据集质量，解码token查看实际内容"""
    print("=" * 80)
    print("检查数据集质量：")
    print("=" * 80)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n样本 {i + 1}:")

        # 解码完整的input_ids（包括系统提示和用户输入）
        full_input = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        print(f"完整输入: {full_input}")

        # 找到labels中非-100的部分（模型应该生成的部分）
        valid_labels = [label for label in sample['labels'] if label != -100]
        if valid_labels:
            expected_output = tokenizer.decode(valid_labels, skip_special_tokens=False)
            print(f"期望输出: {expected_output}")
        else:
            print("期望输出: 没有有效的labels")

        # 检查特殊token
        print(f"input_ids长度: {len(sample['input_ids'])}")
        print(f"labels长度: {len(sample['labels'])}")
        print(f"有效labels数量: {len(valid_labels)}")
        print("-" * 80)



def main():
    # 下载数据集


    os.environ['MODELSCOPE_CACHE'] = data_path

    ds = MsDataset.load('krisfu/delicate_medical_r1_data', subset_name='default', split='train')
    print(ds)

    # 分割数据集并储存
    random.seed(50)
    ds = list(ds)

    random.shuffle(ds)
    split_idx = int(len(ds) * 0.9)

    train_ds = ds[:split_idx]
    eval_ds = ds[split_idx:]

    with open('train.json', 'w', encoding='utf-8') as f:
        for line in train_ds:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    with open('eval.json', 'w', encoding='utf-8') as f:
        for line in eval_ds:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    dataset_jsonl_transfer(origin_path='train.json', new_path='train_format.json')
    dataset_jsonl_transfer(origin_path='eval.json', new_path='eval_format.json')

    # 处理数据
    print('---------------------')
    print('处理训练集：')

    train_dataset = get_processed_dataset(train_dataset_path, train_process_path, process_fun)


    print('---------------------')
    print('处理验证集：')

    eval_dataset = get_processed_dataset(eval_dataset_path, eval_process_path, process_fun)

    print(train_dataset)
    print("特征:", train_dataset.features)
    print("行数:", train_dataset.num_rows)
    print("列名:", train_dataset.column_names)
    print("形状:", train_dataset.shape)

    # print('检查训练集质量：')
    # check_dataset_quality(train_dataset, tokenizer, num_samples=3)




    # 训练模型
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    args = TrainingArguments(output_dir='./output/Qwen3-0.6B',
                             per_device_train_batch_size=1,
                             per_device_eval_batch_size=1,
                             gradient_accumulation_steps=4,
                             eval_strategy='steps',
                             eval_steps=100,
                             logging_steps=10,
                             num_train_epochs=1,
                             save_steps=400,
                             learning_rate=5e-5,
                             save_on_each_node=True,
                             gradient_checkpointing=True,  # 梯度检查节点开启
                             report_to='swanlab',


                             )

    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      data_collator=collator, )

    trainer.train()

    # 验证集查看模型输出效果
    test_list = []
    test_df = pd.read_json(eval_dataset_path, lines=True).head(1)
    for _, row in test_df.iterrows():
        message = [{'role': 'system', 'content': row['instruction']},
                   {'role': 'user', 'content': row['input']}, ]
        response = predict(message, model, tokenizer)
        text = f'''
问题:{row['input']}
模型回答:{response}'''
        test_list.append(swanlab.Text(text))
        print(text)
    swanlab.log({'Prediction': test_list})
    trainer.save_model("./output/final_model")
    swanlab.finish()


if __name__ == '__main__':
    main()
