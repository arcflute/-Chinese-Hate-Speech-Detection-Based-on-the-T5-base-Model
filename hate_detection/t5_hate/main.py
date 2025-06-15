import json
import torch
import jieba
import synonyms
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
from tqdm import tqdm



# ========== EDA 增强函数 ==========
def get_synonyms(word: str, topk: int = 5):
    syns = synonyms.nearby(word)[0]
    return [w for w in syns if w != word][:topk]

def synonym_replacement(words, n_sr=1):
    new_words = words.copy()
    candidates = [w for w in words if get_synonyms(w)]
    for w in random.sample(candidates, min(n_sr, len(candidates))):
        idx = new_words.index(w)
        new_words[idx] = random.choice(get_synonyms(w))
    return new_words

def random_insertion(words, n_ri=1):
    new_words = words.copy()
    for _ in range(n_ri):
        candidates = [w for w in new_words if get_synonyms(w)]
        if not candidates: break
        w = random.choice(candidates)
        new_words.insert(random.randint(0, len(new_words)), random.choice(get_synonyms(w)))
    return new_words

def random_swap(words, n_rs=1):
    new_words = words.copy()
    for _ in range(n_rs):
        if len(new_words) < 2: break
        i, j = random.sample(range(len(new_words)), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words

def random_deletion(words, p=0.1):
    if len(words) == 1: return words
    new = [w for w in words if random.random() > p]
    return new if new else [random.choice(words)]

def eda(sentence: str,
        alpha_sr=0.1, alpha_ri=0.1,
        alpha_rs=0.1, p_rd=0.1,
        num_aug=4):
    words = list(jieba.cut(sentence, cut_all=False))
    n = len(words)
    n_sr = max(1, int(alpha_sr * n))
    n_ri = max(1, int(alpha_ri * n))
    n_rs = max(1, int(alpha_rs * n))
    
    augmented = []
    augmented.append("".join(synonym_replacement(words, n_sr)))
    augmented.append("".join(random_insertion(words, n_ri)))
    augmented.append("".join(random_swap(words, n_rs)))
    augmented.append("".join(random_deletion(words, p_rd)))
    
    for _ in range(num_aug - 4):
        op = random.choice([
            lambda w: synonym_replacement(w, n_sr),
            lambda w: random_insertion(w, n_ri),
            lambda w: random_swap(w, n_rs),
            lambda w: random_deletion(w, p_rd),
        ])
        augmented.append("".join(op(words)))
    return augmented

#  模型配置
model_name = "Langboat/mengzi-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#  强制使用GPU
if not torch.cuda.is_available():
    raise RuntimeError("必须使用GPU训练，请检查CUDA环境")
device = torch.device("cuda")
model = model.to(device)

#  加载 JSON 数据
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

#  清洗数据（仅保留有 content 和非空 output 的样本）
def clean_data(data):
    return [d for d in data if d.get("content") and d.get("output") and d["output"].strip()]

#  加载并处理训练数据
all_data = clean_data(load_json("../train.json"))

# ========== 在这里做 EDA 增强 ==========
augmented = []
for item in all_data:
    aug_texts = eda(
        item["content"],
        alpha_sr=0.1, alpha_ri=0.1,
        alpha_rs=0.1, p_rd=0.1,
        num_aug=4
    )
    for txt in aug_texts:
        augmented.append({"content": txt, "output": item["output"]})
# 合并原始和增强数据
all_data = all_data + augmented

#  自动划分训练集和验证集（9:1）
train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)

#  加载测试集
test_data = load_json("../test1.json")

#  转换为 Huggingface Datasets
train_ds = Dataset.from_list(train_data)
val_ds = Dataset.from_list(val_data)
test_ds = Dataset.from_list(test_data)

#  分词函数
def tokenize_function(example):
    model_inputs = tokenizer(
        example["content"],
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["output"],
            max_length=256,
            padding="max_length",
            truncation=True,
        )
    model_inputs["labels"] = [
        (token if token != tokenizer.pad_token_id else -100)
        for token in labels["input_ids"]
    ]
    return model_inputs

# 分词映射
tokenized_train = train_ds.map(tokenize_function)
tokenized_val = val_ds.map(tokenize_function)

# 数据打包器
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=50,
    save_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True,  
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 初始化 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("../saved_model")
tokenizer.save_pretrained("../saved_model")


print("🔎 开始推理测试集...")

model.eval()
with open("demo.txt", "w", encoding="utf-8") as f:
    for item in tqdm(test_data):
        input_text = item["content"]
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding="max_length"
        ).to(device)

        output_ids = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            early_stopping=True
        )
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

 
        if not pred.endswith("[END]"):
            pred += " [END]"

        f.write(pred + "\n")
