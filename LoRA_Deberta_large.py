import os
import random
import json
import csv
from datetime import datetime
import psutil
import pynvml
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

# peft for using LoRA
from peft import LoraConfig, get_peft_model

# ========== 0. 輸出資料夾 & GPU 指定 ==========

#所有resluts都會儲存到以下資料夾
OUTPUT_DIR = "./LoRA_Deberta_large/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # GPU index
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def log_memory(stage=""):
    """
    記錄 CPU / GPU 使用量，寫入 OUTPUT_DIR/memory_usage.log
    """
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**2
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_used = mem_info.used / 1024**2
    gpu_total= mem_info.total / 1024**2

    line = f"[{datetime.now()}][{stage}] RAM={ram_usage:.2f}MB, GPU={gpu_used:.2f}/{gpu_total:.2f}MB"
    print(line)
    with open(os.path.join(OUTPUT_DIR, "memory_usage.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ========== 1. setting parameters ==========
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LEN = 128

MODEL_NAME = "microsoft/deberta-large"  # Hugggingface model path

# LoRA 設定
num_layers = 24  # DeBERTa-large 一般是 24 層
target_modules = []
for i in range(num_layers):
    target_modules.append(f"encoder.layer.{i}.attention.self.in_proj")
    target_modules.append(f"encoder.layer.{i}.attention.self.pos_q_proj")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=target_modules
)


'''
MODEL_NAME = "microsoft/deberta-v2-xlarge"  # Hugggingface model path

# LoRA 設定
num_layers = 24  # DeBERTa-large 一般是 24 層
target_modules = []
for i in range(num_layers):
    target_modules.append(f"encoder.layer.{i}.attention.self.query_proj")
    target_modules.append(f"encoder.layer.{i}.attention.self.key_proj")
    target_modules.append(f"encoder.layer.{i}.attention.self.value_proj")


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=target_modules
)
'''


# ========== 2. 資料集載入: 使用 train split (SST-2, MNLI, CLINC-OOS) & 切8:1:1 ==========

print("[INFO] Loading datasets: SST-2, MNLI, CLINC-OOS...")

# 使用 huggingface datasets 載入資料集
ds_sst2 = load_dataset("glue", "sst2")   
ds_mnli = load_dataset("glue", "mnli")
ds_clinc= load_dataset("clinc_oos", "small")

# 切分資料集
def split_dataset_8_1_1(dataset_list):

    random.shuffle(dataset_list)
    n = len(dataset_list)
    tr_end = int(0.8*n)
    va_end = int(0.9*n)
    train_ = dataset_list[:tr_end]
    valid_ = dataset_list[tr_end:va_end]
    test_  = dataset_list[va_end:]
    return train_, valid_, test_

# SST-2
def load_sst2_all():
    # 這裡以 "train" split 為例，如需包含 validation/test 官方資料, 視實驗需求調整
    data = ds_sst2["train"]  
    return [(ex["sentence"], ex["label"], "SST-2") for ex in data]

# MNLI
def load_mnli_all():
    data = ds_mnli["train"]  # or combine train + val, 依需求
    return [(f"{ex['premise']} [SEP] {ex['hypothesis']}", ex["label"], "MNLI") for ex in data]

# CLINC-OOS
def load_clinc_all():
    data = ds_clinc["train"]  # or combine train+val+test, 依需求
    return [(ex["text"], ex["intent"], "CLINC-OOS") for ex in data]

sst2_raw = load_sst2_all()
sst2_train, sst2_valid, sst2_test = split_dataset_8_1_1(sst2_raw)

mnli_raw = load_mnli_all()
mnli_train, mnli_valid, mnli_test = split_dataset_8_1_1(mnli_raw)

clinc_raw = load_clinc_all()
clinc_train, clinc_valid, clinc_test = split_dataset_8_1_1(clinc_raw)


# 合併三個datasets
train_data = sst2_train + mnli_train + clinc_train
valid_data = sst2_valid + mnli_valid + clinc_valid
test_data  = sst2_test  + mnli_test  + clinc_test

print(f"[INFO] after combining => train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")

# label mapping reset
unique_labels = set()
for txt, lb, ds in (train_data + valid_data + test_data):
    unique_labels.add(lb)
unique_labels = sorted(unique_labels, key=lambda x: str(x))
label2id = {str(lb): i for i, lb in enumerate(unique_labels)}
id2label = {i: str(lb) for lb, i in label2id.items()}
num_labels = len(label2id)
print(f"[INFO] total unique labels: {num_labels}")

# 轉成 dict
def convert_to_dict(data_list):
    out = []
    for txt, lb, ds in data_list:
        lb_id = label2id[str(lb)]
        out.append({"text": txt, "label_id": lb_id, "dataset": ds})
    return out



train_data = convert_to_dict(train_data)
valid_data = convert_to_dict(valid_data)
test_data  = convert_to_dict(test_data)

# 存入 csv檔
def save_csv(data_list, name):
    import csv
    with open(os.path.join(OUTPUT_DIR, f"{name}_split.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text","label_id","dataset"])
        for ex in data_list:
            writer.writerow([ex["text"], ex["label_id"], ex["dataset"]])

save_csv(train_data, "train")
save_csv(valid_data, "valid")
save_csv(test_data,  "test")

# 分開的testing data
test_sst2  = [ex for ex in test_data if ex["dataset"]=="SST-2"]
test_mnli  = [ex for ex in test_data if ex["dataset"]=="MNLI"]
test_clinc = [ex for ex in test_data if ex["dataset"]=="CLINC-OOS"]

# ========== 3. 建立 Dataset, DataLoader ==========

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    texts  = [b["text"] for b in batch]
    labels = [b["label"] for b in batch]
    dsname = [b["dataset"] for b in batch]

    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    enc["dsname"] = dsname
    return enc

train_ds = MyDataset(train_data)
valid_ds = MyDataset(valid_data)
test_ds  = MyDataset(test_data)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

sst2_test_loader  = DataLoader(MyDataset(test_sst2),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
mnli_test_loader  = DataLoader(MyDataset(test_mnli),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
clinc_test_loader = DataLoader(MyDataset(test_clinc), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ========== 4. LoRA model ==========

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

# 構建 LoRA 設定，已在上面定義 lora_config

lora_model = get_peft_model(base_model, lora_config)
lora_model.to(device)

# 檢查
print("[INFO] LoRA modules:", lora_model.peft_config)

optimizer = torch.optim.AdamW(lora_model.parameters(), lr=LR)

# ========== 5. Evaluation ==========

def evaluate(dataloader):
    lora_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labs = batch["labels"].cpu().numpy()

            outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labs.tolist())

    acc = accuracy_score(all_labels, all_preds)
    return acc

# ========== 6. Training Loop ==========

best_val_acc = 0.0
train_log = []

log_memory("Before Training")

for epoch in range(1, EPOCHS+1):
    lora_model.train()
    total_loss, total_step = 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labs = batch["labels"].to(device)

        outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labs)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_step += 1
        loop.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss/total_step if total_step>0 else 0
    val_acc = evaluate(valid_loader)
    info = f"Epoch {epoch}, train_loss={avg_loss:.4f}, val_acc={val_acc:.4f}"
    print(info)
    train_log.append(info)

    log_memory(f"End Epoch {epoch}")

    # save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(lora_model.state_dict(), os.path.join(OUTPUT_DIR, "lora_model_best.pt"))
        print(f"[INFO] Saved best model with val_acc={val_acc:.4f}")

# save model
torch.save(lora_model.state_dict(), os.path.join(OUTPUT_DIR, "lora_model_last.pt"))
log_memory("After Training")

with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w", encoding="utf-8") as f:
    for line in train_log:
        f.write(line+"\n")

# ========== 7. testing ==========

print("[INFO] Evaluate on test set (mixed)...")
test_acc = evaluate(test_loader)
print(f"Overall test acc = {test_acc:.4f}")

sst2_acc  = evaluate(sst2_test_loader)
mnli_acc  = evaluate(mnli_test_loader)
clinc_acc = evaluate(clinc_test_loader)

print(f"SST-2 test acc  = {sst2_acc:.4f}")
print(f"MNLI test acc   = {mnli_acc:.4f}")
print(f"CLINC test acc  = {clinc_acc:.4f}")

test_line = f"Test Results => Overall:{test_acc:.4f}, SST-2:{sst2_acc:.4f}, MNLI:{mnli_acc:.4f}, CLINC:{clinc_acc:.4f}"
print(test_line)
with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "a", encoding="utf-8") as f:
    f.write(test_line+"\n")

log_memory("After Testing")

print("[INFO] Done. All results saved in:", OUTPUT_DIR)
