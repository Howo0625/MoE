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

from transformers import AutoTokenizer, AutoConfig, AutoModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# ================== 1. 設定與初始化 ==================

# 所有resluts都會儲存到以下資料夾
OUTPUT_DIR = "./MoE_DeBERTa_base/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 如果使用多張卡，例："0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 監控 GPU 記憶體
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  

def log_memory(stage=""):
    """
    記錄 CPU / GPU 使用量，並寫入 OUTPUT_DIR/memory_usage.log
    """
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**2  # 轉換為 MB

    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_used = mem_info.used / 1024**2
    gpu_total = mem_info.total / 1024**2

    log_line = f"[{datetime.now()}][{stage}] RAM={ram_usage:.2f}MB, GPU={gpu_used:.2f}/{gpu_total:.2f}MB"
    print(log_line)
    with open(os.path.join(OUTPUT_DIR, "memory_usage.log"), "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

# ================== 2. setting paramerters ==================

BATCH_SIZE = 64
EPOCHS = 3
LR = 2e-5
MAX_LEN = 128

NUM_EXPERTS = 6
TOP_K = 2
DROPOUT = 0.1

MODEL_NAME = "microsoft/deberta-base" # Hugggingface model path

# ================== 3. 資料集載入: 使用 train split (SST-2, MNLI, CLINC-OOS) & 切8:1:1 ==================

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

# ================== 4. 建立 Dataset 與 DataLoader ==================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class MoEDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        return {
            "text": ex["text"],
            "label_id": ex["label_id"],
            "dataset": ex["dataset"]
        }

def collate_fn(batch):
    texts = [ex["text"] for ex in batch]
    labels = [ex["label_id"] for ex in batch]
    datasets = [ex["dataset"] for ex in batch]
    
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    encoding["labels"] = torch.tensor(labels, dtype=torch.long)
    encoding["datasets"] = datasets
    return encoding

# 建立 Dataset 物件
train_dataset = MoEDataset(train_data)
valid_dataset = MoEDataset(valid_data)
test_dataset = MoEDataset(test_data)

# 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 分別建立 test_loader 的子集
sst2_test_loader = DataLoader(MoEDataset(test_sst2), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
mnli_test_loader = DataLoader(MoEDataset(test_mnli), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
clinc_test_loader = DataLoader(MoEDataset(test_clinc), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ================== 5. 定義 MoE 層與 SwiGLU ==================

class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)

def gumbel_top2(logits, temperature=1.0):
    """
    為每個樣本加上 Gumbel noise，並選出 Top-2 的專家。
    """
    # logits: (batch_size, num_experts)
    noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = (logits + noise) / temperature
    vals, idxs = torch.topk(y, k=TOP_K, dim=-1)
    return vals, idxs

class MoEFeedForward(nn.Module):
    def __init__(self, hidden_dim, inter_dim, num_experts=6, dropout=0.1):
        super(MoEFeedForward, self).__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts)
        self.dropout = nn.Dropout(dropout)
        
        # define 每個專家的layer
        self.experts_in = nn.ModuleList([
            nn.Linear(hidden_dim, inter_dim * 2) for _ in range(num_experts)
        ])
        self.experts_act = nn.ModuleList([
            SwiGLU() for _ in range(num_experts)
        ])
        self.experts_out = nn.ModuleList([
            nn.Linear(inter_dim, hidden_dim) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        """
        x: (batch_size, hidden_dim)
        """
        logits = self.router(x)  # (batch_size, num_experts)
        vals, idxs = gumbel_top2(logits)  # vals: (batch_size, 2), idxs: (batch_size, 2)
        
        batch_size = x.size(0)
        hidden_dim = x.size(1)
        
        # 儲存輸出
        out = torch.zeros_like(x)
        
        for i in range(batch_size):
            selected_experts = idxs[i]  # (2,)
            weights = torch.softmax(vals[i], dim=0)  # (2,)
            expert_output = 0
            for k in range(TOP_K):
                expert_id = selected_experts[k].item()
                expert_in = self.experts_in[expert_id](x[i].unsqueeze(0))  # (1, inter_dim*2)
                expert_act = self.experts_act[expert_id](expert_in)  # (1, inter_dim)
                expert_out = self.experts_out[expert_id](expert_act)  # (1, hidden_dim)
                expert_output += weights[k] * expert_out
            out[i] = expert_output.squeeze(0)
        
        out = self.dropout(out)
        return out

# ================== 6. 定義模型 ==================

class DebertaMoEModel(nn.Module):
    def __init__(self, model_name, num_labels, num_experts=6, dropout=0.1):
        super(DebertaMoEModel, self).__init__()
        self.num_labels = num_labels
        self.base_model = AutoModel.from_pretrained(model_name)
        
        hidden_dim = self.base_model.config.hidden_size
        inter_dim = self.base_model.config.intermediate_size
        
        # 定義 MoE 層
        self.moe_ffn = MoEFeedForward(hidden_dim, inter_dim, num_experts=num_experts, dropout=dropout)
        
        # define classification head
        self.classifier = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        labels: (batch_size,)
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        cls_token = last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
        
        # MoE FFN
        moe_out = self.moe_ffn(cls_token)  # (batch_size, hidden_dim)
        
        # 分類
        logits = self.classifier(moe_out)  # (batch_size, num_labels)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return logits, loss

# ================== 7. 建立模型、優化器、日誌 ==================

model = DebertaMoEModel(MODEL_NAME, num_labels=num_labels, num_experts=NUM_EXPERTS, dropout=DROPOUT)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


training_log = []

best_val_acc = 0.0

# ================== 8. 定義評估函式 ==================

def evaluate_accuracy(dataloader):
    """
    計算給定 DataLoader 的準確度
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    return acc

# ================== 9. 定義預測並儲存測試結果 ==================

def predict_and_save(dataloader, data_list, dataset_name):
    """
    對指定的 DataLoader 進行預測，並將結果儲存為 JSON。
    """
    model.eval()
    results = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            datasets = batch["datasets"]
            
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            for i in range(len(preds)):
                results.append({
                    "dataset": datasets[i],
                    "text": data_list[i]["text"],
                    "true_label": id2label[labels[i]],
                    "pred_label": id2label[preds[i]]
                })
    
    # 儲存為 JSON
    with open(os.path.join(OUTPUT_DIR, f"{dataset_name}_test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Saved predictions for {dataset_name} to {os.path.join(OUTPUT_DIR, f'{dataset_name}_test_predictions.json')}")

# ================== 10. 訓練迴圈 ==================

print("[INFO] Starting training...")

log_memory("Before Training")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total_steps = 0
    
    # 使用tqdm顯示進度條
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)
    
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_steps += 1
        
        # 更新loss
        loop.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_train_loss = total_loss / total_steps if total_steps > 0 else 0.0
    val_acc = evaluate_accuracy(valid_loader)
    
    log_line = f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Valid Acc: {val_acc:.4f}"
    print(log_line)
    training_log.append(log_line)
    
    log_memory(f"After Epoch {epoch}")
    
    # save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "moe_model_best.pt"))
        print(f"[INFO] Saved best model with val_acc: {best_val_acc:.4f}")
    
    # 記錄訓練進度
    with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

log_memory("After Training")

# save model
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "moe_model_last.pt"))
print(f"[INFO] Saved last model to {os.path.join(OUTPUT_DIR, 'moe_model_last.pt')}")

# ================== 11. 評估測試集 ==================

print("\n[INFO] Evaluating on Test Set...")

# 整合測試集的acc
overall_test_acc = evaluate_accuracy(test_loader)
print(f"Overall Test Accuracy: {overall_test_acc:.4f}")

# 分別對 SST-2, MNLI, CLINC-OOS 測試集計算acc
acc_sst2 = evaluate_accuracy(sst2_test_loader)
acc_mnli = evaluate_accuracy(mnli_test_loader)
acc_clinc = evaluate_accuracy(clinc_test_loader)

print(f"SST-2 Test Accuracy: {acc_sst2:.4f}")
print(f"MNLI Test Accuracy: {acc_mnli:.4f}")
print(f"CLINC-OOS Test Accuracy: {acc_clinc:.4f}")

# 將測試結果寫入log
test_log = f"Test Accuracies | Overall: {overall_test_acc:.4f}, SST-2: {acc_sst2:.4f}, MNLI: {acc_mnli:.4f}, CLINC-OOS: {acc_clinc:.4f}"
print(test_log)
training_log.append(test_log)

with open(os.path.join(OUTPUT_DIR, "training_log.txt"), "a", encoding="utf-8") as f:
    f.write(test_log + "\n")

log_memory("After Testing")

# ================== 12. 預測並儲存測試結果 ==================

print("\n[INFO] Generating and saving predictions...")

# 預測並儲存結果
def generate_and_save_predictions(dataloader, data_list, dataset_name):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting {dataset_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            for i in range(len(preds)):
                predictions.append({
                    "text": data_list[i]["text"],
                    "dataset": data_list[i]["dataset"],
                    "true_label": id2label[labels[i]],
                    "pred_label": id2label[preds[i]]
                })
    
    # 儲存為 JSON
    with open(os.path.join(OUTPUT_DIR, f"{dataset_name}_test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Saved predictions for {dataset_name} to {os.path.join(OUTPUT_DIR, f'{dataset_name}_test_predictions.json')}")

# 各資料集的預測結果
generate_and_save_predictions(test_loader, test_data, "overall_test")
generate_and_save_predictions(sst2_test_loader, test_sst2, "sst2")
generate_and_save_predictions(mnli_test_loader, test_mnli, "mnli")
generate_and_save_predictions(clinc_test_loader, test_clinc, "clinc_oos")

print("\n[INFO] All results have been saved to the output directory.")
