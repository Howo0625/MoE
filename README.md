# **語言模型專家系統分流機制與聚合機制**
本實驗包含四種程式碼：

>MoE (Mixture of Experts) 版本

>LoRA (Low-Rank Adaptation) Fine-tune 版本，模型包含 Deberta-large / Deberta-v2-xlarge

>Full Finetune 版本，模型 Deberta-large

Datasets則使用到 SST-2、MNLI、CLINC-OOS 三個資料集，用以示範多種微調方式在同一框架下的訓練流程、測試流程，以及對不同資料集的合併與評估。

以下是程式碼的簡要使用說明。
##  1. Installation
實驗環境以及套件主要使用 Python 3.8+、PyTorch、Hugging Face Transformers、Hugging Face datasets、以及一些工具套件如 scikit-learn, psutil, pynvml, tqdm.....

可以透過以下程式碼建立虛擬環境 MoE_Proj，並完成requirements的套件的安裝。

    conda create --name MoE_Proj python=3.9
    conda activate MoE_Proj
    pip install -r requirements.txt

## 2. Datasets
程式中使用了 Hugging Face 的 datasets 來下載 SST-2, MNLI, CLINC-OOS 三個資料集。
>SST-2：load_dataset("glue", "sst2") , 2-Class

>MNLI： load_dataset("glue", "mnli") , 3-Class

>CLINC-OOS：load_dataset("clinc_oos", "small") , 151-Class

都已包含在每個執行程式中，datasets.load_dataset(...) 會自動從Hugging Face Hub下載，並做資料集切分以及label mapping。


_因MNLI以及SST-2的Glue資料集通常不會給testing的真實label，所以這裡會使用 training 資料去再切分成 training / valid / testing 資料。_

## Output
無論哪一個版本的程式碼輸出結果都會生成
1. train_split.csv, valid_split.csv, test_split.csv的資料集
2. memory_usage.log：記錄RAM & GPU usage
3. model 
4. training_log.txt：把每個 epoch 的 loss、accuracy、最終測試結果等資訊寫入此檔案。
   
       # Example
       Epoch 1, train_loss=0.4574, val_acc=0.8783
       Test => Overall:0.8795, SST-2:0.9414, MNLI:0.8772, CLINC:0.4513 

## 如何執行
完成requirements.txt的環境建置，並將以下四個.py檔下載到根目錄
1. MoE_DeBERTa_base.py — MoE 版本
2. LoRA_Deberta_large.py — LoRA Fine-tune 版本
3. LoRA_Deberta_v2_xlarge.py - LoRA Fine-tune 版本
4. FullFT_DebertaLarge.py — Full Finetune 版本
   
       # 執行 MoE 版本
        python MoE_DeBERTa_base.py
        
        # 執行 LoRA 版本
        python LoRA_Deberta_large.py
        python LoRA_Deberta_v2_xlarge.py
        
        # 執行 Full Finetune 版本 (deberta-large)
        python FullFT_DebertaLarge.py



