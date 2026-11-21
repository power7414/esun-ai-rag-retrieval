#!/usr/bin/env python3
"""
Partitioned Offline Finance Embeddings Build Script (with tqdm for each Part)
分批段建檔 + 斷點續跑 + 散熱冷卻 + tqdm 即時進度條
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUM_PARTS = 6

class FinanceEmbeddingBuilder:
    def __init__(self, model_name: str = "TencentBAC/Conan-embedding-v1"):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.use_fp16 = True
            print(f"🚀 開啟極速硬體加速: Apple Silicon Metal GPU (MPS) + FP16 半精度模式")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.use_fp16 = True
            print(f"🚀 開啟極速硬體加速: NVIDIA CUDA GPU + FP16 半精度模式")
        else:
            self.device = torch.device("cpu")
            self.use_fp16 = False
            num_cores = os.cpu_count() or 8
            torch.set_num_threads(num_cores)
            print(f"ℹ️ 使用 CPU 8 核心向量加速模式")

        print(f"載入向量模型: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.use_fp16:
            self.model = AutoModel.from_pretrained(model_name).to(self.device).half()
        else:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            
        self.model.eval()

    def clean_text(self, text: str) -> str:
        return "".join(text.split())

    def chunk_text(self, text: str, chunk_size: int = 384, overlap: int = 96) -> List[str]:
        text = self.clean_text(text)
        if not text:
            return [""]
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    def encode_texts(self, texts: List[str], part_idx: int, batch_size: int = 32) -> np.ndarray:
        all_embeds = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        batch_iterable = range(0, len(texts), batch_size)
        if tqdm is not None:
            batch_iterable = tqdm(batch_iterable, desc=f"  Part {part_idx}/{NUM_PARTS} 進度", total=total_batches, unit="batch", leave=False)

        for i in batch_iterable:
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=384, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
                embeds = outputs.last_hidden_state[:, 0, :]
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
                all_embeds.append(embeds.to(torch.float32).cpu().numpy())

            del inputs, outputs
            if self.device.type == "mps":
                torch.mps.empty_cache()

        return np.vstack(all_embeds)

def build():
    fin_path = PROJECT_ROOT / "data/processed/processed_finance_data.json"
    out_emb_path = PROJECT_ROOT / "data/processed/finance_chunk_embeddings.npy"
    out_meta_path = PROJECT_ROOT / "data/processed/finance_chunk_metadata.json"
    temp_dir = PROJECT_ROOT / "data/processed/part_checkpoints"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not fin_path.exists():
        print(f"找不到檔案: {fin_path}")
        return

    with open(fin_path, "r", encoding="utf-8") as f:
        fin_data = json.load(f)

    valid_data = [item for item in fin_data if str(item.get("id")).isdigit()]
    total_docs = len(valid_data)

    print(f"將 {total_docs} 筆金融文件分為 {NUM_PARTS} 個小分段執行 (每分段約 {total_docs // NUM_PARTS} 頁)...")
    
    doc_splits = np.array_split(valid_data, NUM_PARTS)
    builder = None

    part_emb_files = []
    all_metadata = []

    for part_idx, part_docs in enumerate(doc_splits, 1):
        part_emb_file = temp_dir / f"finance_part_{part_idx}.npy"
        part_meta_file = temp_dir / f"finance_part_{part_idx}.json"
        part_emb_files.append((part_emb_file, part_meta_file))

        if part_emb_file.exists() and part_meta_file.exists():
            print(f"⏩ [Part {part_idx}/{NUM_PARTS}] 已於先前完成，自動跳過！")
            with open(part_meta_file, "r", encoding="utf-8") as f:
                all_metadata.extend(json.load(f))
            continue

        print(f"\n▶️ [Part {part_idx}/{NUM_PARTS}] 開始處理 {len(part_docs)} 頁文件...")
        
        if builder is None:
            builder = FinanceEmbeddingBuilder()

        part_chunks = []
        part_metadata = []

        for item in part_docs:
            pid = int(item["id"])
            text = item.get("text", "")
            chunks = builder.chunk_text(text, chunk_size=384, overlap=96)
            for c_idx, c_text in enumerate(chunks):
                part_chunks.append(c_text)
                part_metadata.append({
                    "pid": pid,
                    "chunk_idx": c_idx
                })

        print(f"  [Part {part_idx}] 切出 {len(part_chunks)} 個 Chunks，開始編碼...")
        start_t = time.time()
        part_embeds = builder.encode_texts(part_chunks, part_idx=part_idx, batch_size=32)
        end_t = time.time()

        print(f"  [Part {part_idx}] ✅ 完成！耗時: {end_t - start_t:.2f} 秒。寫入暫存檔...")
        np.save(str(part_emb_file), part_embeds)
        with open(part_meta_file, "w", encoding="utf-8") as f:
            json.dump(part_metadata, f, ensure_ascii=False)

        all_metadata.extend(part_metadata)

        if part_idx < NUM_PARTS:
            print(f"  ❄️ 散熱冷卻中... 休息 5 秒讓 MacBook Air 恢復溫度...")
            time.sleep(5)

    print("\n" + "=" * 70)
    print("所有分段處理完成！開始拼合為最終向量資料庫...")
    
    all_emb_list = []
    for p_emb_f, _ in part_emb_files:
        all_emb_list.append(np.load(p_emb_f))
    
    final_embeddings = np.vstack(all_emb_list)

    print(f"寫入最終向量檔案至: {out_emb_path} (矩陣維度: {final_embeddings.shape})")
    np.save(str(out_emb_path), final_embeddings)

    print(f"寫入最終 Metadata 至: {out_meta_path}")
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False)

    # 清理暫存資料夾
    for p_emb_f, p_meta_f in part_emb_files:
        if p_emb_f.exists(): p_emb_f.unlink()
        if p_meta_f.exists(): p_meta_f.unlink()
    if temp_dir.exists():
        temp_dir.rmdir()

    print("🎉 離線金融向量資料庫 (embeddings.npy & metadata.json) 拼合預處理完全成功！")

if __name__ == "__main__":
    build()
