#!/usr/bin/env python3
"""
Fast Offline Pre-embedded Finance Inference Test Script
使用預先算好的 finance_chunk_embeddings.npy 與 metadata 測試超高速推論與準確率。
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import jieba
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class FastPreembeddedFinanceRetriever:
    def __init__(self, model_name: str = "TencentBAC/Conan-embedding-v1"):
        print("載入離線預計算向量矩陣與 Metadata...")
        emb_path = PROJECT_ROOT / "data/processed/finance_chunk_embeddings.npy"
        meta_path = PROJECT_ROOT / "data/processed/finance_chunk_metadata.json"
        fin_path = PROJECT_ROOT / "data/processed/processed_finance_data.json"

        self.embeddings = np.load(emb_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        with open(fin_path, "r", encoding="utf-8") as f:
            fin_data = json.load(f)
            self.fin_doc_map = {int(item["id"]): item["text"] for item in fin_data if str(item.get("id")).isdigit()}

        # 建立 pid 到 chunk index 列表的索引映照 (PID -> [chunk_indices])
        self.pid_to_chunk_indices = {}
        for idx, item in enumerate(self.metadata):
            pid = item["pid"]
            if pid not in self.pid_to_chunk_indices:
                self.pid_to_chunk_indices[pid] = []
            self.pid_to_chunk_indices[pid].append(idx)

        print(f"載入完成！共有 {len(self.metadata)} 個 Chunk 向量，涵蓋 {len(self.pid_to_chunk_indices)} 份 PID 文件。")
        print(f"載入 Query 向量模型: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode_query(self, query: str) -> np.ndarray:
        inputs = self.tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeds = outputs.last_hidden_state[:, 0, :]
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            return embeds.cpu().numpy()[0]

    def predict_finance_fast(self, query: str, source: List[int]) -> int:
        pids_to_search = source if source else list(self.pid_to_chunk_indices.keys())

        # 1. 計算 BM25
        cand_pids = []
        cand_corpus_tokens = []
        for pid in pids_to_search:
            text = self.fin_doc_map.get(pid, "")
            if text:
                cand_pids.append(pid)
                cand_corpus_tokens.append(list(jieba.cut(text)))

        if not cand_pids:
            return pids_to_search[0] if pids_to_search else 0

        query_tokens = list(jieba.cut(query))
        bm25 = BM25Okapi(cand_corpus_tokens)
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-8)

        # 2. 超高速向量檢索 (直接點積已載入的向量矩陣)
        q_emb = self.encode_query(query)

        dense_scores = []
        for pid in cand_pids:
            chunk_indices = self.pid_to_chunk_indices.get(pid, [])
            if chunk_indices:
                chunk_embeds = self.embeddings[chunk_indices]
                sims = np.dot(chunk_embeds, q_emb)
                max_sim = float(sims.max())
            else:
                max_sim = 0.0
            dense_scores.append(max_sim)

        dense_scores = np.array(dense_scores)
        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)

        # 3. Hybrid Scoring
        hybrid_scores = 0.4 * bm25_norm + 0.6 * dense_norm
        best_idx = int(hybrid_scores.argmax())
        return cand_pids[best_idx]

def test_speed():
    retriever = FastPreembeddedFinanceRetriever()

    q_path = PROJECT_ROOT / "data/benchmark_experiment/questions.json"
    gt_path = PROJECT_ROOT / "data/benchmark_experiment/ground_truths.json"

    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f).get("questions", [])

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f).get("ground_truths", [])
        gt_map = {item["qid"]: item["retrieve"] for item in gt_raw}

    fin_questions = [q for q in questions if q.get("category") == "finance"]

    print(f"\n開始測試 {len(fin_questions)} 題金融問題之推論速度與準率...")
    start_t = time.time()

    correct = 0
    for q in fin_questions:
        qid = q["qid"]
        query = q["query"]
        source = q.get("source", [])
        gt_pid = gt_map.get(qid)

        pred_pid = retriever.predict_finance_fast(query, source)
        if pred_pid == gt_pid:
            correct += 1

    end_t = time.time()
    total_time = end_t - start_t
    avg_time = total_time / len(fin_questions)
    acc = correct / len(fin_questions) * 100.0

    print("=" * 70)
    print("  Fast Pre-embedded Inference Test Results")
    print("=" * 70)
    print(f"Total Questions Evaluated : {len(fin_questions)}")
    print(f"Accuracy Rate             : {correct}/{len(fin_questions)} ({acc:.2f}%)")
    print(f"Total Inference Time      : {total_time:.3f} seconds")
    print(f"Average Time per Question : {avg_time * 1000:.2f} ms / question")
    print(f"Estimated Time for 300 Qs : {avg_time * 300:.2f} seconds")
    print("=" * 70)

if __name__ == "__main__":
    test_speed()
