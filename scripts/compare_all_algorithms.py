#!/usr/bin/env python3
"""
Compare All Algorithms Script
在同一個指令下實測並對比 4 種演算法的真實檢索準確率：
1. Alg 0: Baseline BM25 (原始 BM25)
2. Alg 1: BM25 + Rule-based Time Boosting (粗暴硬規則加分)
3. Alg 2: Dense Vector Only (前 512 字硬截斷向量)
4. Alg 3: Sliding Window Hybrid (滑動窗口切片 + Dense + BM25 + Max-Pooling)
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import jieba
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class AlgorithmComparator:
    def __init__(self, model_name: str = "TencentBAC/Conan-embedding-v1"):
        print("載入模型與準備對比實驗環境...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        emb_path = PROJECT_ROOT / "data/processed/finance_chunk_embeddings.npy"
        meta_path = PROJECT_ROOT / "data/processed/finance_chunk_metadata.json"
        fin_path = PROJECT_ROOT / "data/processed/processed_finance_data.json"

        with open(fin_path, "r", encoding="utf-8") as f:
            fin_data = json.load(f)
            self.fin_doc_map = {int(item["id"]): item["text"] for item in fin_data if str(item.get("id")).isdigit()}

        self.has_precomputed = emb_path.exists() and meta_path.exists()
        if self.has_precomputed:
            self.embeddings = np.load(emb_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.pid_to_chunk_indices = {}
            for idx, item in enumerate(self.metadata):
                pid = item["pid"]
                if pid not in self.pid_to_chunk_indices:
                    self.pid_to_chunk_indices[pid] = []
                self.pid_to_chunk_indices[pid].append(idx)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeds = outputs.last_hidden_state[:, 0, :]
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            return embeds.cpu().numpy()

    def run_comparison(self, dataset_name: str):
        q_path = PROJECT_ROOT / f"data/{dataset_name}/questions.json"
        gt_path = PROJECT_ROOT / f"data/{dataset_name}/ground_truths.json"

        with open(q_path, "r", encoding="utf-8") as f:
            questions = json.load(f).get("questions", [])

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_raw = json.load(f).get("ground_truths", [])
            gt_map = {item["qid"]: item["retrieve"] for item in gt_raw}

        fin_questions = [q for q in questions if q.get("category") == "finance"]
        total = len(fin_questions)

        correct_alg0 = 0 # Baseline BM25
        correct_alg1 = 0 # BM25 + Rule Boosting
        correct_alg2 = 0 # Truncated Dense Only
        correct_alg3 = 0 # Sliding Window Hybrid (Our Proposed)

        for item in fin_questions:
            qid = item["qid"]
            query = item["query"]
            source = item.get("source", [])
            gt_pid = gt_map.get(qid)

            pids_to_search = source if source else list(self.fin_doc_map.keys())

            cand_pids = []
            cand_corpus_tokens = []
            raw_texts = []

            for pid in pids_to_search:
                text = self.fin_doc_map.get(pid, "")
                if text:
                    cand_pids.append(pid)
                    raw_texts.append(text)
                    cand_corpus_tokens.append(list(jieba.cut(text)))

            if not cand_pids:
                continue

            query_tokens = list(jieba.cut(query))

            # --- Alg 0: Baseline BM25 ---
            bm25 = BM25Okapi(cand_corpus_tokens)
            bm25_scores = bm25.get_scores(query_tokens)
            pred_alg0 = cand_pids[int(bm25_scores.argmax())]

            # --- Alg 1: BM25 + Rule-based Time Boosting (+4.0) ---
            scores_alg1 = bm25_scores.copy()
            year_match = "111" if ("2022" in query or "111" in query) else ("112" if ("2023" in query or "112" in query) else None)
            if year_match:
                for idx, text in enumerate(raw_texts):
                    if year_match in text:
                        scores_alg1[idx] += 4.0
            pred_alg1 = cand_pids[int(scores_alg1.argmax())]

            # --- Alg 2: Truncated Dense Only (前 512 字) ---
            q_emb = self.encode_texts([query])[0]
            trunc_texts = [t[:512] for t in raw_texts]
            trunc_embeds = self.encode_texts(trunc_texts)
            dense_trunc_scores = np.dot(trunc_embeds, q_emb)
            pred_alg2 = cand_pids[int(dense_trunc_scores.argmax())]

            # --- Alg 3: Proposed Sliding Window Hybrid (Chunked Dense + BM25 + Max Pooling) ---
            bm25_norm = bm25_scores / (bm25_scores.max() + 1e-8)

            if self.has_precomputed:
                dense_chunk_scores = []
                for pid in cand_pids:
                    c_indices = self.pid_to_chunk_indices.get(pid, [])
                    if c_indices:
                        chunk_embeds = self.embeddings[c_indices]
                        sims = np.dot(chunk_embeds, q_emb)
                        max_sim = float(sims.max())
                    else:
                        max_sim = 0.0
                    dense_chunk_scores.append(max_sim)
                dense_chunk_scores = np.array(dense_chunk_scores)
            else:
                dense_chunk_scores = dense_trunc_scores

            dense_norm = (dense_chunk_scores - dense_chunk_scores.min()) / (dense_chunk_scores.max() - dense_chunk_scores.min() + 1e-8)
            hybrid_scores = 0.4 * bm25_norm + 0.6 * dense_norm
            pred_alg3 = cand_pids[int(hybrid_scores.argmax())]

            if pred_alg0 == gt_pid: correct_alg0 += 1
            if pred_alg1 == gt_pid: correct_alg1 += 1
            if pred_alg2 == gt_pid: correct_alg2 += 1
            if pred_alg3 == gt_pid: correct_alg3 += 1

        print("\n" + "=" * 80)
        print(f"  Live Comparative Experiment Output on Dataset: [{dataset_name}]")
        print("=" * 80)
        print(f"Total Questions Evaluated                         : {total}")
        print(f"Alg 0: Baseline BM25                              : {correct_alg0}/{total} ({correct_alg0/total*100:.2f}%)")
        print(f"Alg 1: BM25 + Rule Time Boosting (+4.0)            : {correct_alg1}/{total} ({correct_alg1/total*100:.2f}%)")
        print(f"Alg 2: Dense Only (前 512 字截斷)                 : {correct_alg2}/{total} ({correct_alg2/total*100:.2f}%)")
        print(f"Alg 3: Proposed Sliding Window Hybrid (本專案成效) : {correct_alg3}/{total} ({correct_alg3/total*100:.2f}%)")
        print("=" * 80)

def main():
    comparator = AlgorithmComparator()
    comparator.run_comparison("benchmark_experiment")
    comparator.run_comparison("validation_sample")

if __name__ == "__main__":
    main()
