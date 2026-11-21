#!/usr/bin/env python3
"""
Sliding Window Chunking Finance Retrieval Experiment Script
測試滑動窗口切片 (Sliding Window Chunking + Max Pooling) 對金融檢索準確率的提升效果。
"""

import argparse
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

class ChunkedFinanceEvaluator:
    def __init__(self, model_name: str = "TencentBAC/Conan-embedding-v1"):
        print(f"載入 Dense Vector 模型: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def chunk_text(self, text: str, chunk_size: int = 384, overlap: int = 128) -> List[str]:
        """
        滑動窗口切片 (Sliding Window Chunking)
        """
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

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeds = outputs.last_hidden_state[:, 0, :]
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
                all_embeds.append(embeds.cpu().numpy())
        return np.vstack(all_embeds)

    def get_doc_dense_score_max_pooling(self, query: str, raw_texts: List[str], chunk_size: int = 384, overlap: int = 128) -> np.ndarray:
        """
        對每個文件進行滑動窗口切片，各自計算向量後取 Max Pooling 代表該文檔得分
        """
        q_emb = self.encode_texts([query])[0]
        
        doc_scores = []
        for text in raw_texts:
            chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            chunk_embeds = self.encode_texts(chunks)
            sims = np.dot(chunk_embeds, q_emb)
            max_score = float(sims.max())
            doc_scores.append(max_score)
            
        return np.array(doc_scores)

def run_experiment_on_dataset(dataset_name: str, evaluator: ChunkedFinanceEvaluator, chunk_size: int = 384, overlap: int = 128):
    q_path = PROJECT_ROOT / f"data/{dataset_name}/questions.json"
    gt_path = PROJECT_ROOT / f"data/{dataset_name}/ground_truths.json"
    fin_path = PROJECT_ROOT / "data/processed/processed_finance_data.json"

    if not q_path.exists() or not gt_path.exists():
        print(f"找不到數據集: {dataset_name}")
        return

    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f).get("questions", [])

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f).get("ground_truths", [])
        gt_map = {item["qid"]: item["retrieve"] for item in gt_raw}

    with open(fin_path, "r", encoding="utf-8") as f:
        fin_data = json.load(f)
        fin_doc_map = {int(item["id"]): item["text"] for item in fin_data if str(item.get("id")).isdigit()}

    fin_questions = [q for q in questions if q.get("category") == "finance"]

    correct_baseline_bm25 = 0
    correct_truncated_dense = 0
    correct_chunked_dense = 0
    correct_chunked_hybrid = 0
    total = len(fin_questions)

    for item in fin_questions:
        qid = item["qid"]
        query = item["query"]
        source = item.get("source", [])
        gt_pid = gt_map.get(qid)

        pids_to_search = source if source else list(fin_doc_map.keys())

        cand_pids = []
        cand_corpus_tokens = []
        raw_texts = []

        for pid in pids_to_search:
            text = fin_doc_map.get(pid, "")
            if text:
                cand_pids.append(pid)
                raw_texts.append(text)
                cand_corpus_tokens.append(list(jieba.cut(text)))

        if not cand_corpus_tokens:
            continue

        query_tokens = list(jieba.cut(query))

        # 1. Baseline BM25
        bm25 = BM25Okapi(cand_corpus_tokens)
        bm25_scores = bm25.get_scores(query_tokens)
        pred_base_bm25 = cand_pids[int(bm25_scores.argmax())]

        # 2. 舊版：前 512 字截斷 Dense
        q_emb = evaluator.encode_texts([query])[0]
        trunc_texts = [t[:512] for t in raw_texts]
        trunc_embeds = evaluator.encode_texts(trunc_texts)
        trunc_dense_scores = np.dot(trunc_embeds, q_emb)
        pred_trunc_dense = cand_pids[int(trunc_dense_scores.argmax())]

        # 3. 新版：Sliding Window Chunking + Max Pooling Dense
        chunked_dense_scores = evaluator.get_doc_dense_score_max_pooling(query, raw_texts, chunk_size=chunk_size, overlap=overlap)
        pred_chunked_dense = cand_pids[int(chunked_dense_scores.argmax())]

        # 4. 新版：Chunked Dense + BM25 Hybrid
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-8)
        chunked_dense_norm = (chunked_dense_scores - chunked_dense_scores.min()) / (chunked_dense_scores.max() - chunked_dense_scores.min() + 1e-8)
        chunked_hybrid_scores = 0.4 * bm25_norm + 0.6 * chunked_dense_norm
        pred_chunked_hybrid = cand_pids[int(chunked_hybrid_scores.argmax())]

        if pred_base_bm25 == gt_pid:
            correct_baseline_bm25 += 1
        if pred_trunc_dense == gt_pid:
            correct_truncated_dense += 1
        if pred_chunked_dense == gt_pid:
            correct_chunked_dense += 1
        if pred_chunked_hybrid == gt_pid:
            correct_chunked_hybrid += 1

    print("\n" + "=" * 80)
    print(f"  Sliding Window Chunking Experiment Results on [{dataset_name}]")
    print(f"  (Chunk Size: {chunk_size}, Overlap: {overlap})")
    print("=" * 80)
    print(f"Total Finance Questions                              : {total}")
    print(f"1. Baseline BM25                                    : {correct_baseline_bm25}/{total} ({correct_baseline_bm25/total*100:.2f}%)")
    print(f"2. 舊版 512 字硬截斷 Dense                          : {correct_truncated_dense}/{total} ({correct_truncated_dense/total*100:.2f}%)")
    print(f"3. 新版 滑動窗口切片 Dense (Sliding Window Chunk)    : {correct_chunked_dense}/{total} ({correct_chunked_dense/total*100:.2f}%)")
    print(f"4. 新版 滑動窗口切片 Hybrid (Chunked Dense + BM25)   : {correct_chunked_hybrid}/{total} ({correct_chunked_hybrid/total*100:.2f}%)")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["validation_sample", "benchmark_experiment", "all"])
    parser.add_argument("--chunk_size", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    evaluator = ChunkedFinanceEvaluator()

    if args.dataset == "all":
        run_experiment_on_dataset("benchmark_experiment", evaluator, chunk_size=args.chunk_size, overlap=args.overlap)
        run_experiment_on_dataset("validation_sample", evaluator, chunk_size=args.chunk_size, overlap=args.overlap)
    else:
        run_experiment_on_dataset(args.dataset, evaluator, chunk_size=args.chunk_size, overlap=args.overlap)

if __name__ == "__main__":
    main()
