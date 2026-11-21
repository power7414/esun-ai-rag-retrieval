#!/usr/bin/env python3
"""
Standalone pipeline (earlier version; superseded by scripts/unified_pipeline.py).
Measured on the official 150-question example set (2026-11-07): overall 122/150 = 81.33%
1. FAQ: Conan-embedding-v1 question-to-question (Q2Q), CLS pooling   -> 96%
2. Insurance: BM25 over data/processed/processed_insurance_data.json with a 29-term lexicon -> 86%
3. Finance: BM25 over data/processed/processed_finance_data.json with the same lexicon    -> 62%
Needs data/processed/processed_{insurance,finance}_data.json (scripts/build_processed_corpus.py)
and data/reference/faq/pid_map_content.json.

支援指令列參數：
- 執行競賽推論 (Competition Mode):
  python scripts/standalone_rag_pipeline.py --mode competition
- 執行 Benchmark 指標評估 (Benchmark Mode):
  python scripts/standalone_rag_pipeline.py --mode eval
"""

import argparse
import json
import re
import os
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import jieba
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("StandaloneRAG")

DEFAULT_FINANCIAL_LEXICON = [
    "資本適足率", "淨值比率", "風險性資產", "第一類資本", "第二類資本",
    "流動性覆蓋比率", "槓桿比率", "金融消費者保護法", "洗錢防制法",
    "投資型保險", "人壽保險", "健康保險", "傷害保險", "年金保險",
    "解約金", "保險單借款", "寬限期間", "除外責任", "告知義務",
    "要保人", "被保險人", "受益人", "滿期保險金", "完全失能保險金",
    "營業利益", "合併財務報告", "綜合損益總額", "母公司業主", "資產總額"
]

class LexiconCleaner:
    def __init__(self, custom_lexicon: List[str] = None):
        self.lexicon = set(DEFAULT_FINANCIAL_LEXICON)
        if custom_lexicon:
            self.lexicon.update(custom_lexicon)
        for word in self.lexicon:
            jieba.add_word(word)

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        text = re.sub(r'民國\s*([0-9]{2,3})\s*年', r'\1年', text)
        tokens = list(jieba.cut(text))
        return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]

class StandaloneRAGPipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.cleaner = LexiconCleaner()

        self.ins_doc_map = {}
        self.fin_doc_map = {}
        self.faq_items = []
        self.faq_embeddings = None

        # Transformer 向量模型
        self.model_name = "TencentBAC/Conan-embedding-v1"
        self.tokenizer = None
        self.model = None

        self._load_processed_datasets()
        self._init_faq_embedding_engine()

    def _load_processed_datasets(self):
        proc_dir = self.data_dir / "processed"
        ins_path = proc_dir / "processed_insurance_data.json"
        fin_path = proc_dir / "processed_finance_data.json"

        if ins_path.exists():
            with open(ins_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.ins_doc_map = {int(item["id"]): item["text"] for item in data if str(item.get("id")).isdigit()}
            logger.info(f"載入保險預處理資料庫: {len(self.ins_doc_map)} 筆")

        if fin_path.exists():
            with open(fin_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.fin_doc_map = {int(item["id"]): item["text"] for item in data if str(item.get("id")).isdigit()}
            logger.info(f"載入金融預處理資料庫: {len(self.fin_doc_map)} 筆")

    def _init_faq_embedding_engine(self):
        logger.info(f"載入 FAQ 向量模型: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as e:
            logger.error(f"載入 {self.model_name} 模型失敗: {e}")
            return

        faq_json_path = self.data_dir / "reference" / "faq" / "pid_map_content.json"
        if not faq_json_path.exists():
            faq_json_path = self.data_dir / "competition" / "reference" / "faq" / "pid_map_content.json"
        if not faq_json_path.exists():
            faq_json_path = self.data_dir / "benchmark_experiment" / "reference" / "faq" / "pid_map_content.json"

        if not faq_json_path.exists():
            logger.warning("找不到 FAQ pid_map_content.json 檔案")
            return

        with open(faq_json_path, "r", encoding="utf-8") as f:
            faq_raw = json.load(f)

        self.faq_items = []
        for pid_str, items in faq_raw.items():
            pid = int(pid_str)
            if isinstance(items, list):
                for item in items:
                    q = item.get("question", "")
                    if q:
                        self.faq_items.append({"pid": pid, "question": q})

        logger.info(f"開始編碼 FAQ 資料庫 (共 {len(self.faq_items)} 筆問題)...")
        faq_questions = [item["question"] for item in self.faq_items]
        self.faq_embeddings = self._encode_texts(faq_questions)
        logger.info("FAQ 語意向量資料庫編碼建構完成！")

    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
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

    def retrieve_single(self, query: str, category: str, source: List[int]) -> int:
        if not query:
            return source[0] if source else 0

        # --- Strategy 1: FAQ Tencent Conan Embedding Q2Q Vector Match ---
        if category == "faq":
            candidates = self.faq_items
            cand_indices = list(range(len(self.faq_items)))
            if source:
                source_set = set(source)
                cand_indices = [i for i, item in enumerate(self.faq_items) if item["pid"] in source_set]

            if not cand_indices:
                return source[0] if source else 0

            q_emb = self._encode_texts([query])[0]
            cand_embeds = self.faq_embeddings[cand_indices]
            sims = np.dot(cand_embeds, q_emb)
            best_cand_idx = sims.argmax()
            return self.faq_items[cand_indices[best_cand_idx]]["pid"]

        # --- Strategy 2: 保險與金融 Processed Lexicon BM25 Match ---
        else:
            doc_map = self.ins_doc_map if category == "insurance" else self.fin_doc_map
            pids_to_search = source if source else list(doc_map.keys())

            cand_pids = []
            cand_corpus = []
            for pid in pids_to_search:
                text = doc_map.get(pid, "")
                if text:
                    cand_pids.append(pid)
                    cand_corpus.append(self.cleaner.tokenize(text))

            if not cand_corpus:
                return pids_to_search[0] if pids_to_search else 0

            query_tokens = self.cleaner.tokenize(query)
            if not query_tokens:
                return cand_pids[0]

            bm25 = BM25Okapi(cand_corpus)
            scores = bm25.get_scores(query_tokens)
            best_idx = int(scores.argmax())
            return cand_pids[best_idx]

    def predict_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        answers = []
        for item in questions:
            qid = item["qid"]
            query = item.get("query", "")
            category = item.get("category", "")
            source = item.get("source", [])

            best_pid = self.retrieve_single(query, category, source)
            answers.append({
                "qid": qid,
                "retrieve": best_pid
            })
        return answers

def main():
    parser = argparse.ArgumentParser(description="Standalone Hybrid RAG Pipeline")
    parser.add_argument("--mode", type=str, choices=["competition", "eval"], default="competition", help="Execution mode")
    parser.add_argument("--question_path", type=str, default=None, help="Input questions JSON path")
    parser.add_argument("--output_path", type=str, default=None, help="Output prediction JSON path")
    parser.add_argument("--gt_path", type=str, default=None, help="Ground truths JSON path for evaluation")
    args = parser.parse_args()

    if args.mode == "competition":
        q_path = args.question_path or "data/competition/questions.json"
        out_path = args.output_path or "data/competition/pred_retrieve.json"

        logger.info(f"== 啟動正式競賽推論 (Competition Mode) ==")
        logger.info(f"問題檔案: {q_path}")
        logger.info(f"輸出目標: {out_path}")

        with open(q_path, "r", encoding="utf-8") as f:
            questions_raw = json.load(f).get("questions", [])

        pipeline = StandaloneRAGPipeline()
        answers = pipeline.predict_questions(questions_raw)

        # 依照官方規格輸出
        output_payload = {"answers": answers}
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=4)

        logger.info(f"競賽推論完成！共產出 {len(answers)} 筆預測結果至: {out_path}")

    elif args.mode == "eval":
        q_path = args.question_path or "data/dataset/preliminary/questions_example.json"
        gt_path = args.gt_path or "data/dataset/preliminary/ground_truths_example.json"

        logger.info(f"== 啟動 Benchmark 指標評估 (Evaluation Mode) ==")

        with open(q_path, "r", encoding="utf-8") as f:
            questions = json.load(f).get("questions", [])

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_raw = json.load(f).get("ground_truths", [])
            gt_map = {item["qid"]: item["retrieve"] for item in gt_raw}

        pipeline = StandaloneRAGPipeline()

        correct_by_cat = {"insurance": 0, "finance": 0, "faq": 0}
        total_by_cat = {"insurance": 0, "finance": 0, "faq": 0}

        for item in questions:
            qid = item["qid"]
            cat = item["category"]
            query = item["query"]
            source = item.get("source", [])
            gt_pid = gt_map.get(qid)

            pred_pid = pipeline.retrieve_single(query, cat, source)

            total_by_cat[cat] += 1
            if pred_pid == gt_pid:
                correct_by_cat[cat] += 1

        total_all = len(questions)
        correct_all = sum(correct_by_cat.values())
        p1_overall = (correct_all / total_all) * 100.0 if total_all > 0 else 0.0

        print("\n" + "=" * 60)
        print("  Standalone RAG Pipeline Result (Tencent Conan + Processed)")
        print("=" * 60)
        for cat in ["faq", "insurance", "finance"]:
            tot = total_by_cat[cat]
            cor = correct_by_cat[cat]
            acc = (cor / tot * 100.0) if tot > 0 else 0.0
            print(f"  {cat.capitalize():12s}: {cor:2d} / {tot:2d} ({acc:6.2f}%)")
        print("-" * 60)
        print(f"  Overall P@1 : {correct_all:2d} / {total_all:2d} ({p1_overall:6.2f}%)")
        print("=" * 60)

if __name__ == "__main__":
    main()
