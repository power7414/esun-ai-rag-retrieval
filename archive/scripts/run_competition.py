#!/usr/bin/env python3
"""
競賽當天 (Competition Day) 專用執行入口
預設針對競賽正式資料集 (data/competition) 進行推論：
python scripts/run_competition.py \
    --question_path data/competition/questions.json \
    --source_path data/competition/reference \
    --output_path data/competition/pred_retrieve.json
"""
import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="RAG Competition Inference Script")
    parser.add_argument("--question_path", type=str, default="data/competition/questions.json", help="Path to input questions JSON")
    parser.add_argument("--source_path", type=str, default="data/competition/reference", help="Path to reference directory")
    parser.add_argument("--output_path", type=str, default="data/competition/pred_retrieve.json", help="Path to save predictions JSON")

    args = parser.parse_args()

    logger.info(f"== 啟動競賽推論 Pipeline ==")
    logger.info(f"問題檔案: {args.question_path}")
    logger.info(f"參考數據庫: {args.source_path}")

    # 載入問題 JSON
    with open(args.question_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # 執行最優化 Hybrid RAG Pipeline
    pipeline = RAGPipeline(reference_path=args.source_path)
    predictions = pipeline.predict_questions(questions)

    # 保存預測答案
    output_p = Path(args.output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    with open(output_p, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    logger.info(f"預測完成！共產出 {len(predictions)} 筆答案至: {args.output_path}")

if __name__ == "__main__":
    main()
