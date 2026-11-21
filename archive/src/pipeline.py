import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
from .preprocessor import PDFParser, ArticleChunker, LexiconCleaner
from .retriever import FAQ_Q2Q_Retriever, LexiconBM25Retriever, HybridRetriever

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    生產級端到端 RAG 法律與金融問答檢索 Pipeline (Production-Ready RAG Pipeline)
    整合最優預處理、細粒度條款切片、金融字典與 FAQ Q2Q 向量檢索。
    """
    def __init__(self, reference_path: str = "data/reference"):
        self.reference_path = Path(reference_path)
        self.parser = PDFParser()
        self.chunker = ArticleChunker()
        self.cleaner = LexiconCleaner()

        # 初始化檢索器
        self.faq_retriever = FAQ_Q2Q_Retriever()
        self.doc_retriever = LexiconBM25Retriever()

        self._initialize_indices()
        self.hybrid = HybridRetriever(self.faq_retriever, self.doc_retriever)

    def _initialize_indices(self):
        """加載參考庫並建立最佳化檢索索引"""
        faq_path = self.reference_path / "faq" / "pid_map_content.json"
        if faq_path.exists():
            with open(faq_path, "r", encoding="utf-8") as f:
                faq_data = json.load(f)
                self.faq_retriever.build_faq_index(faq_data)

    def predict_questions(self, questions_input: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        對競賽問題列表進行批次預測
        """
        if isinstance(questions_input, dict):
            questions_data = questions_input.get("questions") or questions_input.get("ground_truths") or []
        else:
            questions_data = questions_input

        predictions = []
        for item in questions_data:
            if not isinstance(item, dict):
                continue
            qid = item.get("qid")
            query = item.get("query", "")
            category = item.get("category", "")
            source_list = item.get("source", [])

            retrieved = self.hybrid.retrieve(query, category)
            best_id = retrieved[0]["retrieved_id"] if retrieved else (source_list[0] if source_list else 0)

            predictions.append({
                "qid": qid,
                "retrieve": best_id
            })

        return predictions
