import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ArticleChunker:
    """
    細粒度條款切片器 (Article-level Chunking)
    專門處理保險/金融長篇條款，按「第一條、第二條...」細分切片，避免語意特徵被背景贅字稀釋。
    """
    def __init__(self):
        # 匹配中文條款標題模式，例如 "第一條", "第 1 條", "Article 1"
        self.article_pattern = re.compile(r'(第\s*[0-9一二三四五六七八九十百]+\s*條)')

    def chunk_document(self, doc_id: str, text: str) -> List[Dict[str, Any]]:
        """
        將文檔按條款切分為 Chunk 列表
        """
        if not text:
            return []

        splits = self.article_pattern.split(text)
        chunks = []

        if len(splits) <= 1:
            # 無法依條款切分時，採固定長度滑動窗口備用
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_0",
                "article_title": "全文",
                "content": text.strip()
            })
            return chunks

        # 組合條款標題與內文
        current_title = "前言"
        for i in range(1, len(splits), 2):
            title = splits[i].strip()
            content = splits[i+1].strip() if i+1 < len(splits) else ""
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{title}",
                "article_title": title,
                "content": f"{title} {content}"
            })

        logger.debug(f"文檔 {doc_id} 成功切分為 {len(chunks)} 個條款 Chunk")
        return chunks
