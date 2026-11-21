import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pdfplumber

logger = logging.getLogger(__name__)

class PDFParser:
    """
    PDF 結構化抽取器 (PDF Parser)
    結合 pdfplumber 直接提取半結構化內文與空間表格，避免直接 OCR 產生的噪聲。
    """
    def __init__(self):
        pass

    def extract_text_and_tables(self, pdf_path: str) -> str:
        """
        從 PDF 檔案中提取文字與表格內容並拼接為結構化字串
        """
        path = Path(pdf_path)
        if not path.exists():
            logger.warning(f"PDF 檔案不存在: {pdf_path}")
            return ""

        extracted_chunks = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    # 1. 優先提取表格
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            cleaned_row = [str(cell).strip() for cell in row if cell is not None]
                            if cleaned_row:
                                extracted_chunks.append(" | ".join(cleaned_row))

                    # 2. 提取純文字
                    text = page.extract_text()
                    if text:
                        extracted_chunks.append(text)
        except Exception as e:
            logger.error(f"解析 PDF {pdf_path} 時發生錯誤: {e}")

        return "\n".join(extracted_chunks)
