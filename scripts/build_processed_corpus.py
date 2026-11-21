#!/usr/bin/env python3
"""
Rebuild data/processed/processed_{finance,insurance}_data.json from the official reference PDFs.

These files are used by the *alternative* code paths only (they are NOT needed for the headline
`unified_pipeline.py --finance_method dict_hybrid` run):
  - unified_pipeline.py --finance_method chunk_hybrid   (processed_finance_data.json)
  - standalone_rag_pipeline.py                          (both files)
  - test_improved_finance.py / build_finance_embeddings.py / compare_all_algorithms.py /
    test_fast_finance_inference.py                      (processed_finance_data.json)

Ported from the team's 2024-11-06 preprocessing (`2024_1106_preprocess/preprocess.py` +
`text_processor.py`): pdfplumber page text joined with "\n", then `TextProcessor.preprocess_text`
(special-char normalisation, number normalisation, jieba segmentation with a domain lexicon,
stop-word and single-character filtering). `TextProcessor` below is copied verbatim.
The original fell back to PaddleOCR for PDFs without a text layer; in the archived output those
PDFs have no entry at all (948/1035 finance, 643/643 insurance), so the OCR branch is omitted and
such PDFs are skipped, which matches the archived files.

Output format: [{"id": "<pid>", "text": "<space-separated tokens>", "filename": "<pid>.pdf"}, ...]

Usage:
  python scripts/build_processed_corpus.py --source_path data/reference [--output_dir data/processed]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pdfplumber

# ---------------------------------------------------------------------------
# TextProcessor — verbatim copy of 2024_1106_preprocess/text_processor.py
# ---------------------------------------------------------------------------
import jieba
import re
from typing import List, Set


class TextProcessor:
    def __init__(self):
        self.domain_terms = self.create_domain_terms()
        self._initialize_jieba()
        self._initialize_custom_financial_terms()

    def create_domain_terms(self):
        return {
            "金融領域": {
                "財務報表": {
                    "合併財務報表": ["合併財報", "財報"],
                    "合併資產負債表": ["資產負債表", "資負表"],
                    "合併現金流量表": ["現金流量表", "現流表", "合併現流表"],
                    "合併損益表": ["損益表", "收支表", "合併綜合損益表"],
                    "合併權益變動表": ["權益變動表"]
                },
                "財務指標": {
                    "每股盈餘": ["EPS", "每股收益"],
                    "股東權益報酬率": ["ROE", "股東報酬率"],
                    "資產報酬率": ["ROA", "總資產報酬率"],
                    "營業利益率": ["營益率"],
                    "流動比率": ["流動率"],
                    "速動比率": ["速動率"]
                },
                "支付相關": {
                    "電子支付": ["電子錢包", "行動支付", "手機支付"],
                    "行動銀行": ["手機銀行", "APP銀行"],
                    "網路銀行": ["網銀", "線上銀行"],
                    "跨境支付": ["跨境匯款", "國際支付"],
                    "掃碼支付": ["QR Code支付", "條碼支付"],
                    "非約定轉帳": ["非約轉"]
                },
                "帳戶服務": {
                    "數位存款": ["網路存款", "線上開戶"],
                    "定期存款": ["定存"],
                    "活期存款": ["活存"],
                    "支票存款": ["支存"],
                    "信託帳戶": ["信託專戶"],
                    "保管箱服務": ["保管箱"]
                }
            },
            "保險領域": {
                "保險商品": {
                    "人壽保險": ["壽險", "人壽險"],
                    "健康保險": ["健保", "醫療保險"],
                    "意外保險": ["意外險"],
                    "投資型保險": ["投資保險"],
                    "定期保險": ["定期壽險"],
                    "終身保險": ["終身壽險"]
                },
                "保險條款": {
                    "保險給付": ["理賠", "給付金"],
                    "保險費率": ["保費率", "費率"],
                    "保險金額": ["保額", "理賠金額"],
                    "保險期間": ["保障期間"],
                    "等待期間": ["等待期"],
                    "除外責任": ["除外事項"]
                }
            }
        }

    def _initialize_jieba(self):
        """初始化結巴分詞器"""
        # 添加所有領域詞彙
        for domain in self.domain_terms.values():
            for category in domain.values():
                for term, variants in category.items():
                    jieba.add_word(term)
                    for variant in variants:
                        jieba.add_word(variant)

    def _initialize_custom_financial_terms(self):
        """初始化金融相關自定義詞彙"""
        financial_terms = [
            "支付寶", "LINE Pay", "街口支付", "台灣Pay",
            "行動銀行", "網路銀行", "數位帳戶",
            "信用卡", "金融卡", "簽帳金融卡",
            "定期存款", "活期存款", "支票存款",
            "基金投資", "外匯交易", "股票交易",
            "理財規劃", "信託服務", "保險商品",
            "網路投保", "電子支付", "行動支付",
            "跨境支付", "掃碼支付", "非約定轉帳",
            "晶片金融卡", "約定轉帳", "即時轉帳"
        ]
        
        for term in financial_terms:
            jieba.add_word(term)

    def standardize_special_chars(self, text: str) -> str:
        """標準化特殊字符"""
        char_mappings = {
            '：': ':', '？': '?', '。': '.', '、': ',',
            '；': ';', '（': '(', '）': ')', '「': '"',
            '」': '"', '『': "'", '』': "'", '【': '[',
            '】': ']', '，': ',', '！': '!', '～': '~'
        }
        
        for old, new in char_mappings.items():
            text = text.replace(old, new)
        return text

    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除 URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # 統一空白
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 統一貨幣單位
        text = re.sub(r'NT\$', '新台幣', text)
        text = re.sub(r'USD', '美元', text)
        
        return text

    def standardize_numbers(self, text: str) -> str:
        """標準化數字格式"""
        # 統一金額表示
        text = re.sub(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*[萬仟千]?元', 
                     lambda m: f"{float(m.group(1).replace(',',''))}元", text)
        
        # 統一百分比
        text = re.sub(r'(\d+(?:\.\d+)?)\s*[%％]', r'\1%', text)
        
        # 統一數字單位
        text = re.sub(r'(\d+(?:\.\d+)?)\s*萬', lambda x: str(float(x.group(1)) * 10000), text)
        text = re.sub(r'(\d+(?:\.\d+)?)\s*億', lambda x: str(float(x.group(1)) * 100000000), text)
        
        return text

    def _get_stopwords(self) -> Set[str]:
        """獲取停用詞表"""
        return {
            '的', '是', '在', '和', '了', '與', '這', '那', '他', '她', '它',
            '們', '我', '你', '妳', '您', '我們', '你們', '妳們', '他們', '她們',
            '它們', '及', '或', '而', '但', '則', '乃', '若', '即', '便', '以',
            '將', '與', '及', '和', '或', '嗎', '呢', '吧', '啊', '著', '說',
            '到', '就', '又', '也', '但', '很', '還', '只', '就', '都', '要',
            '這', '那', '有', '無', '好', '對', '能', '會', '可以', '請問',
            '如何', '什麼', '怎麼', '為何', '為什麼', '哪些', '請', '想', '需要'
        }

    def preprocess_text(self, text: str) -> str:
        """文本預處理主函數"""
        # 基礎清理
        text = text.replace('\n', ' ').strip()
        text = self.standardize_special_chars(text)
        text = self.clean_text(text)
        
        # 標準化數字
        text = self.standardize_numbers(text)
        
        # 專業術語標準化
        text = self.standardize_terms(text)
        
        # 分詞
        tokens = self.segment_text(text)
        
        # 過濾 tokens
        tokens = self.filter_tokens(tokens)
        
        return ' '.join(tokens)

    def standardize_terms(self, text: str) -> str:
        """標準化專業術語"""
        for domain in self.domain_terms.values():
            for category in domain.values():
                for standard_term, variants in category.items():
                    pattern = '|'.join(map(re.escape, variants + [standard_term]))
                    text = re.sub(rf'\b({pattern})\b', standard_term, text)
        return text

    def segment_text(self, text: str) -> List[str]:
        """文本分詞"""
        return list(jieba.cut(text))

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """過濾 tokens"""
        stopwords = self._get_stopwords()
        return [
            token for token in tokens 
            if token not in stopwords 
            and len(token.strip()) > 0
            and not (len(token) == 1 and not token.isdigit())
        ]


# ---------------------------------------------------------------------------
# Extraction / driver (from 2024_1106_preprocess/preprocess.py, OCR branch removed)
# ---------------------------------------------------------------------------
_processor = None


def _init_worker():
    global _processor
    _processor = TextProcessor()  # registers the domain lexicon in the global jieba dictionary


def _process_one(file_path):
    filename = os.path.basename(file_path)
    doc_id = os.path.splitext(filename)[0]
    try:
        with pdfplumber.open(file_path) as pdf:
            text_list = []
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    text_list.append(text)
        text = "\n".join(text_list) if text_list else None
    except Exception:
        text = None
    if not text:
        return None
    return {'id': doc_id, 'text': _processor.preprocess_text(text), 'filename': filename}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_path', required=True, help='reference directory containing finance/ and insurance/')
    ap.add_argument('--output_dir', default='data/processed')
    ap.add_argument('--categories', nargs='+', default=['finance', 'insurance'])
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    for category in a.categories:
        folder = os.path.join(a.source_path, category)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pdf')]
        with ProcessPoolExecutor(initializer=_init_worker) as ex:
            docs = [d for d in ex.map(_process_one, files, chunksize=4) if d]
        out = os.path.join(a.output_dir, f'processed_{category}_data.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        print(f'{category}: {len(files)} PDFs -> {len(docs)} documents -> {out}')


if __name__ == '__main__':
    main()
