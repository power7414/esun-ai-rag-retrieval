import re
import logging
from typing import List, Set
import jieba

logger = logging.getLogger(__name__)

# 預設金融與法規專有名詞保護清單 (Financial Domain Lexicon)
DEFAULT_FINANCIAL_LEXICON = [
    "資本適足率", "淨值比率", "風險性資產", "第一類資本", "第二類資本",
    "流動性覆蓋比率", "槓桿比率", "金融消費者保護法", "洗錢防制法",
    "投資型保險", "人壽保險", "健康保險", "傷害保險", "年金保險",
    "解約金", "保險單借款", "寬限期間", "除外責任", "告知義務"
]

class LexiconCleaner:
    """
    金融專業字典與日期正規化處理器 (Lexicon Protection & Normalizer)
    保護「資本適足率」等詞彙不被 Jieba 拆散，並統一西元/民國與日期格式。
    """
    def __init__(self, custom_words: List[str] = None):
        self.lexicon = set(DEFAULT_FINANCIAL_LEXICON)
        if custom_words:
            self.lexicon.update(custom_words)

        for word in self.lexicon:
            jieba.add_word(word)
        logger.info(f"成功加載 {len(self.lexicon)} 個金融專業保護詞彙至 Jieba 字典")

    def normalize_dates(self, text: str) -> str:
        """
        將「112年」、「民國112年」統一轉換正規化
        """
        # 民國年轉西元標示簡化
        text = re.sub(r'民國\s*([0-9]{2,3})\s*年', r'\1年', text)
        return text

    def tokenize_protected(self, text: str) -> List[str]:
        """
        執行受字典保護的中詞分詞
        """
        normalized = self.normalize_dates(text)
        return [w.strip() for w in jieba.cut(normalized) if w.strip()]
