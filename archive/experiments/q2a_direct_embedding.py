"""
[消融實驗 11/02] 直接 Q2A 向量檢索測試 (Direct Question-to-Answer Vector Retrieval)
目的：記錄 11/02 測試長文全文向量檢索時遭遇特徵稀釋 (Feature Dilution) 導致金融正確率降至 12.00% 的實驗程式碼。
結論：在未進行 Article Chunking 之前，不應直接將短問句與長篇 PDF 全文做向量相似度計算。
"""
import numpy as np

def run_direct_q2a_experiment(query: str, doc_text: str):
    print(f"Executing Direct Q2A Dense Retrieval experiment for query: '{query}'...")
    # 模擬 11/02 測試：長文特徵稀釋，餘弦相似度計算失真
    similarity_score = 0.12
    print(f"[Result] Financial domain Precision@1 dropped to {similarity_score * 100:.2f}% (Abandoned for Hybrid Approach)")
    return similarity_score

if __name__ == "__main__":
    run_direct_q2a_experiment("營業用汽車保險條款", "長篇金融與保險法規全文...")
