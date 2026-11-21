#!/usr/bin/env python3
"""
Unified retrieval pipeline (one script, one run, all three categories).

  faq       : Conan-embedding-v1 question-to-question (Q2Q) matching, CLS pooling
              (from scripts/standalone_rag_pipeline.py)
  insurance : pdfplumber plain text + jieba.cut_for_search + TF-IDF cosine
              (from the team's 2024-11-09 vsm_retrieve.py)
  finance   : --finance_method dict_hybrid (default, measured 80% on finance):
                raw report text (data/processed/finance_content.json, built by
                scripts/build_finance_raw.py; all whitespace removed by clean_noise)
                BM25 over jieba + 79-term financial dictionary   x 0.4
                + max-pooled Conan similarity over 384-char windows (128 overlap)   x 0.6
              --finance_method chunk_hybrid (measured 70%): same hybrid over the segmented
                data/processed/processed_finance_data.json (scripts/build_processed_corpus.py)
              --finance_method tfidf (measured 68%): pdfplumber text + tables, TF-IDF
                (from the team's 2024-11-09 bm25_retrieve.py)

Measured on the official 150-question example set (2026-11-07):
  dict_hybrid: insurance 45/50, finance 40/50, faq 48/50, overall 133/150 = 88.67%

Usage:
  python scripts/unified_pipeline.py \
      --question_path data/dataset/preliminary/questions_example.json \
      --source_path   data/reference \
      --output_path   pred_retrieve.json \
      [--gt_path data/dataset/preliminary/ground_truths_example.json]  # prints P@1 per category
      [--model /path/to/local/Conan-embedding-v1]                     # default: download from HF

Caveat: the methods were chosen on the same 150 questions they are scored on (no held-out set),
so the numbers are optimistic.
"""
import argparse
import json
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jieba
import numpy as np
import pdfplumber
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "TencentBAC/Conan-embedding-v1"
STOPWORDS = set(['的', '了', '和', '是', '就', '都', '而', '及', '與', '著'])


# ---------- PDF 讀取（兩種版本，分別對應 1109 vsm / bm25 腳本） ----------
def read_pdf_plain(path):
    with pdfplumber.open(path) as pdf:
        return ''.join((p.extract_text() or '') for p in pdf.pages)


def preprocess_text(text):  # 1109/bm25_retrieve.py 原樣
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return ' '.join(w for w in jieba.cut(text) if w not in STOPWORDS)


def read_pdf_tables(path):  # 1109/bm25_retrieve.py 的 read_pdf（不含 OCR）
    out = ''
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                out += t + '\n'
            for table in page.extract_tables():
                out += "Table:\n"
                for row in table:
                    out += ' | '.join(str(c) for c in row if c is not None) + '\n'
                out += '\n'
    return preprocess_text(out)


def _load_one(args):
    path, mode = args
    try:
        return int(Path(path).stem), (read_pdf_plain(path) if mode == 'plain' else read_pdf_tables(path))
    except Exception:
        return int(Path(path).stem), ''


def load_corpus(folder, mode, cache_dir, needed=None):
    cache = Path(cache_dir) / f"{Path(folder).name}_{mode}.pkl"
    if cache.exists():
        return pickle.load(open(cache, 'rb'))
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pdf')]
    if needed is not None:
        files = [f for f in files if int(Path(f).stem) in needed]
    with ProcessPoolExecutor() as ex:
        corpus = dict(ex.map(_load_one, [(f, mode) for f in files], chunksize=4))
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(corpus, open(cache, 'wb'))
    return corpus


# ---------- 三種檢索 ----------
def retrieve_insurance(query, source, corpus):  # 1109 VSM_retrieve
    docs = [corpus.get(int(s), '') for s in source]
    vec = TfidfVectorizer()
    m = vec.fit_transform([' '.join(jieba.cut_for_search(d)) for d in docs])
    q = vec.transform([' '.join(jieba.cut_for_search(query))])
    return int(source[int(cosine_similarity(q, m)[0].argmax())])


def retrieve_finance(query, source, corpus):  # 1109 TFIDF_retrieve（回傳 source 中的 id）
    docs = [corpus.get(int(s), '') for s in source]
    vec = TfidfVectorizer()
    m = vec.fit_transform(docs)
    q = vec.transform([preprocess_text(query)])
    return int(source[int(cosine_similarity(q, m)[0].argmax())])


class ChunkHybridFinance:  # scripts/test_improved_finance.py 第 4 種方法，邏輯照搬
    def __init__(self, faq_like, processed_json):
        self.enc = faq_like.encode
        data = json.load(open(processed_json, encoding='utf-8'))
        self.docs = {int(x['id']): x['text'] for x in data if str(x.get('id')).isdigit()}

    @staticmethod
    def chunk(text, size=384, overlap=128):
        if not text:
            return ['']
        if len(text) <= size:
            return [text]
        out, st = [], 0
        while st < len(text):
            out.append(text[st:st + size]); st += size - overlap
        return out

    def retrieve(self, query, source):
        from rank_bm25 import BM25Okapi
        pids = [p for p in source if self.docs.get(p)]
        if not pids:
            return source[0]
        texts = [self.docs[p] for p in pids]
        bm = BM25Okapi([list(jieba.cut(t)) for t in texts]).get_scores(list(jieba.cut(query)))
        q = self.enc([query])[0]
        dense = np.array([float((self.enc(self.chunk(t)) @ q).max()) for t in texts])
        bm_n = bm / (bm.max() + 1e-8)
        de_n = (dense - dense.min()) / (dense.max() - dense.min() + 1e-8)
        return pids[int((0.4 * bm_n + 0.6 * de_n).argmax())]


# 2024_1108_RAG/financial_dictionary.py：FINANCIAL_DICT 的報表名稱＋全部細項＋STATEMENT_KEYWORDS（去重 79 詞）
FINANCE_DICT_WORDS = ['不動產、廠房及設備', '保留盈餘', '借款增加（減少）', '其他收入', '其他權益', '利潤', '匯率變動對現金及約當現金之影響', '取得不動產、廠房及設備', '合併權益變動表', '員工未賺得酬勞', '國外營運機構之兌換差額', '存貨', '存貨（增加）減少', '庫藏股註銷', '待註銷股本', '應付債券', '應付帳款', '應付稅款', '應收帳款', '投資性不動產', '投資活動', '投資活動之淨現金流入（流出）', '投資活動現金流量', '折舊費用', '採用權益法認列之關聯企業及合資之變動數', '損益表', '攤銷費用', '支出', '收入', '期初現金及約當現金餘額', '期末現金及約當現金餘額', '未分配盈餘', '本期淨利', '本期現金及約當現金增加（減少）數', '權益', '權益變動表', '歸屬於母公司業主之權益', '法定盈餘公積', '流動負債', '流動資產', '無形資產', '營業利潤', '營業利益', '營業成本', '營業收入', '營業毛利', '營業活動', '營業活動之淨現金流入（流出）', '營業活動現金流量', '營業費用', '特別盈餘公積', '現金及約當現金', '現金流量表', '發放現金股利', '發行股票', '盈餘分配', '短期借款', '研發費用', '稅前淨利', '管理費用', '籌資活動', '籌資活動之淨現金流入（流出）', '籌資活動現金流量', '綜合損益表', '股份基礎給付交易', '股數', '股本', '處分不動產、廠房及設備', '負債', '財務成本', '資本公積', '資產', '資產負債表', '透過其他綜合損益按公允價值衡量之未實現（損）益', '長期借款', '非控制權益', '非流動負債', '非流動資產', '預付款項']


class DictHybridFinance(ChunkHybridFinance):  # B3：字典 BM25（原文）+ 原文切片向量，權重固定 0.4/0.6
    def __init__(self, faq_like, raw_json):
        self.enc = faq_like.encode
        raw = json.load(open(raw_json, encoding='utf-8'))
        self.docs = {int(k): self.clean(v['content']) for k, v in raw.items() if str(k).isdigit()}
        self.tok = jieba.Tokenizer()  # 獨立斷詞器：字典不影響保險用的全域 jieba
        for w in FINANCE_DICT_WORDS:
            self.tok.add_word(w)

    @staticmethod
    def clean(t):  # best_retrievers/finance_preprocessor.py clean_noise（空白全去）
        t = re.sub(r'\s+', '', t); t = re.sub(r'~\s*\d+\s*~', '', t); t = re.sub(r'-\s*\d+\s*-', '', t)
        t = t.replace('$', '')
        return re.sub(r'金管會審字第\s*[\w\d]+', '', t)

    def retrieve(self, query, source):
        from rank_bm25 import BM25Okapi
        pids = [p for p in source if self.docs.get(p)]
        if not pids:
            return source[0]
        texts = [self.docs[p] for p in pids]
        bm = BM25Okapi([list(self.tok.cut(t)) for t in texts]).get_scores(list(self.tok.cut(query)))
        q = self.enc([query])[0]
        dense = np.array([float((self.enc(self.chunk(t)) @ q).max()) for t in texts])
        bm_n = bm / (bm.max() + 1e-8)
        de_n = (dense - dense.min()) / (dense.max() - dense.min() + 1e-8)
        return pids[int((0.4 * bm_n + 0.6 * de_n).argmax())]


class FAQRetriever:  # standalone_rag_pipeline.py 的 Q2Q
    def __init__(self, faq_json, model_name=MODEL_NAME):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        raw = json.load(open(faq_json, encoding='utf-8'))
        self.items = [(int(pid), it['question']) for pid, lst in raw.items() for it in lst if it.get('question')]
        self.emb = self.encode([q for _, q in self.items])

    def encode(self, texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            x = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                e = self.model(**x).last_hidden_state[:, 0, :]
            out.append(torch.nn.functional.normalize(e, p=2, dim=1).numpy())
        return np.vstack(out)

    def retrieve(self, query, source):
        idx = [i for i, (pid, _) in enumerate(self.items) if pid in set(source)]
        q = self.encode([query])[0]
        return self.items[idx[int((self.emb[idx] @ q).argmax())]][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--question_path', required=True)
    ap.add_argument('--source_path', required=True)
    ap.add_argument('--output_path', required=True)
    ap.add_argument('--gt_path')
    ap.add_argument('--cache_dir', default='data/processed/unified_cache')
    ap.add_argument('--finance_method', choices=['dict_hybrid', 'chunk_hybrid', 'tfidf'], default='dict_hybrid')
    ap.add_argument('--processed_finance', default='data/processed/processed_finance_data.json')
    ap.add_argument('--raw_finance', default='data/processed/finance_content.json',
                    help='raw finance text for dict_hybrid (built by scripts/build_finance_raw.py)')
    ap.add_argument('--model', default=MODEL_NAME,
                    help='Hugging Face id or local directory of Conan-embedding-v1')
    a = ap.parse_args()

    qs = json.load(open(a.question_path, encoding='utf-8'))['questions']
    need = {c: {s for q in qs if q['category'] == c for s in q['source']} for c in ('insurance', 'finance')}
    ins = load_corpus(os.path.join(a.source_path, 'insurance'), 'plain', a.cache_dir, need['insurance'])
    faq = FAQRetriever(os.path.join(a.source_path, 'faq', 'pid_map_content.json'), a.model)
    if a.finance_method == 'tfidf':
        fin = load_corpus(os.path.join(a.source_path, 'finance'), 'tables', a.cache_dir, need['finance'])
    elif a.finance_method == 'dict_hybrid':
        fin_h = DictHybridFinance(faq, a.raw_finance)
    else:
        fin_h = ChunkHybridFinance(faq, a.processed_finance)

    answers = []
    for q in qs:
        c = q['category']
        if c == 'faq':
            r = faq.retrieve(q['query'], q['source'])
        elif c == 'insurance':
            r = retrieve_insurance(q['query'], q['source'], ins)
        elif a.finance_method == 'tfidf':
            r = retrieve_finance(q['query'], q['source'], fin)
        else:
            r = fin_h.retrieve(q['query'], q['source'])
        answers.append({'qid': q['qid'], 'retrieve': r})
    json.dump({'answers': answers}, open(a.output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'wrote {len(answers)} answers -> {a.output_path}')

    if a.gt_path:
        gt = {g['qid']: g['retrieve'] for g in json.load(open(a.gt_path, encoding='utf-8'))['ground_truths']}
        cat = {q['qid']: q['category'] for q in qs}
        pred = {x['qid']: x['retrieve'] for x in answers}
        tot = {}
        for qid, g in gt.items():
            t = tot.setdefault(cat[qid], [0, 0]); t[1] += 1; t[0] += int(pred.get(qid) == g)
        for c in ('insurance', 'finance', 'faq'):
            print(f'{c:10s}: {tot[c][0]}/{tot[c][1]} = {tot[c][0] / tot[c][1] * 100:.2f}%')
        ok = sum(v[0] for v in tot.values()); n = sum(v[1] for v in tot.values())
        print(f'overall   : {ok}/{n} = {ok / n * 100:.2f}%')


if __name__ == '__main__':
    main()
