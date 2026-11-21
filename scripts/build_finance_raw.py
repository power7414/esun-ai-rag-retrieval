#!/usr/bin/env python3
"""
Build the raw-text finance corpus used by `unified_pipeline.py --finance_method dict_hybrid`.

Input : <reference>/finance/*.pdf (official competition reference documents)
Output: data/processed/finance_content.json
        {"<pid>": {"id": "<pid>", "category": "finance", "content": "<raw page text>"}, ...}
        plus data/processed/finance_failed_files.json (PDFs with no extractable text layer,
        i.e. scanned images; these get no entry, exactly like the original file).

This reproduces the format of the team's original
`2024_1106_preprocess/reference_json/finance_content.json`. The script that originally produced it
is not in the project archive; the extraction method was recovered by comparison: the team used
PyPDF2 elsewhere (2024_1102 / 2024_1105 code), and `"".join(page.extract_text())` followed by
`.strip()` reproduces the original `content` field. We use `pypdf` (the maintained successor of
PyPDF2). Retrieval only ever sees the text after `clean_noise` (all whitespace removed, page numbers
`~N~` / `-N-`, `$` and FSC filing numbers stripped — `DictHybridFinance.clean` in
unified_pipeline.py, copied from best_retrievers/finance_preprocessor.py), so `--verify` compares on
that cleaned form.

Usage:
  python scripts/build_finance_raw.py --source_path data/reference
  python scripts/build_finance_raw.py --source_path data/reference --verify <original finance_content.json>
"""
import argparse
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pypdf import PdfReader

logging.getLogger('pypdf').setLevel(logging.ERROR)  # malformed-PDF warnings are noise here


def clean_noise(t):  # identical to DictHybridFinance.clean (best_retrievers/finance_preprocessor.py)
    t = re.sub(r'\s+', '', t)
    t = re.sub(r'~\s*\d+\s*~', '', t)
    t = re.sub(r'-\s*\d+\s*-', '', t)
    t = t.replace('$', '')
    return re.sub(r'金管會審字第\s*[\w\d]+', '', t)


def extract(path):
    try:
        text = ''.join((p.extract_text() or '') for p in PdfReader(path).pages).strip()
    except Exception as e:  # unreadable PDF -> treated as failed
        logging.warning('failed to read %s: %s', path, e)
        text = ''
    return Path(path).stem, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_path', required=True, help='reference directory containing finance/*.pdf')
    ap.add_argument('--output_path', default='data/processed/finance_content.json')
    ap.add_argument('--verify', help='optional: original finance_content.json to compare against')
    a = ap.parse_args()

    folder = os.path.join(a.source_path, 'finance')
    files = sorted((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pdf')),
                   key=lambda f: int(Path(f).stem))
    out, failed = {}, []
    with ProcessPoolExecutor() as ex:
        for (pid, text), f in zip(ex.map(extract, files, chunksize=4), files):
            if text:
                out[pid] = {'id': pid, 'category': 'finance', 'content': text}
            else:
                failed.append(os.path.relpath(f, a.source_path))

    Path(a.output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    fail_path = Path(a.output_path).with_name('finance_failed_files.json')
    json.dump(failed, open(fail_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'{len(files)} PDFs -> {len(out)} with text, {len(failed)} without a text layer')
    print(f'wrote {a.output_path} and {fail_path}')

    if a.verify:
        ref = json.load(open(a.verify, encoding='utf-8'))
        same_keys = set(ref) == set(out)
        diff = [k for k in ref if k in out and clean_noise(ref[k]['content']) != clean_noise(out[k]['content'])]
        print(f'verify: same document ids = {same_keys} '
              f'(original {len(ref)}, rebuilt {len(out)}); cleaned-text mismatches = {len(diff)}')
        if diff:
            print('  mismatched ids:', ', '.join(sorted(diff, key=int)[:30]))


if __name__ == '__main__':
    main()
