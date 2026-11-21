#!/usr/bin/env python3
"""
Precision@1 scorer for any prediction file in the competition submission format.

  pred file : {"answers": [{"qid": 1, "retrieve": 392}, ...]}
  gt file   : {"ground_truths": [{"qid": 1, "retrieve": 392, "category": "insurance"}, ...]}

A question counts as correct when the predicted pid equals the ground-truth pid (compared as int).
Per-category scores use the `category` field of the ground truth, or of --question_path if given.

Usage:
  python scripts/evaluate.py --pred_path pred_retrieve.json \
      --gt_path data/dataset/preliminary/ground_truths_example.json
"""
import argparse
import json


def load_list(path, keys):
    data = json.load(open(path, encoding='utf-8'))
    if isinstance(data, dict):
        for k in keys:
            if k in data:
                return data[k]
        raise SystemExit(f'{path}: expected one of the keys {keys}')
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_path', required=True)
    ap.add_argument('--gt_path', required=True)
    ap.add_argument('--question_path', help='optional, for categories if the gt file has none')
    a = ap.parse_args()

    pred = {int(x['qid']): int(x['retrieve']) for x in load_list(a.pred_path, ('answers', 'predictions'))}
    gts = load_list(a.gt_path, ('ground_truths',))
    cat = {int(g['qid']): g.get('category', 'all') for g in gts}
    if a.question_path:
        cat.update({int(q['qid']): q['category'] for q in load_list(a.question_path, ('questions',))})

    tot = {}
    for g in gts:
        qid = int(g['qid'])
        t = tot.setdefault(cat[qid], [0, 0])
        t[1] += 1
        t[0] += int(pred.get(qid) == int(g['retrieve']))
    missing = sum(1 for g in gts if int(g['qid']) not in pred)

    for c in sorted(tot):
        print(f'{c:10s}: {tot[c][0]}/{tot[c][1]} = {tot[c][0] / tot[c][1] * 100:.2f}%')
    ok = sum(v[0] for v in tot.values())
    n = sum(v[1] for v in tot.values())
    print(f'overall   : {ok}/{n} = {ok / n * 100:.2f}%')
    if missing:
        print(f'(warning: {missing} ground-truth questions have no prediction)')


if __name__ == '__main__':
    main()
