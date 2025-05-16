#!/usr/bin/env python
# -*- coding: utf-8 -*-
# r"""
# evaluate_oof.py - Compute span-level micro-F1 directly from OOF pickle
#
# Usage:
# cd C:\Users\SIMON\Desktop\NLP\src
# python .\evaluate_oof.py --oof_pkl ..\models\modelsoof_df_0.pkl --train_csv ..\nbme-score-clinical-patient-notes\train.csv --threshold 0.5

# r"""
import argparse
import pandas as pd
import ast, re, numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OOF span-micro-F1')
    parser.add_argument('--oof_pkl',    required=True,  help='Path to the OOF pickle file')
    parser.add_argument('--train_csv',  required=True,  help='Path to train.csv with ground-truth annotations')
    parser.add_argument('--max_len',    type=int, default=512, help='Max sequence length used at inference')
    parser.add_argument('--threshold',  type=float, default=None,
                        help='Threshold for converting probabilities to spans; auto-tune if unset')
    return parser.parse_args()

def create_true_spans(df_true):
    """
    Parse the ground-truth `location` column into a list of [start, end] spans.
    """
    out = []
    for locs in df_true['location']:
        spans = []
        if isinstance(locs, str):
            try:
                lst = ast.literal_eval(locs)
            except:
                lst = []
        elif isinstance(locs, list):
            lst = locs
        else:
            lst = []
        for s in lst:
            nums = re.findall(r"\d+", s)
            if len(nums) >= 2:
                spans.append([int(nums[0]), int(nums[1])])
        out.append(spans)
    return out

def get_char_probs(texts, probs_matrix, tokenizer):
    """
    Map token-level probabilities to character-level probabilities.
    """
    char_probs = [np.zeros(len(t)) for t in texts]
    for i, (txt, probs) in enumerate(zip(texts, probs_matrix)):
        enc = tokenizer(
            txt,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=512,
            truncation=True
        )
        for (s, e), p in zip(enc['offset_mapping'], probs):
            char_probs[i][s:e] = p
    return char_probs

def probs_to_spans(char_probs, th):
    """
    Convert character-level probabilities to spans using threshold `th`.
    """
    spans_out = []
    import itertools
    for probs in char_probs:
        idxs = np.where(probs >= th)[0]
        groups = [list(g) for _, g in itertools.groupby(
            idxs, key=lambda x, c=itertools.count(): x - next(c)
        )]
        spans = [[min(g), max(g) + 1] for g in groups]
        spans_out.append(spans)
    return spans_out

def tune_threshold(true_spans, char_probs):
    """
    Search threshold from 0.1 to 0.9 to maximize span-micro-F1.
    """
    ths, best_th, best_f1 = np.linspace(0.1, 0.9, 81), 0, 0
    for th in ths:
        pred_spans = probs_to_spans(char_probs, th)
        bin_t, bin_p = [], []
        for tr, pr in zip(true_spans, pred_spans):
            if not tr and not pr:
                continue
            L = max(
                max((e for _, e in tr), default=0),
                max((e for _, e in pr), default=0)
            )
            bt = np.zeros(L);
            bp = np.zeros(L)
            for s, e in tr:
                bt[s:e] = 1
            for s, e in pr:
                bp[s:e] = 1
            bin_t.append(bt)
            bin_p.append(bp)
        y_true = np.concatenate(bin_t)
        y_pred = np.concatenate(bin_p)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    print(f"[TUNE] best threshold={best_th:.2f}, F1={best_f1:.4f}")
    return best_th

def main():
    args = parse_args()

    # 1) Load OOF pickle
    oof = pd.read_pickle(args.oof_pkl)
    print("OOF columns:", oof.columns.tolist())

    # 2) Load ground-truth train.csv
    train = pd.read_csv(args.train_csv)
    train['location'] = train['location'].apply(
        lambda x: x if isinstance(x, list) else ast.literal_eval(x)
    )

    # 3) Merge OOF with ground-truth spans
    oof = oof.merge(
        train[['pn_num', 'case_num', 'feature_num', 'location']],
        on=['pn_num', 'case_num', 'feature_num'], how='left'
    )
    # If merge yields location_x and location_y, take the ground-truth
    if 'location_y' in oof.columns:
        oof['location'] = oof['location_y']

    # 4) Extract text, true spans, and prediction probabilities
    texts = oof['pn_history'].astype(str).tolist()
    true_spans = create_true_spans(oof)
    # Auto-detect integer column names as probability columns
    pred_cols = [c for c in oof.columns if isinstance(c, int)]
    pred_cols += [int(c) for c in oof.columns if isinstance(c, str) and c.isdigit()]
    pred_cols = sorted(set(pred_cols))
    probs_matrix = oof[pred_cols].values

    # 5) Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

    # 6) Convert to character-level probabilities
    char_probs = get_char_probs(texts, probs_matrix, tokenizer)

    # 7) Determine threshold
    if args.threshold is not None:
        th = args.threshold
        print(f"Using threshold={th}")
    else:
        th = tune_threshold(true_spans, char_probs)

    # 8) Convert probabilities to spans
    pred_spans = probs_to_spans(char_probs, th)

    # 9) Compute final span-micro-F1
    bin_t, bin_p = [], []
    for tr, pr in zip(true_spans, pred_spans):
        if not tr and not pr:
            continue
        L = max(
            max((e for _, e in tr), default=0),
            max((e for _, e in pr), default=0)
        )
        bt = np.zeros(L, dtype=int)
        bp = np.zeros(L, dtype=int)
        for s, e in tr:
            bt[s:e] = 1
        for s, e in pr:
            bp[s:e] = 1
        bin_t.append(bt)
        bin_p.append(bp)
    if bin_t:
        y_true = np.concatenate(bin_t)
        y_pred = np.concatenate(bin_p)
        final_f1 = f1_score(y_true, y_pred)
    else:
        final_f1 = 0.0
    print(f"[RESULT] span-micro-F1 = {final_f1:.4f}")

if __name__ == '__main__':
    main()
