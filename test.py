import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm


# ================= Configuration =================
class CFG:
    model_name = 'microsoft/deberta-v3-large'
    output_dir = r'C:\Users\SIMON\Desktop\NLP\models'
    checkpoint_path = os.path.join(
        output_dir,
        'microsoft-deberta-v3-large_fold0_best.pth'
    )

    oof_pkl = os.path.join(
        output_dir,
        'modelsoof_df_0.pkl'
    )
    max_len = 512
    batch_size = 8
    dropout = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= Utility Functions =================
def spans_to_binary(spans, length=None):
    length = np.max(spans) if length is None and len(spans)>0 else length
    binary = np.zeros(length, dtype=int)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def micro_f1(preds, truths):
    from sklearn.metrics import f1_score
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def span_micro_f1(preds, truths):
    bin_p, bin_t = [], []
    for p,t in zip(preds, truths):
        if not len(p) and not len(t):
            continue
        length = max(np.max(p) if p else 0, np.max(t) if t else 0)
        bin_p.append(spans_to_binary(p, length))
        bin_t.append(spans_to_binary(t, length))
    return micro_f1(bin_p, bin_t)


def create_labels_for_scoring(df):
    truths = []
    for loc_list in df['location']:
        spans = []
        if isinstance(loc_list, list):
            for loc in loc_list:
                start, end = map(int, loc.split())
                spans.append([start, end])
        truths.append(spans)
    return truths


def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i,(text,pred) in enumerate(zip(texts,predictions)):
        enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        for (s,e),p in zip(enc['offset_mapping'], pred):
            results[i][s:e] = p
    return results


def get_results(char_probs, th=0.5):
    import itertools
    results = []
    for probs in char_probs:
        idxs = np.where(probs>=th)[0] + 1
        groups = [list(g) for _,g in itertools.groupby(idxs, key=lambda n,c=itertools.count(): n-next(c))]
        spans = [f"{min(g)-1} {max(g)}" for g in groups]
        results.append(';'.join(spans))
    return results


def get_predictions(results):
    preds = []
    for res in results:
        spans = []
        if res:
            for part in res.split(';'):
                s,e = map(int, part.split())
                spans.append([s,e])
        preds.append(spans)
    return preds


def tune_threshold(oof_df, tokenizer):
    # Tune threshold on OOF predictions
    pred_cols = sorted([c for c in oof_df.columns if isinstance(c,int)])
    preds_array = oof_df[pred_cols].values
    texts = oof_df['pn_history'].values
    char_probs = get_char_probs(texts, preds_array, tokenizer)
    true_spans = create_labels_for_scoring(oof_df)
    ths = np.linspace(0.1,0.9,81)
    f1s = []
    for th in ths:
        res = get_results(char_probs, th)
        pred_spans = get_predictions(res)
        f1s.append(span_micro_f1(pred_spans, true_spans))
    best_idx = int(np.argmax(f1s))
    best_th = ths[best_idx]
    print(f"Best threshold = {best_th:.2f}, F1 = {f1s[best_idx]:.4f}")
    return best_th

# ================= Dataset and Model =================
def collate_fn(batch):
    # batch: list of dicts
    return {
        k: torch.stack([item[k] for item in batch], dim=0)
        for k in batch[0]
    }

class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row['pn_history'], row['feature_text'],
            add_special_tokens=True,
            padding='max_length', truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k,v in inputs.items()}

class CustomModel(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.config.hidden_size, 2)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        seq_out = outputs.last_hidden_state
        logits = self.fc(self.dropout(seq_out))
        return logits

# ================= Inference =================
def inference(test_df, tokenizer, model, threshold):
    ds = TestDataset(test_df, tokenizer, CFG.max_len)
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            for k in batch:
                batch[k] = batch[k].to(CFG.device)
            logits = model(batch)
            probs = F.softmax(logits, dim=2)[:,:,1].cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)
    char_probs = get_char_probs(test_df['pn_history'].values, all_probs, tokenizer)
    test_df['location'] = get_results(char_probs, th=threshold)
    return test_df

# ================= Main =================
def main():
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = CustomModel(CFG.model_name, CFG.dropout)
    ckpt = torch.load(
    CFG.checkpoint_path,
    map_location='cpu',
    weights_only=False
    )
    model.load_state_dict(ckpt['model'])
    model.to(CFG.device)

    # Determine threshold
    if os.path.exists(CFG.oof_pkl):
        oof_df = pd.read_pickle(CFG.oof_pkl)
        threshold = 0.5
    else:
        threshold = 0.5
        print("No OOF .pkl found, using threshold=0.5")

    # Load and prepare test data
    test = pd.read_csv(os.path.join(CFG.output_dir, '../nbme-score-clinical-patient-notes/test.csv'))
    features = pd.read_csv(os.path.join(CFG.output_dir, '../nbme-score-clinical-patient-notes/features.csv'))
    notes = pd.read_csv(os.path.join(CFG.output_dir, '../nbme-score-clinical-patient-notes/patient_notes.csv'))
    test = test.merge(features, on=['feature_num','case_num'], how='left') \
               .merge(notes, on=['pn_num','case_num'], how='left')

    # Run inference
    result_df = inference(test, tokenizer, model, threshold)

    # Save submission
    submission = result_df[['id','location']]
    submission.to_csv('submission.csv', index=False)
    print("Saved submission.csv")

print(">>> CFG.output_dir =", CFG.output_dir)
print(">>> CFG.checkpoint_path =", CFG.checkpoint_path)
import os; print("exists?", os.path.exists(CFG.checkpoint_path))


if __name__ == '__main__':
    main()