import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
import os


# print(scores)
# print(label)
# Yes ids: [7414]
# No ids: [2308]

def load_scores(dir='logits/qwen2.5/vox1-o/'):
    emb_path = os.listdir(dir)
    emb_path = emb_path
    embs = [np.load(dir+e) for e in tqdm(emb_path, desc="loading logits")]
    # print(emb_path)
    spk_tuples = [[s.split('-')[0] for s in e.split('_') if 'id' in s] for e in emb_path]
    label = np.array([int(a==b) for a,b in spk_tuples])
    # print(embs[0].shape)
    scores = np.array([np.max(e[:, 7414])/np.max(e[:, 2308]) for e in embs])
    return scores, label

def eer_from_scores(scores: np.ndarray, labels: np.ndarray):
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    i = np.nanargmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[i] + fnr[i])

    # Optional: try to interpolate around the crossing if it exists
    # (only if we have a neighbor on each side and the sign changes)
    if 0 < i < len(fpr) - 1:
        a, b = (fpr[i] - fnr[i]), (fpr[i+1] - fnr[i+1])
        if a == 0:
            eer = fpr[i]
        elif a * b < 0:  # crossing between i and i+1
            w = a / (a - b)  # in [0,1]
            eer = fpr[i] + w * (fpr[i+1] - fpr[i])
            thr_i = thr[i] + w * (thr[i+1] - thr[i])
            return eer, thr_i

    return eer, thr[i]


for trial in ['e', 'h', 'o']:
    scores, label = load_scores(dir=f'logits/qwen2.5/vox1-{trial}/')
    print(scores.shape, label.shape)

    eer, eer_thr = eer_from_scores(scores, label)
    eer = min(eer, 1-eer)
    print(f"EER Vox1-{trial}:", 100*eer, '%')