import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
import os


# print(scores)
# print(label)
# Yes ids: [7414]
# No ids: [2308]
def extract_numbers(s: str) -> str:
    s = s.replace('*', ' ').replace(',',' ').replace('"',' ').replace('.',' ')
    if ' No ' in s and ' Yes ' in s:
        if 'from different speakers' in s or 'not the same' in s: reverse=True
        else:
            print('YESANDNO:',s)
            exit()
    elif ' No ' in s: reverse=True
    elif ' Yes ' in s: reverse=False
    else:
        if 'from different speakers' in s or 'not the same' in s: reverse=True
        else:
            print('no:',s)
            exit()

    if 'confidence' in s:
        s = s.split('confidence')[1]
        num = ''.join(char for char in s if char.isdigit())
    else: num='90'
    if len(num)==0: num=-1
    elif reverse: num = 100-int(num)
    else:num = int(num)
    return num

def load_scores(file='slurm_output/qwen2.5_vox1-E.log'):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [l for l in lines if '[outputs] Conversation' in l]
    t = len(lines)
    # print(len(lines), lines[0])
    lines = [l for l in lines if 'Yes' in l or 'No' in l]
    # print(f"lines with confidence: {len(lines)}/{t}")

    scores = np.array([extract_numbers(l.split('(same=')[0]) for l in lines])
    label = np.array([l.split('(same=')[1].split(', ')[0]=='True' for l in lines])
    # print(label[:10], scores[:10])
    scores, label = scores[scores>=0], label[scores>=0]
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


for trial in ['o', 'e', 'h']:
    scores, label = load_scores(f'outputs/audio-flamingo-3-hf-vox1-{trial}.log')
    set_scores = [int(s) for s in set(scores)]
    print(scores.shape, label.shape, set_scores, len(set_scores), 
             100*np.sum([int(s%10!=0) for s in scores])/len(scores),
             100*np.sum([int(s%5!=0) for s in scores])/len(scores), sep='\t')

    eer, eer_thr = eer_from_scores(scores, label)
    eer = min(eer, 1-eer)
    print(f"EER Vox1-{trial}:", 100*eer, '%')