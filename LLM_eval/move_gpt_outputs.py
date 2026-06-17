import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
import os


# print(scores)
# print(label)
# Yes ids: [7414]
# No ids: [2308]

def move_file(input, output):
    with open(input, 'r') as f:
        lines = f.readlines()
    lines = ['[outputs] Conversation' + l for l in lines]
    with open(output, 'w') as f:
        f.writelines(lines)


    
for trial in ['o', 'e', 'h']:
    move_file(input=f"../local/ASV_experiments/outputs_voxceleb1-{trial}.csv", output = f'outputs/gpt4-vox1-{trial}.log')


        