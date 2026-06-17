import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
import os
import re


# print(scores)
# print(label)
# Yes ids: [7414]
# No ids: [2308]
df = pd.read_csv("/export/corpora5/VoxCeleb1_v2/vox1_meta.csv", sep='\t')
gender_dic = {row['VoxCeleb1 ID']:row['Gender'] for _, row in df.iterrows()}
accent_dic = {row['VoxCeleb1 ID']:row['Nationality'].lower() for _, row in df.iterrows()}
accs = np.array(list((df['Nationality'])))
accs = [(a,100*len(accs[accs==a])/len(df)) for a in set(accs)]
print(sorted(accs, reverse=True, key=lambda x:x[1])[:10])
print(len(accs))

g = np.array(list((df['Gender'])))
g = [(a,100*len(g[g==a])/len(df)) for a in set(g)]
print(sorted(g, reverse=True, key=lambda x:x[1]))

exit()

def extract_gender_mentions(text):
    """
    Extract all occurrences of the words 'male' and 'female'
    from the input string, preserving order of appearance.
    Matching is case-insensitive.
    """
    pattern = r'\b(male|female)\b'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return matches

def extract_accent_mentions(text):
    text = text.lower()
    if 'accent' in text:
        text = text.replace('possibly ', '').replace('likely ', '').replace('with a ', '').replace('and ', '').replace('however ', '').replace('from a ', '').replace('specific ', '').replace('immediately ', '').replace('conversational ', '').replace('different ', '').replace('variation', '').replace('region ', '').replace('distinct ', '').replace('dialect', '').replace('country ', '').replace('english', '').replace('regional ', '').replace('dialectal ', '').replace('**', '').replace('local ', '').replace('slight ', '').replace('notable ', '').replace('or ', '').replace('formal ', '').replace('immediatly ', '').replace('clearly ', '')
        splits = text.split('accent')[1:]
        splits = [s.split('tone')[0] if 'tone' in s else s[:50] for s in splits]
        splits = [s.strip(' :-.,') for s in splits if len(s.strip(' '))>4]
        return splits[:2]
    else: return []

def find_spk_id(line):
    if '(same=' in line:
        line = line.split('model_id=')[1]
        model, segment = line.strip(')\n').split(', segment_id=')
    else:
        model, segment = line.split('Conversation')[1].split('\t')[:2]

    return model.split('-')[0], segment.split('-')[0]

def extract_numbers(s: str) -> str:
    s = s.split('raw')[1].split('(same=')[0].lower()    
    num = ''.join(char for char in s if char.isdigit())
    if len(num)==0: num=-1
    else:num = int(num)
    return num

def load_scores(file='slurm_output/qwen2.5_vox1-E.log'):
    with open(file, 'r+') as f:
        lines = f.readlines()
    lines = [l for l in lines[1:] if '[outputs] Conversation' in l]
    outputs = len(lines)
    c_lines = [l for l in lines if 'confidence ' in l.lower()]
    Fails = 100*(outputs-len(c_lines))/outputs
    if Fails>90:
        print(f"Too many fails ({Fails}%), trying again just checking if there is a number")
        c_lines = [l for l in lines if extract_numbers(l)>=0]
        Fails = 100*(outputs-len(c_lines))/outputs
    gender_correct = 0
    total = 0
    for l in lines:
        genders = extract_gender_mentions(l.lower())
        spk1, spk2 = find_spk_id(l)
        g1, g2 = gender_dic[spk1], gender_dic[spk2]
        if len(genders)==2:
            total+=2
            if genders[0][0]==g1: gender_correct+=1
            if genders[1][0]==g2: gender_correct+=1
        elif len(genders)>1:
            total+=2
            if g1!=g2 and len(set(genders))==2: gender_correct+=2
            if g1==g2 and len(set(genders))==1 and genders[0][0]==g1: gender_correct+=2
                
    gender_correct = 100*gender_correct/total
    predicts_gender = 100*(total / (2*len(lines)))

    accent_found = 0
    accent_correct = 0
    accs = []
    for l in lines:
        total+=2
        spk1, spk2 = find_spk_id(l)
        a1, a2 = accent_dic[spk1], accent_dic[spk2]
        accents = extract_accent_mentions(l)
        if len(accents)==2:
            for pr, gt in zip(accents, [a1,a2]):
                # was an accent predicted?
                if 'american' in pr or 'united states' in pr or 'australian' in pr or 'norway' in pr or 'south asian' in pr or 'irish' in pr or 'hispanic' in pr or 'british' in pr or 'south african' in pr or 'midwestern' in pr or 'canadian' in pr or'russian' in pr or'scottish' in pr or 'turkish' in pr or 'australian' in pr or 'spanish' in pr or 'london' in pr or 'german' in pr or 'new york' in pr or 'indian' in pr or 'hindi' in pr or 'uk' in pr or 'french' in pr or 'mandarin' in pr or 'paris' in pr:
                    accent_found+=1
                    
                    if gt=='usa':
                        if 'american' in pr or 'midwestern' in pr or 'new york' in pr or 'united states' in pr: accent_correct+=1
                    elif gt=='uk':
                        if 'british' in pr or 'uk' in pr or 'scottish' in pr: accent_correct+=1
                    elif gt=='canada':
                        if 'canadian' in pr or 'american' in pr: accent_correct+=1
                    elif 'british' in pr or 'american' in pr: continue

                    elif gt=='india':
                        if 'india' in pr or 'hindi' in pr or 'south asian' in pr: accent_correct+=1
                    elif 'india' in pr or 'hindi' in pr: continue
                    
                    elif gt=='france':
                        if 'french' in pr or 'paris' in pr: accent_correct+=1
                    elif gt=='mexico' or gt=='spain':
                        if 'spanish' in pr or 'hispanic' in pr: accent_correct+=1
                    elif gt=='norway':
                        if 'norway' in pr: accent_correct+=1
                    elif gt=='ireland':
                        if 'irish' in pr: accent_correct+=1
                    elif gt=='australia':
                        if 'australian' in pr: accent_correct+=1
                    elif gt=='germany':
                        if 'german' in pr: accent_correct+=1
                    else:
                        continue
                        print(gt, pr)

                elif 'neutral' in pr or 'not specified' in pr or 'southern' in pr or 'not identifiable' in pr or 'non-native' in pr or 'unknown' in pr or len(pr)<3:
                    continue
                else:
                    accs.append(pr)
                    
                    
                # else: accs.append(pr_a)
        # elif len(accents)==1:
        #     print("Found one!", accents[0])

    accs = np.array(accs)
    accs = [(a,len(accs[accs==a])) for a in set(accs)]
    accs = [(a,b) for a,b in accs if b>10]
    # print("REMAININGS:", sorted(accs, reverse=True, key=lambda x:x[1]))

    accent_correct = 0 if accent_found==0 else 100*accent_correct/accent_found
    predicts_accent = 100*(accent_found / (2*len(lines)))

    return Fails, gender_correct, predicts_gender, accent_correct, predicts_accent



for model in ['audio-flamingo-3-coraa', 'qwen2.5', 'gemini','gemini2.5', 'audio-flamingo-3-hf', 'gpt4', 'Kimi-Audio-7B-Instruct']:
    fails, gender_acc, show_gender, accent_acc, show_accent = 0, 0, 0, 0, 0
    for trial in ['o', 'e', 'h']:
        f,ga,gf, aa, af = load_scores(f'outputs/{model}-vox1-{trial}.log')
        fails+=f
        gender_acc+=ga
        show_gender+=gf
        accent_acc+=aa
        show_accent+=af
    print(f"### model {model} ###\t\tnumber of fails = {fails/3:.2f}%, gender accuracy = {gender_acc/3:.2f}%, gender shown = {show_gender/3:.2f}%, accent accuracy = {accent_acc/3:.2f}%, accent shown = {show_accent/3:.2f}%")
    
        