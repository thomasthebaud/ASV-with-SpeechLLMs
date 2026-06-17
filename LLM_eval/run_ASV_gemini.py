import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time

# import librosa
import numpy as np
import pandas as pd
import logging

import argparse

from google import genai
from google.genai import types

TOTAL = 5000
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--trial")
args = parser.parse_args()
trial = args.trial

# model = "gemini-3-flash-preview"
model = 'gemini-2.5-flash-lite'
user_text = """
            These are two distinct audios.
            First think about the elements that characterize each speaker, such as their gender, accent, tone, prosody, speech rate, 
            Give the characteristics for each audio
            then from those characteristics, infer the likelihood both speakers are the same. 
            Answer by Yes or No, and give a confidence score between 0 and 100:  
            0 correspond to the certainty they are from different speakers, 
            100 corresponds to the certainty that they are from the same speaker, 
            and 50 means you are uncertain."""

# Step 1: load CSV with trials
df = pd.read_csv(f"data/trials_{trial}.csv")
df['modelaudio'] = [f'/export/corpora5/VoxCeleb1_v2/wav/{i[:7]}/{i[8:-6]}/{i[-5:]}.wav' for i in df['modelid']]
df['segmentaudio'] = [f'/export/corpora5/VoxCeleb1_v2/wav/{i[:7]}/{i[8:-6]}/{i[-5:]}.wav' for i in df['segmentid']]
df = df.sample(n=TOTAL, random_state=SEED).reset_index()
n_total = len(df)

client = genai.Client(api_key='yourkey')


n_eval_done = 0

out_dir = Path.cwd() / f"voxceleb1-test-{trial}"
os.makedirs(out_dir, exist_ok=True)

for idx, row in df.iterrows():
    model_file = row['modelaudio']
    segment_file = row['segmentaudio']
    model_id = row['modelid']
    segment_id = row['segmentid']
    target = row['targettype']=='target'

    n_eval_done += 1

    print(
        f"[progress] Conversation {idx}/{TOTAL} "
        f"[eval {n_eval_done}/{TOTAL}] "
        f"(idx={idx}, prompt_id={idx}, model_id={model_id}, segment_id={segment_id})",
        flush=True,
    )
    
    with open(model_file, "rb") as f:
        audio_bytes_1 = f.read()

    with open(segment_file, "rb") as f:
        audio_bytes_2 = f.read()

    success = False
    fail = 0
    while fail<10:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    user_text, 
                    types.Part.from_bytes(data=audio_bytes_1, mime_type="audio/wav"),
                    types.Part.from_bytes(data=audio_bytes_2, mime_type="audio/wav"),
                ],
            )
            success = True
            break
        except Exception as e:
            fail+=1
            print(
                f"[error] Conversation {idx}/{TOTAL} "
                f"after {fail} failures: {e} "
                f"(idx={idx}, prompt_id={idx}, model_id={model_id}, segment_id={segment_id})",
                flush=True,
            )
            time.sleep(5)

    if fail==10: exit("too many failures")

    try: 
        raw = response.text.replace('\n', ' ').replace('\t', ' ')
    except:
        raw = "  "

    Same = model_id.split('-')[0]==segment_id.split("-")[0]
    print(
        f"[outputs] Conversation {idx}/{TOTAL} "
        f"raw {raw}\t"
        f"(same={Same}, model_id={model_id}, segment_id={segment_id})\n",
        flush=True,
    )