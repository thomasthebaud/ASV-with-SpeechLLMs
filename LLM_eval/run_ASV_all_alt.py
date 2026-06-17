from model import ASV_LLM
import random

SEED = 42
TOTAL = 5000
confidence_score = True
use_logits = False

model_name = "moonshotai/Kimi-Audio-7B-Instruct"
# model_name = "nvidia/audio-flamingo-3-hf"

print(f"Loading Model {model_name}")
model = ASV_LLM(model_name=model_name)

def get_df_no_pandas(csv):
    with open(csv, 'r') as f:
        lines =f.readlines()

    samples = random.sample(range(0, len(model_ids)), TOTAL)
    model_ids = [lines[i].split(',')[0] for i in samples]
    segment_ids = [lines[i].split(',')[1] for i in samples]
    model_files = [f'/export/corpora5/VoxCeleb1_v2/wav/{i[:7]}/{i[8:-6]}/{i[-5:]}.wav' for i in model_ids]
    segment_files = [f'/export/corpora5/VoxCeleb1_v2/wav/{i[:7]}/{i[8:-6]}/{i[-5:]}.wav' for i in segment_ids]
    return model_files, segment_files, model_ids, segment_ids


for trial in ['o', 'e', 'h']:
    print(f"starting trial {trial}")
    logfile = "/home/tthebau1/SPEAR/followup7B/outputs/"+model_name.split('/')[1] + f"_vox1-{trial}.log"

    print(f"Loading dataset voxceleb1-test-{trial}")
    df = df.sample(n=TOTAL, random_state=SEED).reset_index()
    model_files, segment_files, model_ids, segment_ids = get_df_no_pandas(f"/home/tthebau1/SPEAR/followup7B/data/trials_{trial}.csv")
    

    #Initialize logs    
    with open(logfile, 'w') as f:
        f.write("-- Starting evaluating {model_name} on voxceleb1-test-{trial} --")
        if confidence_score: f.write(" Using confidence score --")
        if use_logits: f.write(" Saving logits --")
        f.write("\n")

    for idx, (model_file, segment_file, model_id, segment_id) in zip(model_files, segment_files, model_ids, segment_ids):
        same = model_id.split('-')[0]==segment_id.split("-")[0]
        
        if not use_logits: raw_output = model.process(model_file, segment_file, return_logits=use_logits, confidence_question=confidence_score)
        else: raw_output, logits = model.process(model_file, segment_file, return_logits=use_logits, confidence_question=confidence_score)
        raw = raw_output.replace('\n', ' ').replace('\t', ' ')

        with open(logfile, 'a') as f:
            f.write(f"[outputs] Conversation {idx}/{TOTAL}\traw {raw}\t(same={same}, model_id={model_id}, segment_id={segment_id})\n")
