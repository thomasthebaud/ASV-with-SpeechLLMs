#!/usr/bin/env python3
"""\
Pipeline script (UPDATED: outside-judge only; Evidence-First + scalar score; Balanced Position; BOTH speakers per turnover):

1) Call filter_roles.py to create a filtered subset CSV from interactions_role_ABmapped.csv,
   restricted to a set of *role labels of interest*.

2) For each conversation (row) in the filtered subset:
   - Call turnover_extractor.py to extract A→B / B→A adjacent-utterance turnover clips.
   - For each speaker (A and/or B) whose ROLE is in roles_of_interest:
       * Evaluate that speaker regardless of whether they are Speaker 1 or Speaker 2 in the clip:
           - AtoB clip: Speaker 1 = A, Speaker 2 = B
           - BtoA clip: Speaker 1 = B, Speaker 2 = A
       * Look up the selected OUTSIDE-JUDGE question config by category_a/category_b.

       * For each target-speaker evaluation in a turnover clip, run the judge model twice (Balanced Position):
           - Variant 0: POSITIVE definition block first, then NEGATIVE
           - Variant 1: NEGATIVE definition block first, then POSITIVE
         Each run must return JSON:
           {"evidence": [...], "score": -2|-1|0|1|2}

         Turnover score is averaged: score_avg = (score_v0 + score_v1) / 2.
         flip_rate counts sign disagreements between v0 and v1 (excluding zeros).

   - Aggregate across all valid turnovers for that speaker in the conversation:
       * evidence_*: list of per-turnover evidence + scores
       * avg_score_*: mean(score_avg)
       * flip_rate_*: fraction of turnovers with sign(score_v0) != sign(score_v1) and both nonzero
       * fail_note_*: list of raw model outputs for turnovers/variants that failed JSON parsing

3) Write results back to the same --filtered-csv file *incrementally* (after each conversation).

Notes:
- This script expects the question-config JSON to contain exactly ONE question in "outside_judge".
  Use select_one_question.py to extract one question from a multi-question config.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time

# import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

import logging

class _SuppressQwenSystemPromptWarning(logging.Filter):
    _needle = "System prompt modified, audio output may not work as expected"
    def filter(self, record):
        return self._needle not in record.getMessage()

logging.getLogger().addFilter(_SuppressQwenSystemPromptWarning())

# Fixed script paths (set these at the top as needed)
FILTER_SCRIPT = Path("filter_roles.py")
TURNOVER_SCRIPT = Path("turnover_extractor.py")


def _str2bool(v: Any) -> bool:
    """Argparse helper: parse common true/false strings into bool."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


# ------------ Judge helpers ------------

def _decode_ids(processor: Qwen2_5OmniProcessor, token_ids: torch.Tensor) -> str:
    """
    Robustly decode generated token ids across processor/tokenizer variants.
    """
    # Prefer processor.batch_decode if available
    if hasattr(processor, "batch_decode"):
        try:
            return processor.batch_decode([token_ids], skip_special_tokens=True)[0]
        except Exception:
            pass

    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "decode"):
        try:
            return tok.decode(token_ids.tolist(), skip_special_tokens=True)
        except Exception:
            try:
                return tok.decode(token_ids, skip_special_tokens=True)
            except Exception:
                pass

    # Fallback
    return str(token_ids)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of a single JSON object from model text.
    Uses a JSONDecoder scan so it can recover a JSON object even if the model
    outputs extra content before/after the JSON.

    Returns:
        dict if a JSON object is found and parsed; else None.
    """
    if not text:
        return None

    s = text.strip()

    # If model outputs just an integer score, accept it as {"score": int}
    if re.fullmatch(r"-?\d+", s):
        try:
            return {"score": int(s)}
        except Exception:
            return None

    # Fast path: whole string is valid JSON dict
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    decoder = json.JSONDecoder()

    # Scan from each '{' position and try raw_decode
    start = 0
    while True:
        i = s.find("{", start)
        if i == -1:
            break
        try:
            obj, end = decoder.raw_decode(s, i)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = i + 1

    # Also handle the case where the JSON is an object inside code fences
    # (e.g., ```json {...} ```), by scanning the unfenced text.
    # This is still a scan, just with common fence markers removed.
    unfenced = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).replace("```", "")
    if unfenced != s:
        start = 0
        while True:
            i = unfenced.find("{", start)
            if i == -1:
                break
            try:
                obj, end = decoder.raw_decode(unfenced, i)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            start = i + 1

    # Regex fallback: if there's a "score": <int> somewhere but JSON is malformed
    m_score = re.search(r'"?score"?\s*[:=]\s*(-?\d+)', s)
    if m_score:
        try:
            return {"score": int(m_score.group(1))}
        except Exception:
            return None

    return None

def _coerce_evidence(evidence_val: Any) -> List[str]:
    """
    Convert various evidence formats into a list of strings.
    """
    if evidence_val is None:
        return []
    if isinstance(evidence_val, list):
        out = []
        for x in evidence_val:
            if isinstance(x, str):
                t = x.strip()
                if t:
                    out.append(t)
            else:
                t = str(x).strip()
                if t:
                    out.append(t)
        return out
    if isinstance(evidence_val, str):
        # Split on newlines / bullets
        lines = [ln.strip(" \t-•") for ln in evidence_val.splitlines()]
        return [ln for ln in lines if ln]
    # Fallback: string cast
    t = str(evidence_val).strip()
    return [t] if t else []

def build_judge_conv_prefix(
    audio1_path: str,
    audio2_path: str,
) -> List[Dict[str, Any]]:

    user_text = (
        """
        These are two distinct audios.
        Answer by Yes or No: Are those from the same speaker?
        """
    )
    system_text = "You are a helpful assistant that can understand and respond to speech."

    conv_prefix = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "audio", "path": audio1_path},
                {"type": "audio", "path": audio2_path},
            ],
        },
    ]
    return conv_prefix


from typing import Optional, Tuple, List, Dict, Any
import torch


def judge_turnover_score(
    processor: Qwen2_5OmniProcessor,
    model: Qwen2_5OmniThinkerForConditionalGeneration,
    audio1_path: str,
    audio2_path: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
):
    """
    Returns:
        raw_text: generated string
        logits: tensor of shape (seq_len, vocab_size)
    """

    conv_prefix = build_judge_conv_prefix(
        audio1_path=audio1_path,
        audio2_path=audio2_path,
    )

    inputs = processor.apply_chat_template(
        conv_prefix,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "return_dict_in_generate": True,   # IMPORTANT
        "output_scores": True,             # IMPORTANT
    }

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        pad_id = getattr(tok, "pad_token_id", None)
        eos_id = getattr(tok, "eos_token_id", None)
        if pad_id is None and eos_id is not None:
            gen_kwargs["pad_token_id"] = eos_id

    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Full generated sequence (prompt + completion)
    sequences = outputs.sequences

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = sequences[0, prompt_len:]  # (gen_len,)

    # Decode text
    raw_text = _decode_ids(processor, gen_ids).strip()

    # outputs.scores is a tuple of length gen_len
    # each element: (batch_size, vocab_size)
    # We stack and remove batch dim
    logits = torch.stack(outputs.scores, dim=0)  # (gen_len, batch, vocab)
    logits = logits[:, 0, :].detach().cpu().numpy()  # (gen_len, vocab_size)

    return raw_text, logits



def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# ------------ Audio helpers ------------

def normalize_audio(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    m = float(np.max(np.abs(x)))
    if m > 0:
        x = x / m
    return x.astype(np.float32)


def extract_speaker_segments_from_turnover(
    wav_path: Path,
    target_sr: int,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    name = wav_path.name
    if "_AtoB_" in name:
        tag = "AtoB"
    elif "_BtoA_" in name:
        tag = "BtoA"
    else:
        raise RuntimeError(f"Turnover file {name} does not contain AtoB/BtoA tag.")

    data, sr = sf.read(str(wav_path), always_2d=True)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Expected stereo turnover, got shape {data.shape} for {wav_path}.")

    L = data[:, 0]
    R = data[:, 1]

    if tag == "AtoB":
        spk1 = L
        spk2 = R
    else:  # BtoA
        spk1 = R
        spk2 = L

    # if sr != target_sr:
    #     spk1 = librosa.resample(spk1, orig_sr=sr, target_sr=target_sr)
    #     spk2 = librosa.resample(spk2, orig_sr=sr, target_sr=target_sr)
    #     sr = target_sr

    spk1 = normalize_audio(spk1)
    spk2 = normalize_audio(spk2)
    return spk1, spk2, sr, tag


# ------------ Core per-conversation scoring ------------

def run_turnover_extractor_for_row(
    turnover_script: Path,
    prompt_id_unique: str,
    a_id: str,
    b_id: str,
):
    cmd = [
        sys.executable,
        str(turnover_script),
        "--prompt-id-unique",
        prompt_id_unique,
        "--a-id",
        a_id,
        "--b-id",
        b_id,
    ]
    print("Running turnover_extractor.py:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run filter_roles -> turnover_extractor -> Qwen scoring (Evidence-First + scalar score + BPC)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed forwarded to filter_roles.py for deterministic subset selection. "
            "Default: None (non-deterministic sampling)."
        ),
    )

    parser.add_argument(
        "--trial",
        type=str,
        default='o',
    )
    parser.add_argument(
        "--start-over",
        type=_str2bool,
        default=True,
        help=(
            "If true, overwrite the filtered CSV and re-run from scratch (current behavior). "
            "If false, append only new rows from the newly-sampled subset and resume unfinished evaluation. "
            "Default: true."
        ),
    )
    parser.add_argument(
        "--filtered-csv",
        type=Path,
        default=Path("voxceleb1_outputs.csv"),
        help="Path to write/read filtered subset CSV (default: ./filtered_subset.csv).",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen2.5-Omni-3B",
        help="Qwen2.5-Omni Thinker checkpoint ID (default: Qwen/Qwen2.5-Omni-3B).",
    )
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=16,
        help="Max new tokens to generate for judge JSON output (default: 128).",
    )
    parser.add_argument(
        "--judge-do-sample",
        action="store_true",
        help="If set, use sampling (stochastic decoding). Otherwise deterministic decoding.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (used only if --judge-do-sample).",
    )
    parser.add_argument(
        "--judge-top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (used only if --judge-do-sample).",
    )
    return parser.parse_args()

def _fmt_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS; return ?? if unknown."""
    try:
        s = float(seconds)
    except Exception:
        return "??:??:??"
    if not np.isfinite(s) or s < 0:
        return "??:??:??"
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def main():
    args = parse_args()
    # Step 1: load CSV with trials
    df = pd.read_csv(f"data/trials_{args.trial}.csv")
    df['modelaudio'] = [f'/export/corpora5/VoxCeleb1_v2/wav/{i[:7]}/{i[8:-6]}/{i[-5:]}.wav' for i in df['modelid']]
    df['segmentaudio'] = [f'/export/corpora5/VoxCeleb1_v2/wav/{i[:7]}/{i[8:-6]}/{i[-5:]}.wav' for i in df['segmentid']]
    outputs = []
    n_total=5000
    df = df.sample(n=n_total, random_state=42).reset_index()

    os.makedirs(f"logits/qwen2.5/vox1-{args.trial}", exist_ok=True)
    
    out_cols = ["raw"]
    for col in out_cols:
        if col not in df.columns:
            df[col] = ""

    # Step 2: load Qwen model
    print(f"Loading Qwen model: {args.qwen_model}")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.qwen_model)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.qwen_model,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    audio_fe = getattr(processor, "feature_extractor", None) or getattr(processor, "audio_processor", None)
    target_sr = getattr(audio_fe, "sampling_rate", 16000)

    tmpdir = Path(tempfile.mkdtemp(prefix="qwen_turnovers_"))

    run_t0 = time.perf_counter()

    n_eval_total = n_total
    n_eval_done = 0
    
    print("What are Qwen2.5 Omni tokens for yes and no?")
    tok = getattr(processor, "tokenizer", None)
    yes_ids = tok.encode(" Yes", add_special_tokens=False)
    no_ids  = tok.encode(" No", add_special_tokens=False)
    print("Yes ids:", yes_ids)
    print("No ids:", no_ids)

    out_dir = Path.cwd() / f"voxceleb1-test-{args.trial}"
    os.makedirs(out_dir, exist_ok=True)
    for idx, row in df.iterrows():
        model_file = row['modelaudio']
        segment_file = row['segmentaudio']
        model_id = row['modelid']
        segment_id = row['segmentid']
        target = row['targettype']=='target'

        
        # base_prefix = f"{idx}_model-{model_id}_segment-{segment_id}_"

        # --- timing tracker: elapsed / estimated total for remaining evaluations ---
        n_eval_done += 1
        elapsed_s = time.perf_counter() - run_t0
        est_total_s = (elapsed_s / n_eval_done) * n_eval_total if n_eval_done > 0 else float("nan")

        print(
            f"[progress] Conversation {idx}/{n_total} "
            f"[eval {n_eval_done}/{n_eval_total}] "
            f"{_fmt_hms(elapsed_s)}/{_fmt_hms(est_total_s)} "
            f"(idx={idx}, prompt_id={idx}, model_id={model_id}, segment_id={segment_id})",
            flush=True,
        )
        
        raw, logits = judge_turnover_score(
            processor=processor,
            model=model,
            audio1_path=model_file,
            audio2_path=segment_file,
            max_new_tokens=args.judge_max_new_tokens,
            do_sample=args.judge_do_sample,
            temperature=args.judge_temperature,
            top_p=args.judge_top_p,
        )

        raw = raw.replace('\n', ' ').replace('\t', ' ')
        df.at[idx, "raw"] = raw

        Same = model_id.split('-')[0]==segment_id.split("-")[0]
        print(
            f"[outputs] Conversation {idx}/{n_total} "
            f"raw {raw}\t"
            f"(same={Same}, model_id={model_id}, segment_id={segment_id})\n",
            flush=True,
        )
        np.save(f"logits/qwen2.5/vox1-{args.trial}/{model_id}_{segment_id}.npy", logits)


    print(f"\nUpdated filtered CSV in-place: {args.filtered_csv}")
    df.to_csv(args.filtered_csv, index=False)


if __name__ == "__main__":
    main()
