# from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
# from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
# from huggingface_hub import snapshot_download

from kimia_infer.api.kimia import KimiAudio

import torch, torchaudio
# from peft import PeftModel
# import os
# import soundfile as sf

MAX_NEW_TOKENS=256

def _decode_ids(processor, token_ids: torch.Tensor) -> str:
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

def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)  # (channels, time)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0)  # (time,)

class Qwen2_5_overlay():
    def __init__(self):
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-3B",
            device_map="auto",
            torch_dtype="auto",
        )
        self.model.eval()
    def process(self, audio1, audio2, question, return_logits=False):
        user_text = ( question )
        system_text = "You are a helpful assistant that can understand and respond to speech."
        conv_prefix = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "audio", "path": audio1},
                {"type": "audio", "path": audio2},
            ]}]
        inputs = self.processor.apply_chat_template(
            conv_prefix,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        if return_logits:
            gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "return_dict_in_generate": True,   # IMPORTANT
            "output_scores": True,             # IMPORTANT
            }
        else:
            gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}

        # Avoid generate() warnings when pad_token_id is unset
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            pad_id = getattr(tok, "pad_token_id", None)
            eos_id = getattr(tok, "eos_token_id", None)
            if pad_id is None and eos_id is not None:
                gen_kwargs["pad_token_id"] = eos_id

        with torch.no_grad():
            if return_logits:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                sequences = outputs.sequences
                logits = torch.stack(outputs.scores, dim=0)  # (gen_len, batch, vocab)
            else:
                sequences = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = sequences[0, prompt_len:] 
        raw_text = _decode_ids(self.processor, gen_ids).strip()
    
        if return_logits: return raw_text, logits[:, 0, :].detach().cpu().numpy()
        else: return raw_text

class Kimi_overlay():
    def __init__(self):
        self.model = KimiAudio(model_path="moonshotai/Kimi-Audio-7B-Instruct", load_detokenizer=False)
    def process(self, audio1, audio2, question, return_logits=False):
        # audio1 = load_audio(audio1)
        # audio2 = load_audio(audio2)

        # Move to device
        # audio1 = audio1.to('cuda')
        # audio2 = audio2.to('cuda')

        messages = [
            {"role": "user", "message_type": "text", "content": question},
            {"role": "user", "message_type": "audio", "content": audio1},
            {"role": "user", "message_type": "audio", "content": audio2},
        ]
        _, raw_output = self.model.generate(
            messages,
            output_type="text",
            max_new_tokens=128,
            text_temperature=0.0,
            text_top_k=1,
        )
        return raw_output

class Flamingo_overlay():
    def __init__(self):
        model_id = "nvidia/audio-flamingo-3-hf"
        local_id = snapshot_download(model_id)

        self.processor = AutoProcessor.from_pretrained(local_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(local_id, device_map="auto")

        non_lora_path = os.path.join(local_id, "think", "non_lora_trainables.bin")
        non_lora_trainables = torch.load(non_lora_path)
        self.model.load_state_dict(non_lora_trainables, strict=False)

        self.model = PeftModel.from_pretrained(self.model, local_id, subfolder="think")

    def process(self, audio1, audio2, question, return_logits=False):
        output_path = self.concat_audios(audio1, audio2)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text","text": question},
                    {"type": "audio","path": output_path}
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return decoded_outputs[0]

    def concat_audios(self, path1, path2, output_path="temp.wav"):
        # Load audio files
        audio1, sr1 = sf.read(path1)
        audio2, sr2 = sf.read(path2)

        # Concatenate along time axis
        concatenated = list(audio1) + list(audio2)

        # Save result
        sf.write(output_path, concatenated, sr1)
        return output_path
