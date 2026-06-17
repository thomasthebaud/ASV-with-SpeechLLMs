# Speaker Verification with Speech-Aware LLMs

Code for the paper **Speaker Verification with Speech-Aware LLMs: Evaluation and Augmentation** by Thomas Thebaud, Yuzhe Wang, Laureano Moro-Velazquez, Jesus Villalba-Lopez, and Najim Dehak.

This repository contains two related code paths:

- `LLM_eval/`: direct evaluation of off-the-shelf speech-aware LLMs on speaker verification trials.
- root-level training code: lightweight speaker-verification augmentation of an LLM by injecting audio/speaker embeddings through a learned connector and training LoRA adapters.

Paper:

- arXiv: https://arxiv.org/abs/2603.10827
- PDF: https://arxiv.org/pdf/2603.10827
- Published at Interspeech 2026.

Contact: Thomas Thebaud, Johns Hopkins University. Please use the contact information from the paper or publication page.

## Repository Layout

```text
.
├── LLM_eval/                         # Off-the-shelf speech-aware LLM evaluation
│   ├── data/trials_{o,e,h}.csv        # VoxCeleb1 trial lists used by the scripts
│   ├── model.py                       # Wrapper selecting Qwen, Kimi, or AudioFlamingo
│   ├── overlay.py                     # Model-specific audio prompting wrappers
│   ├── run_ASV_all.py                 # Kimi/AudioFlamingo-style confidence scoring script
│   ├── run_ASV_gemini.py              # Gemini API confidence scoring script
│   ├── run_ASV_qwen.py                # Qwen confidence scoring script
│   ├── run_ASV_qwen_logits.py         # Qwen Yes/No logit scoring script
│   ├── get_eer_from_logs_*.py         # Log parsers and EER computation
│   └── requirements_kimi.txt          # Extra dependencies for Kimi-style evaluation
├── config/
│   ├── data/                          # Train/dev/test split configs
│   └── model/                         # Connector configs
├── data/                              # CSV manifests and trial files
├── launch/ASV_LLMs_experiments/       # Example SLURM launchers
├── local/                             # Data-prep and older/local experiment helpers
├── model/                             # Audio encoder, connector, and LLM loading code
├── dataset_ASV.py                     # Pair sampling and collators for ASV training/testing
├── train_ASV.py                       # Train speaker-aware LLM augmentation
├── test_ASV.py                        # Evaluate trained augmentation checkpoints
├── trainer_ASV.py                     # PyTorch Lightning module
├── utils.py                           # CLI/config assembly
├── environment.yml                    # Conda environment used for augmentation experiments
└── requirements.txt                   # Python package snapshot
```

## Setup

The augmentation experiments were run with a conda environment close to `environment.yml`:

```bash
conda env create -f environment.yml
conda activate speechllm
```

For Kimi/open-weight speech-aware LLM evaluation, install the additional packages in `LLM_eval/requirements_kimi.txt` in an environment compatible with the target model:

```bash
pip install -r LLM_eval/requirements_kimi.txt
```

The scripts assume access to VoxCeleb audio and trial manifests. Several evaluation scripts currently contain absolute paths from the original cluster, especially:

- `/export/corpora5/VoxCeleb1_v2/wav/...`
- `/export/corpora5/VoxCeleb1_v2/vox1_meta.csv`
- `/home/tthebau1/SPEAR/followup7B/...`

Edit these paths before running on a new machine. API-based scripts should also be changed to read keys from environment variables or a private local file; do not commit real API keys.

## LLM Evaluation

The `LLM_eval/` directory evaluates whether existing speech-aware LLMs can perform automatic speaker verification directly from two audio files.

The evaluation protocol is:

1. Load a VoxCeleb1 trial list: `LLM_eval/data/trials_o.csv`, `trials_e.csv`, or `trials_h.csv`.
2. Convert each `modelid` and `segmentid` trial entry into a VoxCeleb waveform path.
3. Prompt a speech-aware LLM with both audio files.
4. Obtain a verification score either from a requested confidence score in `[0, 100]` or from the Yes/No token likelihood ratio when logits are available.
5. Parse logs and compute EER.

### Confidence-Score Evaluation

For open-weight wrappers such as Kimi Audio and AudioFlamingo, use `run_ASV_all.py` after selecting the model in the script:

```bash
cd LLM_eval
python run_ASV_all.py
```

By default, `run_ASV_all.py` uses:

```python
model_name = "moonshotai/Kimi-Audio-7B-Instruct"
```

The wrapper is intended for:

```python
"moonshotai/Kimi-Audio-7B-Instruct"
"nvidia/audio-flamingo-3-hf"
```

Qwen has dedicated scripts, described below.

Before running, update the script's `logfile` location and VoxCeleb audio root if your data are not in the original cluster paths.

For Gemini API evaluation:

```bash
cd LLM_eval
python run_ASV_gemini.py --trial o
python run_ASV_gemini.py --trial e
python run_ASV_gemini.py --trial h
```

`run_ASV_gemini.py` prints `[outputs]` lines that can be redirected to a log file. Replace the hardcoded API key with a private key loaded from your environment, for example `GOOGLE_API_KEY`.

For Qwen confidence-score evaluation:

```bash
cd LLM_eval
python run_ASV_qwen.py --trial o --qwen-model Qwen/Qwen2.5-Omni-3B
```

Useful options:

- `--trial`: one of `o`, `e`, `h`.
- `--filtered-csv`: output CSV path, default `voxceleb1_outputs.csv`.
- `--qwen-model`: Hugging Face model ID, default `Qwen/Qwen2.5-Omni-3B`.
- `--judge-max-new-tokens`: generation length for the answer.

### Logit-Based Evaluation

When the model exposes logits, use `run_ASV_qwen_logits.py`:

```bash
cd LLM_eval
python run_ASV_qwen_logits.py --trial o --qwen-model Qwen/Qwen2.5-Omni-3B
```

This follows the paper's log-likelihood-ratio scoring idea:

```text
score = log p(Yes | prompt) - log p(No | prompt)
```

Use this style of score when you want a finer-grained ASV score than a model-generated integer confidence.

### Computing EER From Logs

After evaluation, use the parser matching the model/log format:

```bash
cd LLM_eval
python get_eer_from_logs_kimi.py
python get_eer_from_logs_qwen.py
python get_eer_from_logs_gemini.py
python get_eer_from_logs_flamingo.py
```

These scripts expect logs under names such as `outputs/Kimi-Audio-7B-Instruct_vox1-o.log` or `slurm_output/qwen2.5-vox1-o.log`. Edit the filename patterns at the bottom of each parser if your logs are elsewhere.

`get_stats_from_logs.py` computes additional coarse speaker-characteristic statistics, such as gender and accent mentions, using VoxCeleb1 metadata. Update the metadata path before use.

## LLM Augmentation

The augmentation code trains a speaker-aware LLM for ASV by feeding audio or precomputed speaker embeddings into an LLM through a connector.

The main components are:

- `dataset_ASV.py`: samples anchor, same-speaker, and different-speaker pairs from CSV manifests.
- `model/encoder.py`: loads a speech encoder such as WavLM or passes through precomputed embeddings.
- `model/connector.py`: projects audio/speaker embeddings into the LLM embedding dimension using `linear`, `mlp`, or `cnn` connectors.
- `model/llm.py`: loads TinyLLaMA or Ministral3 and optionally adds LoRA adapters.
- `trainer_ASV.py`: computes next-token training and validation/test EER.
- `train_ASV.py`: training entry point.
- `test_ASV.py`: checkpoint evaluation entry point.

### Data Manifests

Training and testing read CSV files from `data/`. For raw-audio training, each CSV should include at least:

```text
audio_path,speaker
```

For precomputed speaker embeddings, each CSV should include:

```text
embedding_path,speaker
```

Trial-style test CSVs additionally use enrollment/test identifiers and targets as prepared in the provided `data/*.csv` files.

Dataset split selection is controlled by JSON files in `config/data/`, for example:

```text
config/data/ASV_voxceleb2_ecapatdnn.json
config/data/ASV_voxceleb2_XS_ecapatdnn.json
config/data/ASV_voxceleb2_XXS_ecapatdnn.json
config/data/ASV_voxceleb1.json
```

Connector settings are controlled by JSON files in `config/model/`, for example:

```text
config/model/linear192.json
config/model/linear192_2512.json
config/model/linear.json
```

### Training Examples

Train the ECAPA-TDNN embedding + linear connector + TinyLLaMA setup:

```bash
python train_ASV.py \
  --encoder 'precomputed/ecapa-tdnn' \
  --connector 'linear192' \
  --llm 'TinyLlama-1.1B-Chat-v1.0' \
  --batch-size 32 \
  --lr 0.0001 \
  --group 'ASV' \
  --use-config ASV_voxceleb2_ecapatdnn.json \
  --total-training-epoch 100 \
  --nickname "_ASV"
```

Some launch scripts also show raw-audio WavLM + CNN experiments. This snapshot does not include the referenced `cnn_str1.2.1.json` connector file, so add the matching connector config under `config/model/` before running this variant:

```bash
python train_ASV.py \
  --encoder 'microsoft/wavlm-base-plus' \
  --connector 'cnn_str1.2.1' \
  --llm 'TinyLlama-1.1B-Chat-v1.0' \
  --batch-size 4 \
  --lr 0.00001 \
  --meanpool -1 \
  --group 'ASV' \
  --use-config ASV_voxceleb1.json \
  --total-training-epoch 20 \
  --nickname "_ASV"
```

Equivalent SLURM launchers are available in:

```text
launch/ASV_LLMs_experiments/train/
```

For example:

```bash
sbatch launch/ASV_LLMs_experiments/train/EcapaTDNN_linear_TinyLlama_ASV.sh
```

Checkpoints are written to:

```text
checkpoints/<group>/<model_name>/
```

This repository's `checkpoints` path is a symlink in the original workspace, so update it if needed.

### Testing a Trained Augmentation

Use `test_ASV.py` with the same model/data options and the epoch to test:

```bash
python test_ASV.py \
  --encoder 'precomputed/ecapa-tdnn' \
  --connector 'linear192' \
  --llm 'TinyLlama-1.1B-Chat-v1.0' \
  --batch-size 32 \
  --lr 0.0001 \
  --group 'ASV' \
  --use-config ASV_voxceleb2_ecapatdnn.json \
  --epoch-to-test 23 \
  --nickname "_ASV"
```

The script loads checkpoints from:

```text
checkpoints/<group>/<model_name>/<model_name>epoch-epoch=<epoch>.ckpt
```

Test predictions and logs are written to:

```text
exp/test_predictions/<model_name>/<test_set>/
```

Equivalent SLURM launchers are available in:

```text
launch/ASV_LLMs_experiments/test/
```

### Useful Training Options

- `--encoder`: audio encoder name, e.g. `precomputed/ecapa-tdnn` or `microsoft/wavlm-base-plus`.
- `--connector`: connector config name from `config/model/` without `.json`.
- `--llm`: supported shortcuts include `TinyLlama-1.1B-Chat-v1.0` and `Ministral-3-3B-Base-2512`.
- `--use-config`: data config file from `config/data/`.
- `--no-lora`: freeze the LLM and train only the connector.
- `--ft-encoder`: fine-tune the audio encoder.
- `--ft-layers`: encoder layer range to fine-tune, formatted like `0-6`.
- `--use-text`, `--prob-text`, `--no-audio`: ablation switches for text/audio inputs.
- `--meanpool`: temporal pooling setting used in the connector path.
- `--epoch-to-test`: checkpoint epoch for `test_ASV.py`; in the current code, `last` maps to the base/untrained initialization path rather than automatically resolving `last.ckpt`.

## Citation

Please cite the paper if this code is useful for your research:

```bibtex
@inproceedings{thebaud2026speaker,
  title     = {Speaker Verification with Speech-Aware LLMs: Evaluation and Augmentation},
  author    = {Thebaud, Thomas and Wang, Yuzhe and Moro-Velazquez, Laureano and Villalba-Lopez, Jesus and Dehak, Najim},
  booktitle = {Proceedings of Interspeech 2026},
  year      = {2026},
  note      = {arXiv:2603.10827},
  url       = {https://arxiv.org/abs/2603.10827},
  doi       = {10.48550/arXiv.2603.10827}
}
```

An arXiv-only reference is:

```bibtex
@misc{thebaud2026speakerarxiv,
  title         = {Speaker Verification with Speech-Aware LLMs: Evaluation and Augmentation},
  author        = {Thomas Thebaud and Yuzhe Wang and Laureano Moro-Velazquez and Jesus Villalba-Lopez and Najim Dehak},
  year          = {2026},
  eprint        = {2603.10827},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2603.10827}
}
```

## License

See `LICENSE`.
