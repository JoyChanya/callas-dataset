# Callas — YouTube Thai Scam-Call Dataset Builder

Collects Thai scam-call conversations from public YouTube clips, transcribes
them, splits them into speaker turns, and produces a labeled CSV:

> **text → is_scam (scam / not scam) → scam_type**

First data source for the **Callas** project (Thai scam call + message
detection). Later combined with synthetic dialogues and existing datasets
(e.g. `Paam1/ThaiScamCall` on Hugging Face).

---

## 1. Setup

### Python (IMPORTANT)
The ML stack (torch / transformers / pyannote) has **no wheels for Python
3.14**. Use **3.11 or 3.12**. A venv is already created with `python3.12`:

```bash
source .venv/bin/activate          # macOS / Linux
# or call ./.venv/bin/python directly
```

To recreate it:
```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

### System dependency: ffmpeg
```bash
brew install ffmpeg        # macOS
```

### Hugging Face token (only for the `diarize` stage)
`pyannote` is gated. Create a token and **accept the model terms** on these two
pages while logged in to HF:
- https://hf.co/pyannote/speaker-diarization-3.1
- https://hf.co/pyannote/segmentation-3.0

Then:
```bash
cp .env.example .env        # put HF_TOKEN=hf_xxx in .env
```

### Compute
`device: auto` in `config.yaml` picks **cuda → mps → cpu**. On Apple Silicon
(M1/M2) it uses **mps**; `pyannote` falls back to CPU automatically if an mps
op is unsupported. No NVIDIA GPU required, but transcribe/diarize are slower.

### Recommended: run everything on Colab (`callas_colab.ipynb`)
An 8 GB M1 is limited to the **medium** model and gives rough ASR on
phone-quality audio. Easiest + best-quality path is to run the **whole
pipeline on a free Colab GPU** with the **large** model — no local machine
needed. The notebook:

1. installs deps (pins `transformers<5`) + clones this repo,
2. mounts Google Drive and symlinks `data/` → `MyDrive/callas-data` so all
   output persists across sessions,
3. runs search → download → transcribe (large, GPU) → diarize → segment →
   clean → prelabel,
4. you edit `review.csv` → save `review_done.csv` → run export.

Open `callas_colab.ipynb` in Colab (GPU runtime), run top to bottom. Every
stage is resumable.

Overrides it uses (also work locally):
```bash
python run_pipeline.py --stage transcribe --device cuda \
    --model biodatlab/whisper-th-large-v3-combined
python run_pipeline.py --stage diarize --device cuda
```
> transformers **5.x** currently does not emit Whisper segment timestamps; the
> notebook pins `transformers<5`, which does. Without timestamps the `segment`
> stage falls back to a single speaker.
> If YouTube shows a bot-check on Colab, set `youtube.cookiefile` in
> `config.yaml` (see the optional cookies cell in the notebook).

---

## 2. Pipeline

```
search → download → transcribe → diarize → segment → clean & mask → prelabel → human review → export
```

| Stage | Command | Output | Needs |
|---|---|---|---|
| 1 search | `--stage search` | `data/meta/videos.csv` | net |
| 2 download | `--stage download` | `data/raw/audio/{id}.wav`, subs | net, ffmpeg |
| 3 transcribe | `--stage transcribe` | `data/interim/asr/{id}.json` | torch, model DL |
| 4 diarize | `--stage diarize` | `data/interim/diar/{id}.rttm` | **HF_TOKEN** |
| 5 segment | `--stage segment` | `data/interim/turns/{id}.json` | — |
| 6 clean | `--stage clean` | `data/interim/clean/{id}.json` | pythainlp |
| 7 prelabel | `--stage prelabel` | `data/review/review.csv` | — |
| 8 review | *(human edits CSV)* | `data/review/review_done.csv` | you |
| 9 export | `--stage export` | `data/final/*.csv` + `stats.md` | — |

### CLI
```bash
python run_pipeline.py --stage search --max-per-query 50
python run_pipeline.py --stage download
python run_pipeline.py --stage all --limit 5     # quick test on 5 videos
python run_pipeline.py --stage prelabel --llm    # optional local vLLM
python run_pipeline.py --stage export
```
Every stage is **resumable** — finished items are skipped, so re-running is safe.

### Human review (stage 8)
Open `data/review/review.csv`. For each turn, fill the empty columns:
`speaker_role` (scammer / victim / caller_legit / receiver / narrator),
`is_scam` (1/0), `scam_type`, `keep` (1/0), `text_corrected`, `notes`.
Save as **`review_done.csv`**. Rows with `narrator` or `keep=0` are dropped on
export. (Optional Gradio app `src/review_app.py` can be added later.)

---

## 3. Output files (`data/final/`)
UTF-8 **with BOM** so Thai shows correctly in Excel.
- `turns.csv` — one row per speaker turn (main file).
- `conversations.csv` — one row per call (all turns joined as `[ROLE] text`).
- `simple.csv` — `text, is_scam, scam_type` (conversation level), ready for
  WangchanBERTa / TF-IDF.
- `stats.md` — counts of videos / conversations / turns / hours, class balance.

### scam_type labels (Callas proposal, Table 1)
`fake_authority`, `romance`, `investment`, `phishing_otp`, `lottery_prize`,
`other_scam`, `none`.

---

## 4. Data & copyright policy
- **Do not commit or redistribute audio/video.** Raw media lives in `data/raw/`
  and `data/interim/` — both git-ignored. Released data has only video id, URL,
  timestamps, text, and labels.
- **PII masking (Thai PDPA):** phone → `[PHONE]`, 13-digit id → `[ID_CARD]`,
  bank account → `[ACCOUNT]`, URL → `[URL]`, person names → `[NAME]`. Done
  before anything reaches `data/final/`. Organisation names (ธนาคาร, สภ., DHL)
  are **kept** — they are scam signals. Unmasked text stays only in
  git-ignored `data/interim/`.

---

## 5. Known limitations
- Many clips are prank / "scam-baiting" calls, so the "victim" often knows it
  is a scam — behavior differs from real victims.
- News clips mix anchor narration with call audio; narrator turns must be
  removed in review.
- Phone-quality audio and background music reduce ASR accuracy; human
  correction is required.
- Legitimate-call examples are scarce on YouTube; the not-scam class will need
  synthetic dialogues later.

---

## 6. Tests
```bash
./.venv/bin/python -m pytest tests/ -q
```
Covers PII-masking regexes, ASR↔diarization alignment, and CSV schema +
label coalescing.
