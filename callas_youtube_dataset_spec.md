# Callas – YouTube Thai Scam Call Dataset Builder (Spec for Claude Code)

## 1. Goal
Build a Python pipeline that collects Thai scam-call conversations from public YouTube clips, transcribes them, splits them into speaker turns, and produces a labeled CSV:

**text → is_scam (scam / not scam) → scam_type**

This dataset is the first data source for the Callas project (a Thai scam call and message detection system). Later it will be combined with synthetic dialogues and existing datasets (e.g. `Paam1/ThaiScamCall` on Hugging Face).

## 2. Constraints
- Python 3.10+, runs on **one local GPU**. No paid or external cloud APIs (no OpenAI, Google, etc.). Hugging Face model downloads are allowed.
- **Do not commit or redistribute audio/video files.** Raw media lives in `data/raw/` and is git-ignored. Released data contains only video ID, URL, timestamps, text and labels.
- Personal data (names, phone numbers, bank account numbers, ID card numbers) must be masked before anything is written to `data/final/` (Thai PDPA).
- Every stage must be **resumable**: skip items already processed, so the pipeline can be re-run safely.
- All settings in one `config.yaml`.

## 3. Pipeline overview

```
search → download → transcript → diarize → segment → clean & mask → pre-label → human review → export CSV
```

| Stage | Input | Output |
|---|---|---|
| 1. search | keyword list | `data/meta/videos.csv` |
| 2. download | videos.csv | `data/raw/audio/{id}.wav`, `data/raw/subs/{id}.th.vtt` |
| 3. transcribe | wav | `data/interim/asr/{id}.json` (word/segment timestamps) |
| 4. diarize | wav | `data/interim/diar/{id}.rttm` |
| 5. segment | asr + diar | `data/interim/turns/{id}.json` (one item per speaker turn) |
| 6. clean & mask | turns | `data/interim/clean/{id}.json` |
| 7. pre-label | clean turns | `data/review/review.csv` |
| 8. review | review.csv (edited by human) | `data/review/review_done.csv` |
| 9. export | review_done.csv | `data/final/*.csv` |

## 4. Stage details

### 4.1 Search (`src/search.py`)
- Use `yt-dlp` with `ytsearchN:<query>` and `--flat-playlist` to collect video IDs, titles, channel, duration, upload date.
- Default Thai queries (keep in config, easy to extend):
  - Scam: `แกล้งแก๊งคอลเซ็นเตอร์`, `ป่วนแก๊งคอลเซ็นเตอร์`, `อัดเสียงแก๊งคอลเซ็นเตอร์`, `เสียงจริงมิจฉาชีพโทรมา`, `คุยกับมิจฉาชีพ`, `แก๊งคอลเซ็นเตอร์ อ้างตำรวจ`, `แก๊งคอลเซ็นเตอร์ พัสดุตกค้าง`, `มิจฉาชีพ โทรหลอก OTP`, `หลอกลงทุน คลิปเสียง`, `โทรแจ้งถูกรางวัล มิจฉาชีพ`
  - Not scam (legitimate calls): `ตัวอย่างบทสนทนาทางโทรศัพท์ ภาษาไทย`, `บทสนทนา call center ธนาคาร`, `โทรศัพท์ติดต่อลูกค้า ตัวอย่าง`
- Filter: duration between 30 s and 30 min (configurable). Deduplicate by video ID.
- Output columns: `video_id, url, title, channel, duration_s, upload_date, query, query_group (scam|legit), status`.
- Allow a manual `data/meta/manual_urls.txt` (one URL per line) to be merged in.

### 4.2 Download (`src/download.py`)
- `yt-dlp` best audio → convert to **16 kHz mono WAV** with ffmpeg.
- Also try to fetch Thai subtitles (manual first, then auto) with `writesubtitles`, `writeautomaticsub`, `subtitleslangs: ["th"]`. Record `has_manual_subs`, `has_auto_subs` in videos.csv.
- Rate-limit politely (sleep between downloads, configurable). Log failures and continue.

### 4.3 Transcribe (`src/transcribe.py`)
- Default model: Thonburian Whisper `biodatlab/whisper-th-medium-combined` via `transformers` pipeline, `chunk_length_s=30`, `return_timestamps=True`. Model name configurable (so large variants or pythaiasr can be swapped in).
- Save segments with start/end times as JSON.
- If YouTube subtitles exist, also save them parsed to the same JSON format (`source: "youtube_manual" | "youtube_auto" | "asr"`) so they can be compared later.

### 4.4 Diarize (`src/diarize.py`)
- `pyannote/speaker-diarization-3.1` (requires `HF_TOKEN` env var and accepting the model terms on Hugging Face).
- Allow `min_speakers` / `max_speakers` in config (default 2–4, because news clips may include an anchor).

### 4.5 Segment into turns (`src/segment.py`)
- Align ASR segments to diarization by maximum time overlap; merge consecutive segments of the same speaker into one **turn**.
- Each turn: `turn_index, speaker (SPEAKER_00…), start, end, text`.
- Drop turns shorter than a configurable minimum (e.g. < 0.5 s or empty text).

### 4.6 Clean and mask PII (`src/clean.py`)
- Normalize Thai text with `pythainlp.util.normalize`; strip repeated whitespace.
- Mask with regex:
  - Thai phone numbers (`0\d{1,2}-?\d{3}-?\d{4}`, `+66…`) → `[PHONE]`
  - 13-digit ID numbers → `[ID_CARD]`
  - bank account numbers (10–12 digits, with or without dashes) → `[ACCOUNT]`
  - URLs → `[URL]`
- Person names: use a PyThaiNLP NER model to replace `PERSON` entities with `[NAME]`. Keep organization names (e.g. "ธนาคาร", "สภ.", "DHL") because they are important scam signals.
- Keep the unmasked text only in `data/interim/` (git-ignored) for reviewers.

### 4.7 Pre-label (`src/prelabel.py`)
Goal: speed up human labeling. All pre-labels must be reviewed by a human.
- Default method: keyword rules (config file `labels/keywords.yaml`) per scam type, e.g. ตำรวจ / ฟอกเงิน / บัญชีม้า → fake_authority; OTP / รหัส / ลิงก์ / พัสดุ → phishing_otp; ผลตอบแทน / ลงทุน / ถอนเงินไม่ได้ → investment; รางวัล / โชคดี / ค่าธรรมเนียม → lottery_prize; ที่รัก / แฟน / ฉุกเฉิน → romance.
- Optional method (flag `--llm`): a local open-source Thai-capable instruct model served through **vLLM** (OpenAI-compatible endpoint on localhost). Prompt it with the whole conversation and ask for JSON: `{is_scam, scam_type, scammer_speaker, scam_onset_turn, rationale}`.
- The query group from search (scam/legit) is a weak hint only, not a label.
- Also guess speaker roles per video: `scammer`, `victim`, `caller_legit`, `receiver`, `narrator` (news anchor / YouTuber talking to the audience, not part of the call).

### 4.8 Human review (`data/review/review.csv`)
- One row per turn, sorted by video then turn. Columns include pre-labels plus empty columns for the reviewer to correct: `speaker_role`, `is_scam`, `scam_type`, `keep (1/0)`, `text_corrected`, `notes`.
- Optional (nice to have): a small **Gradio** app `src/review_app.py` that shows one video at a time, plays each turn's audio slice, and lets the reviewer edit text and labels, saving to `review_done.csv`.

### 4.9 Export (`src/export.py`)
Produce three CSV files in `data/final/` (UTF-8 with BOM so Thai displays correctly in Excel):

**a) `turns.csv`** – one row per speaker turn (main file)

| column | description |
|---|---|
| `id` | `{video_id}_{turn_index:03d}` |
| `video_id`, `url` | source |
| `start`, `end` | seconds |
| `turn_index` | order in the call |
| `speaker_role` | scammer / victim / caller_legit / receiver |
| `text` | cleaned, PII-masked, human-corrected text |
| `is_scam` | 1 = scam conversation, 0 = not scam |
| `scam_type` | fake_authority / romance / investment / phishing_otp / lottery_prize / other_scam / none |
| `is_after_onset` | 1 if this turn is at or after the turn where scam intent first appears |
| `transcript_source` | asr / youtube_manual / youtube_auto |
| `label_source` | human / rule / llm |

Rows with `speaker_role = narrator` or `keep = 0` are excluded.

**b) `conversations.csv`** – one row per call: `video_id, url, text` (all turns joined as `[ROLE] text` lines), `is_scam, scam_type, n_turns, scam_onset_turn, duration_s`.

**c) `simple.csv`** – the minimal file: `text, is_scam, scam_type` (conversation level), ready for WangchanBERTa / TF-IDF training.

Also print and save `data/final/stats.md`: number of videos, conversations, turns, hours of audio, counts per `scam_type`, and scam vs not-scam balance.

## 5. Label definitions (from Callas proposal, Table 1)
| scam_type | meaning |
|---|---|
| `fake_authority` | impersonates police, bank, customs, DSI, etc.; demands transfer to a "safe" account |
| `romance` | emotional relationship, then money for a fake emergency |
| `investment` | high returns, fake profits, fees before withdrawal |
| `phishing_otp` | impersonates bank/courier to get OTP or make victim open a link / install an app |
| `lottery_prize` | fake prize requiring a fee or bank details |
| `other_scam` | clearly a scam but none of the above |
| `none` | legitimate call |

## 6. Project structure
```
callas-dataset/
├── config.yaml
├── requirements.txt
├── README.md
├── .gitignore            # data/raw, data/interim, .env
├── labels/keywords.yaml
├── src/
│   ├── search.py
│   ├── download.py
│   ├── transcribe.py
│   ├── diarize.py
│   ├── segment.py
│   ├── clean.py
│   ├── prelabel.py
│   ├── review_app.py     # optional
│   ├── export.py
│   └── utils.py          # logging, paths, resumable helpers
├── run_pipeline.py       # CLI: run all stages or one stage
├── data/
│   ├── meta/  raw/  interim/  review/  final/
└── tests/
```

CLI examples:
```
python run_pipeline.py --stage search --max-per-query 50
python run_pipeline.py --stage download
python run_pipeline.py --stage all --limit 5        # quick test on 5 videos
python run_pipeline.py --stage export
```

## 7. Dependencies
`yt-dlp`, `ffmpeg` (system), `torch`, `transformers`, `accelerate`, `pyannote.audio`, `pythainlp`, `pandas`, `pyyaml`, `webvtt-py`, `tqdm`, `gradio` (optional), `openai` client (only for calling local vLLM, optional).

## 8. Acceptance criteria
1. `--stage all --limit 5` runs end to end on 5 videos without manual steps (except HF token) and produces all three CSVs.
2. Re-running a stage does not re-download or re-process finished items.
3. No raw audio, unmasked text, or personal data in `data/final/`.
4. Unit tests for: PII masking regexes, ASR-diarization alignment, CSV schema validation.
5. `stats.md` is generated with class counts.
6. README explains setup (HF token, pyannote terms, ffmpeg), each stage, and the data/copyright policy.

## 9. Known limitations (write these in the README)
- Many clips are prank/"scam-baiting" calls, so the "victim" often knows it is a scam; behavior differs from real victims.
- News clips mix anchor narration with call audio; narrator turns must be removed.
- Phone-quality audio and background music reduce ASR accuracy; human correction is required.
- Legitimate-call examples are scarce on YouTube; the not-scam class will need to be filled with synthetic dialogues later.

## 10. Suggested build order
1. utils + config + search → check `videos.csv` by hand.
2. download + transcribe on 3 videos.
3. diarize + segment.
4. clean/mask + tests.
5. prelabel (rules first, LLM later) + review CSV.
6. export + stats.
7. Optional Gradio review app.
