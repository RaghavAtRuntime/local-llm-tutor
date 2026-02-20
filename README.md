# Local LLM Tutor

A **locally hosted AI tutor assistant** with voice interaction, adaptive feedback, and session tracking — no cloud APIs required.

## Features

- 🎤 **Voice I/O** — Speech-to-text via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and text-to-speech via [pyttsx3](https://github.com/nateshmbhat/pyttsx3)
- 🧠 **Multi-layer evaluation** — Exact match + semantic similarity (sentence-transformers) + key concept detection
- 🤖 **Optional LLM feedback** — Enhanced explanations via a local [Ollama](https://ollama.com/) model (disabled by default)
- 📊 **Session persistence** — SQLite-backed history and performance statistics
- 🔊 **Voice Activity Detection** — Auto-stop recording on silence
- ⌨️ **Text-only mode** — Run without any audio hardware

## Architecture

```
local-llm-tutor/
├── src/
│   └── local_llm_tutor/
│       ├── __init__.py           # Package init, exports Tutor
│       ├── tutor.py              # Main orchestrator loop
│       ├── quiz_engine.py        # Question bank loader & sequencer
│       ├── evaluation_engine.py  # Answer evaluation (exact/semantic/concept)
│       ├── feedback_generator.py # Adaptive feedback templates + LLM
│       ├── session_manager.py    # SQLite session & stats tracking
│       ├── llm_core.py           # Ollama LLM integration
│       ├── tts_module.py         # Text-to-speech (pyttsx3)
│       ├── stt_module.py         # Speech-to-text (faster-whisper)
│       └── audio_controller.py  # VAD recording & interrupt detection
├── tests/                        # pytest test suite (no hardware required)
├── data/
│   └── questions.json            # Question bank (12 questions across 3 topics)
├── config.yaml                   # All runtime configuration
├── main.py                       # CLI entry point
├── requirements.txt
└── setup.py
```

```
                        ┌─────────────────────────────────┐
                        │           Tutor (tutor.py)       │
                        │  orchestrates the session loop   │
                        └────┬──────┬──────┬──────┬────────┘
                             │      │      │      │
               ┌─────────────┘  ┌───┘  ┌───┘  ┌───┘
               ▼                ▼      ▼      ▼
        ┌────────────┐  ┌─────────┐ ┌──────┐ ┌───────────┐
        │ QuizEngine │  │Evaluator│ │  TTS │ │ SessionMgr│
        │(questions) │  │(scoring)│ │  STT │ │ (SQLite)  │
        └────────────┘  └────┬────┘ └──────┘ └───────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
           ┌──────────────┐   ┌──────────────┐
           │  Feedback    │   │   LLMCore    │
           │  Generator   │──▶│  (Ollama,    │
           │  (templates) │   │  optional)   │
           └──────────────┘   └──────────────┘
```

## Installation

**Prerequisites:** Python 3.9+

```bash
# Clone and enter the repo
git clone <repo-url>
cd local-llm-tutor

# Install (core + dev extras)
pip install -e ".[dev]"

# Or install all requirements directly
pip install -r requirements.txt
```

For audio support (microphone recording):
```bash
pip install -e ".[audio]"
# On Linux you may also need: sudo apt-get install portaudio19-dev
```

## Usage

### Text-only mode (no microphone needed)
```bash
python main.py --text-only
```

### Voice mode
```bash
python main.py
```

### Options
```
python main.py --help

  --config PATH           Config file (default: config.yaml)
  --questions PATH        Question bank JSON (default: data/questions.json)
  --mode {sequential,random,difficulty}
  --difficulty {easy,medium,hard}
  --text-only             Keyboard input/print output only
  -v, --verbose           Debug logging
```

### Examples
```bash
# Easy questions only, random order, text mode
python main.py --text-only --difficulty easy --mode random

# Medium ML questions with verbose logging
python main.py --text-only --difficulty medium --verbose
```

### Special voice commands
During a session you can say:
- **"repeat"** / **"say again"** — Re-read the current question
- **"explain"** / **"explain more"** — Get a detailed explanation (requires Ollama)
- **"skip"** / **"next"** — Skip the current question
- **"quit"** / **"exit"** — End the session early

## Configuration

Edit `config.yaml` to tune all settings:

```yaml
tts:
  rate: 150          # Speech rate (words per minute)
  volume: 1.0        # Volume 0.0–1.0
  voice_index: 0     # System voice index

stt:
  model: "base"      # Whisper model: tiny/base/small/medium/large
  language: "en"
  energy_threshold: 300   # Mic sensitivity for VAD

evaluation:
  semantic_threshold_correct: 0.75   # Min similarity for "correct"
  semantic_threshold_partial: 0.45   # Min similarity for "partial"

quiz:
  mode: "sequential"        # sequential | random | difficulty
  difficulty_filter: null   # easy | medium | hard | null (all)
  time_limit: null          # seconds per question, null = unlimited

llm:
  provider: "ollama"
  model: "llama3.2"
  base_url: "http://localhost:11434"
  enabled: false    # Set true if Ollama is running locally

session:
  db_path: "tutor_sessions.db"
```

### Enabling LLM feedback (optional)

1. Install [Ollama](https://ollama.com/) and pull a model:
   ```bash
   ollama pull llama3.2
   ```
2. Set `llm.enabled: true` in `config.yaml`

## Running Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ -v --cov=local_llm_tutor --cov-report=term-missing
```

Tests require **no audio hardware, no ML models, and no external services**.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB+ |
| CPU | Any x86-64 | Modern multi-core |
| GPU | Not required | Optional (speeds up Whisper/LLM) |
| Microphone | Not required (text-only mode) | Any USB/built-in mic |
| Storage | ~500 MB (Whisper base model) | 2 GB+ |

## Question Bank

The default `data/questions.json` includes 12 questions across three topics:

| Topic | Count | Difficulties |
|-------|-------|--------------|
| Python | 5 | easy, medium |
| ML Basics | 4 | medium, hard |
| Data Structures | 3 | easy, medium |

Add your own questions following the schema:
```json
{
  "question_id": 13,
  "question": "Your question here?",
  "expected_answer": "The expected answer text.",
  "key_concepts": ["concept1", "concept2"],
  "difficulty": "easy",
  "topic": "Your Topic"
}
```
