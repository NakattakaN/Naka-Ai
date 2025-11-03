# Naka-chan: The Unhinged AI Interface

This project is a modular Python interface for running a local, quantized LLM as an "AI-friend" or VTuber persona named **Naka**. It is designed to be resource-efficient (running a 4-bit Llama 3.2 3B model on an RTX 3050) and features multiple input/output modalities, including Discord, local voice, and screen-reading capabilities.

The core persona is "Naka," a sharp-tongued, sarcastic, and unhinged AI VTuber.

## 🚀 Features

* **Efficient LLM:** Runs a 4-bit quantized Llama 3.2 3B model (or any compatible base model) using `transformers`, `bitsandbytes`, and `accelerate`.
* **LoRA Support:** Easily loads PEFT LoRA adapters to customize the base model's persona.
* **Dual Interface:**
    * **Discord:** Runs as a Discord bot that responds to mentions.
    * **Local Voice:** Runs as a local-only assistant with push-to-talk (PTT) voice input.
* **Persistent Memory:**
    * **Short-Term:** Standard conversation history per channel.
    * **Long-Term:** A persistent vector memory using **FAISS** for similarity search and **SQLite** for metadata storage, allowing Naka to recall past conversations.
* **Multimodal Input:**
    * **Speech-to-Text (`na-stt`):** Uses `faster-whisper` for local, PTT voice recognition.
    * **Vision (`na-vision`):** Can capture the user's screen, generate a text description (`SmolVLM2`), and feed it to the LLM as context.
* **Voice Output (`na-tts`):** Uses the `kokoro` engine to generate spoken responses.

## 🛠️ Architecture & Core Components

This project is split into several key parts:

* **`dc_llama.py` (Main Code):** The primary script that boots the model, connects to Discord, and manages all I/O loops.
* **`na_stt.py` (`FastWhisperRecognizer`):** Handles push-to-talk voice input using `sounddevice` and `faster-whisper`.
* **`na_tts.py` (`TTTS`):** Handles generating audio from the LLM's text responses.
* **`na_vision.py` (`SmolVLM2`):** A 4-bit quantized vision model that captions screen captures taken with `mss` and `cv2`.

### Key Dependencies

* **LLM:** `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`
* **Vector Memory:** `faiss-cpu` (or `faiss-gpu`), `sentence-transformers`, `sqlite3`
* **Interface:** `discord.py`
* **I/O:** `faster-whisper`, `keyboard`, `sounddevice`, `mss`, `opencv-python`

---

## ⚙️ Setup & Installation

### 1. Prerequisites

* Python 3.9+
* An **NVIDIA GPU** (e.g., RTX 3050 or better)
* **CUDA** correctly installed and compatible with your PyTorch version.

### 2. Installation

1.  Clone this repository:
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```
2.  Install the required Python packages. (A `requirements.txt` file is recommended, but here are the essentials).
    ```bash
    pip install torch transformers peft bitsandbytes accelerate
    pip install sentence-transformers faiss-cpu # or faiss-gpu
    pip install discord.py faster-whisper keyboard sounddevice mss opencv-python-headless
    ```

### 3. Configuration

Before running, you **must** update the hardcoded paths and tokens in `dc_llama.py`:

```python
# === Discord Token ===
DISCORD_TOKEN = "xxxxxx"  # <-- PUT YOUR DISCORD BOT TOKEN HERE

# === Model Paths ===
BASE_MODEL = r"base model path"  # <-- Path to your Llama 3.2 3B model
LORA_DIR = r"lora path"           # <-- Path to your PEFT LoRA adapter

# === Database Config ===
DB_PATH = r"C:\Users\atoca\Desktop\Naka-chan\local model\Naka-Brain\Memories\memo.db" # <-- Path to your SQLite DB
FAISS_INDEX = "memory_index.faiss" # <-- Path for your FAISS index file
MODEL_NAME = r"C:\Users\atoca\Desktop\Naka-chan\local model\Naka-Brain\Memories\hipocam" # <-- Path to SentenceTransformer model
