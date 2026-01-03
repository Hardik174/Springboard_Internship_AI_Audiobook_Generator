# 📖 AI Audiobook Generator

An end-to-end AI-powered pipeline that converts written books into **engaging, natural-sounding audiobooks**.  
The system extracts text, enriches it for narration using an AI language model, and then generates expressive speech using a Text-to-Speech (TTS) engine.

---

## 🚀 Features

- Automatic text extraction from book files  
- Chunk-wise processing for large documents  
- Context-aware narration enrichment using AI  
- **No summarization or hallucination — meaning is preserved**
- Natural audiobook-style narration  
- High-quality AI speech synthesis  
- Modular design — easy to extend and customize  
- Works across any content domain  

---

## 🧠 How It Works

1. **Text Extraction** — raw text is extracted from the input book  
2. **Smart Chunking** — content is split safely for LLM processing  
3. **Narration Enrichment** — AI rewrites text in audiobook-style  
4. **Context Carryover** — continuity maintained across chunks  
5. **Text-to-Speech Conversion** — expressive speech is generated  
6. **Audiobook Assembly** — audio chunks are merged seamlessly  

---

## 🏗️ Tech Stack

- Python  
- LLM (Mistral / Llama / Local LM Studio etc.)
- TTS Engine (Sarvam AI / others)
- Pydub + FFmpeg for audio processing

---

## 📂 Project Structure

AI-Audiobook-Generator/
│
├── Text_Extraction_Module.py
├── audiobook_api.py
├── TTS_Module.py
├── enriched_text.md
├── extracted_text.md
├── output_audio/
│ ├── chunk_1.wav
│ ├── chunk_2.wav
│ └── final_audiobook.wav
│
└── README.md


---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
git clone https://github.com/your-username/ai-audiobook-generator.git
cd ai-audiobook-generator

### 2️⃣ Create and activate environment
conda create -n audiobook python=3.10
conda activate audiobook

### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Install FFmpeg

Windows: download from ffmpeg.org and add to PATH

Mac:
brew install ffmpeg

Linux:
sudo apt install ffmpeg

### ▶️ Usage
Step 1 — Extract text
from Text_Extraction_Module import extract_text
extract_text("book.pdf", "extracted_text.md")

Step 2 — Generate enriched narration
python audiobook_api.py

Step 3 — Convert text to speech & merge
python TTS_Module.py


### 🎧 Final audiobook saved as:

output_audio/final_audiobook.wav

### 📌 Key Design Principles

No summarization

No hallucinations

Meaning preserved

Natural narration flow

Ready for long-form listening

### 🔍 Use Cases

Audiobook creation

Accessibility support

E-learning

Research narration

Long-form article listening

### 🧪 Future Enhancements

Voice cloning

Multi-speaker support

Multi-language TTS

Background ambience

Web dashboard

Cloud deployment

RAG-based continuity


### 📜 License

This project is licensed under the MIT License.

