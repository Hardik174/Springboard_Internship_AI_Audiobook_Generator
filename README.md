📖 AI Audiobook Generator

An end-to-end AI-powered pipeline that converts written books into engaging, natural-sounding audiobooks.
The system extracts text, enriches it for narration using an AI language model, and then generates expressive speech using a Text-to-Speech (TTS) engine.

🚀 Features

✔ Automatic text extraction from book files
✔ Chunk-wise processing to handle large documents
✔ Context-aware narration enrichment using AI
✔ Strict meaning preservation — no summarization or hallucination
✔ Natural audiobook-style narration
✔ High-quality AI TTS synthesis
✔ Supports long-form content across domains
✔ Modular & extendable pipeline

🧠 How It Works

1️⃣ Text Extraction
Raw text is extracted from the input book/document.

2️⃣ Smart Chunking
The content is split into model-friendly chunks while preserving sentence structure.

3️⃣ Narration Enrichment via LLM
Each chunk is passed through an AI model that:

improves readability & flow

adds audiobook-style tone

preserves original meaning

4️⃣ Context Carryover
A small excerpt from previous chunks is retained so narration stays consistent.

5️⃣ Text-to-Speech Conversion
The enriched text is converted into human-like narration audio.

6️⃣ Final Audiobook Assembly
All generated audio files are merged into a single audiobook.

🏗️ Tech Stack

Python

LLM (e.g., Mistral / Llama / LM Studio deployment)

TTS Engine (e.g., Sarvam AI, Coqui, etc.)

Audio processing — Pydub / FFmpeg

📂 Project Structure
AI-Audiobook-Generator/
│
├── Text_Extraction_Module.py
├── audiobook_api.py
├── TTS_Module.py
├── enriched_text.md
├── extracted_text.md
├── output_audio/
│   ├── chunk_1.wav
│   ├── chunk_2.wav
│   └── final_audiobook.wav
│
└── README.md

⚙️ Setup & Installation
1️⃣ Clone the repo
git clone https://github.com/your-username/ai-audiobook-generator.git
cd ai-audiobook-generator

2️⃣ Create & activate environment
conda create -n audiobook python=3.10
conda activate audiobook

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Install FFmpeg (required for audio)

Windows → download from ffmpeg.org & add to PATH

Mac → brew install ffmpeg

Linux → sudo apt install ffmpeg

▶️ Usage
Step 1 — Extract Text
from Text_Extraction_Module import extract_text
extract_text("book.pdf", "extracted_text.md")

Step 2 — Generate Enriched Narration
python audiobook_api.py

Step 3 — Convert to Speech & Merge
python TTS_Module.py


🎧 Final audiobook saved as:

output_audio/final_audiobook.wav

📌 Key Design Principles

✔ Do not summarize
✔ Preserve original meaning
✔ Maintain storytelling flow
✔ Keep narration enjoyable & natural
✔ Support long-form listening

🔍 Example Use Cases

🎙 Audiobook creation
📚 Accessibility for visually-impaired users
🏫 Education & e-learning
🎓 Research papers to voice
📜 Long-form articles & documentation

🧪 Future Enhancements

🔹 Speaker selection & voice cloning
🔹 Multi-language support
🔹 Background music & soundscapes
🔹 UI dashboard
🔹 Cloud deployment pipeline
🔹 RAG-based context enhancement

🤝 Contributions

Pull requests are welcome!
If you’d like to collaborate, improve code, or add features — feel free to contribute.

📜 License

MIT License — free to use & modify.
