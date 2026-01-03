📖 AI Audiobook Generator

An end-to-end AI-powered pipeline that converts written books into engaging, natural-sounding audiobooks.
The system extracts text, enriches it for narration using an AI language model, and then generates expressive speech using a Text-to-Speech (TTS) engine.

🚀 Features

✔ Automatic text extraction from book files
✔ Chunk-wise processing for large documents
✔ Context-aware narration enrichment using AI
✔ No summarization or hallucination — meaning is preserved
✔ Natural audiobook-style narration
✔ High-quality AI speech synthesis
✔ Modular design — easy to extend and customize
✔ Works across any content domain

🧠 How It Works

1️⃣ Text Extraction
Raw text is extracted from the input book/document.

2️⃣ Smart Chunking
The content is split into model-friendly chunks while preserving sentence structure.

3️⃣ Narration Enrichment via LLM
Each chunk is refined into audiobook-friendly narration while retaining meaning.

4️⃣ Context Carryover
A small excerpt from previous narration is passed forward to maintain flow.

5️⃣ Text-to-Speech Conversion
The enriched text is converted into expressive speech.

6️⃣ Audiobook Assembly
All audio chunks are merged into one seamless audiobook file.

🏗️ Tech Stack

Python

LLM (e.g., Mistral / Llama / LM Studio)

TTS Engine (e.g., Sarvam AI, Coqui, etc.)

Audio Processing — Pydub / FFmpeg

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
1️⃣ Clone the repository
git clone https://github.com/your-username/ai-audiobook-generator.git
cd ai-audiobook-generator

2️⃣ Create and activate environment
conda create -n audiobook python=3.10
conda activate audiobook

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Install FFmpeg

Windows: Download from https://ffmpeg.org
 and add to PATH

Mac:

brew install ffmpeg


Linux:

sudo apt install ffmpeg

▶️ Usage
Step 1 — Extract Text
from Text_Extraction_Module import extract_text
extract_text("book.pdf", "extracted_text.md")

Step 2 — Generate Enriched Narration
python audiobook_api.py

Step 3 — Convert Text to Speech & Merge
python TTS_Module.py


🎧 Final audiobook saved as:

output_audio/final_audiobook.wav

📌 Key Design Principles

✔ No summarization
✔ No hallucinations
✔ Meaning preserved
✔ Natural narration flow
✔ Ready for long-form listening

🔍 Use Cases

🎙 Audiobook creation
📚 Accessibility support
🏫 E-learning
🎓 Research narration
📜 Long-form article listening

🧪 Future Enhancements

🔹 Voice cloning
🔹 Multi-speaker support
🔹 Multi-language TTS
🔹 Background ambience/music
🔹 Web dashboard
🔹 Cloud deployment
🔹 RAG-assisted narration continuity

📜 License

This project is licensed under the MIT License.
