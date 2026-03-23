🛡️ Toxic Speech Neutralizer
An AI-powered system that doesn't just block hate speech—it rephrases it. This project uses a hybrid transformer pipeline to detect toxicity and suggest polite, socially acceptable alternatives while preserving the original intent
Core Features
Real-time Detection: High-accuracy toxicity scoring using Toxic-BERT.

Contextual Rewriting: Semantic-aware detoxification using T5-Paranmt.

Modern UI: A sleek web interface built with Streamlit.

Hardware Optimized: Native support for NVIDIA GPUs (RTX Series) via CUDA.

🛠️ Tech Stack
Language: Python 3.11

AI Frameworks: PyTorch, Hugging Face Transformers

Models: * unitary/toxic-bert (Classification)

s-nlp/t5-paranmt-detox (Seq2Seq Generation)

Frontend: Streamlit

📦 Installation & Setup
1. Clone the Repository
Bash
git clone https://github.com/YOUR_USERNAME/Toxic-Speech-Neutralizer.git
cd Toxic-Speech-Neutralizer
2. Create a Virtual Environment
PowerShell
python -m venv .venv
.\.venv\Scripts\activate
3. Install Dependencies
PowerShell
pip install -r requirements.txt
4. Run the Application
PowerShell
streamlit run app.py
🧪 How it Works (The Pipeline)
The system follows a three-stage neural pipeline:

Preprocessing: Cleans input and prepares tensors.

Detection (Encoder): Toxic-BERT analyzes the sentiment and intent. If the toxicity score exceeds 50%, the rewrite trigger is activated.

Neutralization (Decoder): A fine-tuned T5 model performs a beam search to generate a paraphrased version that lacks the original "toxicity" but maintains the core meaning.
