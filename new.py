import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline

class ToxicNeutralizer:
    def __init__(self):
        print("Initializing Neural Models... (This may take a moment on first run)")
        
        # 1. Toxicity Classifier (Detection)
        self.det_model_name = "unitary/toxic-bert"
        self.det_tokenizer = AutoTokenizer.from_pretrained(self.det_model_name)
        self.det_model = AutoModelForSequenceClassification.from_pretrained(self.det_model_name)
        
        # 2. Detoxification Model (Rewriting)
        # Using a T5 model fine-tuned for detoxification tasks
        self.rewrite_model_name = "s-nlp/t5-paranmt-detox"
        self.rewrite_tokenizer = AutoTokenizer.from_pretrained(self.rewrite_model_name)
        self.rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(self.rewrite_model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det_model.to(self.device)
        self.rewrite_model.to(self.device)

    def get_toxicity_score(self, text):
        """Returns a confidence score between 0 and 1."""
        inputs = self.det_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.det_model(**inputs)
        
        # Using the 'toxic' label index (usually 0 in this model)
        probs = torch.sigmoid(outputs.logits)
        return torch.max(probs).item()

    def neutralize(self, text, threshold=0.5):
        """Main pipeline: Detect -> Decide -> Rewrite."""
        score = self.get_toxicity_score(text)
        
        if score < threshold:
            return {
                "original": text,
                "neutralized": text,
                "is_toxic": False,
                "confidence": round(score, 4)
            }
        
        # Generation Logic
        inputs = self.rewrite_tokenizer(f"detoxify: {text}", return_tensors="pt", max_length=128, truncation=True).to(self.device)
        
        # Beam search for higher quality natural language
        outputs = self.rewrite_model.generate(
            inputs["input_ids"], 
            max_length=128, 
            num_beams=5, 
            early_stopping=True
        )
        
        # Removes the prompt prefix and any weird casing/spacing
        neutral_text = self.rewrite_tokenizer.decode(outputs[0], skip_special_tokens=True)
        neutral_text = neutral_text.replace("detoxify:", "").replace("Detoxify:", "").strip()
        
        return {
            "original": text,
            "neutralized": neutral_text,
            "is_toxic": True,
            "confidence": round(score, 4)
        }

# --- Execution & Testing ---
if __name__ == "__main__":
    engine = ToxicNeutralizer()
    
    test_sentences = [
        "You are so incredibly stupid, I can't believe you even graduated.",
        "Shut up and do your job properly for once!",
        "The weather is quite nice in Navi Mumbai today.", # Non-toxic test
        "This project is a total disaster and you should be ashamed."
    ]
    
    print("\n" + "="*50)
    print("TOXIC SPEECH NEUTRALIZER RESULTS")
    print("="*50)
    
    for text in test_sentences:
        res = engine.neutralize(text)
        status = "[TOXIC]" if res['is_toxic'] else "[CLEAN]"
        print(f"\n{status} Score: {res['confidence']}")
        print(f"Input:  {res['original']}")
        print(f"Output: {res['neutralized']}")