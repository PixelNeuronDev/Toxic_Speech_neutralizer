import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# Page Config
st.set_page_config(page_title="Toxic Speech Neutralizer", page_icon="🛡️")

@st.cache_resource
def load_models():
    # Using the same high-quality models we tested
    det_name = "unitary/toxic-bert"
    rew_name = "s-nlp/t5-paranmt-detox"
    
    det_tok = AutoTokenizer.from_pretrained(det_name)
    det_mod = AutoModelForSequenceClassification.from_pretrained(det_name)
    
    rew_tok = AutoTokenizer.from_pretrained(rew_name)
    rew_mod = AutoModelForSeq2SeqLM.from_pretrained(rew_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_mod.to(device)
    rew_mod.to(device)
    
    return det_tok, det_mod, rew_tok, rew_mod, device

st.title("🛡️ Toxic Speech Neutralizer")
st.markdown("Enter a sentence below to detect toxicity and generate a polite version.")

with st.spinner("Loading AI models... This might take a minute on your first run."):
    det_tok, det_mod, rew_tok, rew_mod, device = load_models()

user_input = st.text_area("User Input:", placeholder="Type something here...")

if st.button("Analyze & Neutralize"):
    if user_input.strip():
        # 1. Detection
        inputs = det_tok(user_input, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = det_mod(**inputs)
        score = torch.sigmoid(outputs.logits).max().item()
        
        # 2. Results UI
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Toxicity Score", f"{score:.2%}")
            if score > 0.5:
                st.error("Status: Toxic")
            else:
                st.success("Status: Clean")

        with col2:
            if score > 0.5:
                # 3. Rewriting
                rew_inputs = rew_tok(f"detoxify: {user_input}", return_tensors="pt", max_length=128, truncation=True).to(device)
                gen = rew_mod.generate(rew_inputs["input_ids"], max_length=128, num_beams=5)
                output = rew_tok.decode(gen[0], skip_special_tokens=True)
                
                # Cleanup "detoxify:" prefix if it exists
                clean_output = output.replace("detoxify:", "").replace("Detoxify:", "").strip()
                
                st.subheader("Neutralized Version:")
                st.info(clean_output)
            else:
                st.subheader("Neutralized Version:")
                st.write("No changes needed.")
    else:
        st.warning("Please enter some text first.")