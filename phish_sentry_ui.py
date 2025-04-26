import streamlit as st
# FIRST call
st.set_page_config(page_title="PhishSentry Demo",
                   page_icon="🛡️",
                   layout="centered")

import re, joblib
from pathlib import Path

# ── same cleaner used at training time ──────────────────
URL_PATTERN   = re.compile(r"http\S+")
PUNCT_PATTERN = re.compile(r"[^\w\s]")
SPACE_PATTERN = re.compile(r"\s+")

def preprocess_text(text: str) -> str:
    text = URL_PATTERN.sub("", text)
    text = PUNCT_PATTERN.sub(" ", text)
    text = text.lower()
    return SPACE_PATTERN.sub(" ", text).strip()
# --------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "artifacts" / "model.joblib"

@st.cache_resource(show_spinner="Loading model…")
def load_pipeline():
    return joblib.load(MODEL_PATH)

model = load_pipeline()

LABELS = {0: "Safe Email ✅", 1: "Phishing Email 🚨"}

# ── UI ──────────────────────────────────────────────────
st.title("🛡️ PhishSentry — Email Phishing Detector")
st.markdown(
    "Paste an email (subject + body) below and click **Scan email**.\n\n"
    "Prediction threshold is **0.50** by default."
)

email_text = st.text_area("Email content", height=300)

if st.button("Scan email"):
    if not email_text.strip():
        st.warning("Please paste some email text first.")
    else:
        prob  = float(model.predict_proba([email_text])[0, 1])
        label = int(prob >= 0.5)
        st.header(LABELS[label])
        st.progress(prob, text=f"Confidence: {prob*100:.1f}%")

        tokens = re.findall(
            r"(https?://\S+|verify|suspend|login|account|password)",
            email_text, flags=re.I
        )
        if tokens:
            st.subheader("⚡ Notable tokens")
            st.write(", ".join(dict.fromkeys(tokens)))
