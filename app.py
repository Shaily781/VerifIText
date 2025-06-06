import streamlit as st
import pickle
import pandas as pd
from textstat import flesch_reading_ease
import string
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load saved model and feature columns
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Set page config
st.set_page_config(page_title="VerifIText – AI vs Human Detector", page_icon="🧠")

# ----------------------------
# 🎯 HEADER SECTION
# ----------------------------
st.markdown("<h1 style='text-align: center;'>🧠 VerifIText</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>AI vs Human Content Classifier</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic; color: #6c757d;'>I Detect. I Defend. I Deliver Clarity.</p>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# 📩 TEXT INPUT SECTION
# ----------------------------
st.subheader("🔍 Paste Your Text Below")

user_input = st.text_area("Enter text here:", height=200, placeholder="Type or paste your content...")

st.markdown("")

# ----------------------------
# 🧠 FEATURE EXTRACTION
# ----------------------------
def extract_features(text):
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha]
    word_count = len(words)
    sentence_count = len(list(doc.sents))
    avg_word_length = sum(len(token) for token in words) / word_count if word_count > 0 else 0
    lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0
    readability_score = flesch_reading_ease(text)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "lexical_diversity": lexical_diversity,
        "readability_score": readability_score
    }

# ----------------------------
# 🚀 PREDICT BUTTON
# ----------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔎 Classify Text", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            features = extract_features(user_input)
            X_new = pd.DataFrame([features])[feature_columns]
            prediction = model.predict(X_new)[0]
            label = "🧠 AI-Generated" if prediction == 0 else "🧍‍♂️ Human-Written"

            st.markdown("---")
            st.success(f"**Prediction:** {label}")
            st.markdown("")

            with st.expander("🧬 View Extracted Features", expanded=True):
                feature_df = pd.DataFrame([features])
                st.dataframe(feature_df.style.format(precision=3),use_container_width=True)

# ----------------------------
# 📌 FOOTER
# ----------------------------
st.markdown("---")
st.markdown("""
<p style='text-align: center; font-size: 13px; color: gray;'>
Built with ❤️ by Shaily Soni · <a href='https://www.linkedin.com/in/shaily-soni-a658481b4/' target='_blank'>LinkedIn</a>
</p>
""", unsafe_allow_html=True)
