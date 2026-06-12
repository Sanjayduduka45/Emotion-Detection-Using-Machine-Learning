import streamlit as st
import pickle
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="EmotionScope",
    page_icon="💜",
    layout="centered"
)

# =========================
# LOAD FILES
# =========================

model = pickle.load(open("xgb_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
emotion_map = pickle.load(open("emotions.pkl", "rb"))

# =========================
# TEXT PREPROCESSING
# =========================

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text)

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# =========================
# HEADER
# =========================

st.title("💜 EmotionScope")
st.subheader("Reveal the feelings behind every word")

st.write(
    "Analyze text and identify the underlying emotion."
)

# =========================
# INPUT
# =========================

text = st.text_area(
    "Enter Text",
    height=200,
    placeholder="Type your text here..."
)

# =========================
# BUTTON
# =========================

if st.button("🔍 Analyze Emotion", use_container_width=True):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    # PREPROCESS TEXT
    clean_text = preprocess_text(text)

    # TF-IDF
    vector = tfidf.transform([clean_text])

    # PREDICTION
    prediction = model.predict(vector)[0]

    emotion = emotion_map[prediction]

    emoji_map = {
        "joy": "😊",
        "sadness": "😢",
        "anger": "😠",
        "love": "❤️",
        "fear": "😨",
        "surprise": "😲"
    }

    emoji = emoji_map.get(emotion.lower(), "✨")

    st.divider()

    st.success("Analysis Completed")

    st.metric(
        label="Detected Emotion",
        value=f"{emoji} {emotion.upper()}"
    )

    descriptions = {
        "joy": "This text expresses happiness and positivity.",
        "sadness": "This text reflects sadness or disappointment.",
        "anger": "This text contains anger or frustration.",
        "love": "This text expresses affection and care.",
        "fear": "This text indicates fear or anxiety.",
        "surprise": "This text shows surprise or shock."
    }

    st.info(
        descriptions.get(
            emotion.lower(),
            "Emotion detected successfully."
        )
    )

    st.subheader("Submitted Text")
    st.write(text)

    # DEBUG (remove later)
    st.write("Processed Text:", clean_text)
    st.write("Prediction ID:", prediction)