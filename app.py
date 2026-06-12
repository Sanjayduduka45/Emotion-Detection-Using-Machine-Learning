import streamlit as st
import pickle
import re
import string
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# PAGE CONFIGURATION


st.set_page_config(
    page_title="EmotionScope",
    page_icon="💜",
    layout="centered"
)


# LOAD FILES


try:
    model = pickle.load(open("xgb_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    emotion_map = pickle.load(open("emotions.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# PREPROCESSING


stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    # Stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# UI


st.title("💜 EmotionScope")
st.subheader("Reveal the feelings behind every word")

st.write(
    "Analyze text and identify the underlying emotion."
)

text = st.text_area(
    "Enter Text",
    height=200,
    placeholder="Type your text here..."
)


# EMOJI MAP


emoji_map = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "love": "❤️",
    "fear": "😨",
    "surprise": "😲"
}


# DESCRIPTION MAP


descriptions = {
    "joy": "This text expresses happiness and positivity.",
    "sadness": "This text reflects sadness or disappointment.",
    "anger": "This text contains anger or frustration.",
    "love": "This text expresses affection and care.",
    "fear": "This text indicates fear or anxiety.",
    "surprise": "This text shows surprise or shock."
}


# PREDICTION


if st.button("🔍 Analyze Emotion", use_container_width=True):

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    try:
        # Same preprocessing used during training
        clean_text = preprocess_text(text)

        # TF-IDF Transformation
        vector = tfidf.transform([clean_text])

        # Prediction
        prediction = model.predict(vector)[0]

        # Convert label to emotion
        emotion = emotion_map[prediction]

        emoji = emoji_map.get(emotion.lower(), "✨")

        st.divider()

        st.success("Analysis Completed")

        st.metric(
            label="Detected Emotion",
            value=f"{emoji} {emotion.upper()}"
        )

        st.info(
            descriptions.get(
                emotion.lower(),
                "Emotion detected successfully."
            )
        )

        st.subheader("Submitted Text")
        st.write(text)

        # DEBUG SECTION
        with st.expander("Debug Info"):
            st.write("Processed Text:", clean_text)
            st.write("Prediction ID:", prediction)

    except Exception as e:
        st.error(f"Prediction Error: {e}")