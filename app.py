import streamlit as st
import pickle

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

    # Prediction
    vector = tfidf.transform([text])
    prediction = model.predict(vector)[0]
    emotion = emotion_map[prediction]

    # Emojis
    emoji_map = {
        "joy": "😊",
        "sadness": "😢",
        "anger": "😠",
        "love": "❤️",
        "fear": "😨",
        "surprise": "😲"
    }

    emoji = emoji_map.get(emotion.lower(), "✨")

    # Results
    st.divider()

    st.success("Analysis Completed")

    st.metric(
        label="Detected Emotion",
        value=f"{emoji} {emotion.upper()}"
    )

    # Optional Description
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