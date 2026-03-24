import streamlit as st
import pickle
import numpy as np
from textblob import TextBlob
from scipy.sparse import hstack
import matplotlib.pyplot as plt


import string


model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

#creating dict
emotional_words = ["shocking","breaking","unbelievable","amazing","secret","scandal","danger"]

# Clean text
def clean_text(text):

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


# Count emotional words
def count_emotional_words(text):

    words = text.lower().split()
    count = 0

    for word in words:
        if word in emotional_words:
            count += 1

    return count


# Virality score cal
def virality_score(title):

    sentiment = TextBlob(title).sentiment.polarity
    exclamation = title.count("!")
    emotional = count_emotional_words(title)
    title_len = len(title)

    score = (
        0.3 * abs(sentiment) +
        0.3 * exclamation +
        0.2 * emotional +
        0.2 * (title_len/100)
    )

    return min(score * 100, 100)


# UI
st.title("📰 Fake News Virality Predictor")

st.write("Enter a news title and content to detect fake news and virality risk.")

title = st.text_input("News Title")

text = st.text_area("News Content")

if st.button("Analyze News"):

    combined = clean_text(title + " " + text)
    text_vec = vectorizer.transform([combined])


    title_len = len(title)
    exclamation = title.count("!")
    sentiment = TextBlob(title).sentiment.polarity
    emotional = count_emotional_words(title)

    features = np.array([[title_len, exclamation, sentiment, emotional]]) * 0.3


    X_input = hstack((text_vec, features))

    prediction = model.predict(X_input)[0]

    if prediction == 0:
        st.error("⚠️ Fake News Detected")
    else:
        st.success("✅ Real News")

    score = virality_score(title)

    st.subheader("Virality Risk Score")

    st.progress(int(score))

    st.write("Virality Score:", round(score,2), "%")

    st.subheader("📊 Why this news may go viral")


    sent_score = abs(sentiment) * 10
    len_score = title_len / 10

    feature_names = ["Emotion", "Excitement (!)", "Sentiment", "Length"]
    feature_values = [emotional, exclamation, sent_score, len_score]

    fig, ax = plt.subplots()

    ax.barh(feature_names, feature_values, color="green")

    ax.set_title("Virality Factors Contribution")

    st.pyplot(fig)

    



