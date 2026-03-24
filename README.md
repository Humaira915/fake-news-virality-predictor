# 📰 Fake News & Viral Predictor

This project is a simple machine learning application that analyzes news articles and predicts:

* whether the news is **Fake or Real**
* how likely it is to go **viral**

It’s built as a hands-on NLP project to understand how text data can be transformed into meaningful predictions.


# What this project does

✔ classify the news as Fake or Real
✔ generate a virality score (0–100%)
✔ show a features graph explaining the result


# How it will work (in simple terms)

* The text is cleaned (lowercase, punctuation removed, etc.)
* Both **title + content are combined** for better context
* Text is converted into numbers using TF-IDF
* A Logistic Regression model is used for prediction
* Additional features like:

  * sentiment
  * number of exclamation marks
  * emotional words
  * title length are also used



# Why this project is interesting

* adding a "virality prediction layer"
* including "feature-based explainability"
* combining "text + behavioral signals (emotion, tone)"


# Tech Stack

* Python
* Scikit-learn
* NLP (TF-IDF, TextBlob)
* Streamlit (for UI)
* Pandas, NumPy


# Limitations

* The model relies on patterns in text, not deep understanding
* It may struggle with:

  * sarcasm
  * partially true news
  * unseen writing styles
 * Virality score is heuristic-based, not learned from real engagement data


# Future Improvements

* Use advanced NLP models (like transformers)
* Improve dataset quality and diversity
* Add source credibility analysis
* Deploy the app online


# Final Note

This project was built as part of my learning journey in machine learning and NLP.
It focuses not just on prediction, but also on understanding "why"a model makes a decision.


