# 🎬 Review Sentiment Analyzer

Try Review Sentiment Analyzer: https://review-sentiment-analysis-mqgnyyqmzyhuuz6wyhkafb.streamlit.app/

A Streamlit web app that classifies a movie review as **Positive** or **Negative** using a TF-IDF + Logistic Regression model.

## Features
- Analyze a single review with a confidence score and verdict
- Highlights the words that pushed the prediction up or down
- Batch mode: upload a CSV of reviews, view results and download predictions
- Adjustable positive threshold and a history of past predictions

## Tech Stack
Python · Streamlit · scikit-learn · NLTK · pandas · joblib

## How It Works
Text is cleaned and stemmed (`wordopt`), converted to TF-IDF features, and classified by Logistic Regression (`0` = Negative, `1` = Positive).
