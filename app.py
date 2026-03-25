import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

MAX_LENGTH = 256
BATCH_SIZE = 16


class ReviewClassifier:
    def __init__(
        self,
        sentiment_model_dir: str = "saved_models/sentiment",
        tag_model_dir: str = "saved_models/tag",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sentiment_tokenizer = DistilBertTokenizerFast.from_pretrained(sentiment_model_dir)
        self.tag_tokenizer = DistilBertTokenizerFast.from_pretrained(tag_model_dir)

        self.sentiment_model = DistilBertForSequenceClassification.from_pretrained(
            sentiment_model_dir
        ).to(self.device)
        self.tag_model = DistilBertForSequenceClassification.from_pretrained(
            tag_model_dir
        ).to(self.device)

        self.sentiment_model.eval()
        self.tag_model.eval()

        with open(Path(sentiment_model_dir) / "id2label.json", "r") as f:
            self.id2sentiment = {int(k): v for k, v in json.load(f).items()}

        with open(Path(tag_model_dir) / "id2label.json", "r") as f:
            self.id2tag = {int(k): v for k, v in json.load(f).items()}

    @staticmethod
    def _combine_text(title: pd.Series, text: pd.Series) -> pd.Series:
        combined = (
            title.fillna("").astype(str).str.strip()
            + ". "
            + text.fillna("").astype(str).str.strip()
        ).str.strip()
        combined = combined.str.replace(r"^\.\s*", "", regex=True)
        return combined

    def _predict_batch(self, texts: list[str]):
        sentiment_labels, sentiment_scores = [], []
        tag_labels, tag_scores = [], []

        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]

            sentiment_inputs = self.sentiment_tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            sentiment_inputs = {k: v.to(self.device) for k, v in sentiment_inputs.items()}

            tag_inputs = self.tag_tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            tag_inputs = {k: v.to(self.device) for k, v in tag_inputs.items()}

            with torch.no_grad():
                sentiment_outputs = self.sentiment_model(**sentiment_inputs)
                tag_outputs = self.tag_model(**tag_inputs)

                sentiment_probs = torch.softmax(sentiment_outputs.logits, dim=1).cpu().numpy()
                tag_probs = torch.softmax(tag_outputs.logits, dim=1).cpu().numpy()

            sentiment_ids = np.argmax(sentiment_probs, axis=1)
            tag_ids = np.argmax(tag_probs, axis=1)

            sentiment_labels.extend([self.id2sentiment[int(idx)] for idx in sentiment_ids])
            sentiment_scores.extend(
                [float(sentiment_probs[j, idx]) for j, idx in enumerate(sentiment_ids)]
            )

            tag_labels.extend([self.id2tag[int(idx)] for idx in tag_ids])
            tag_scores.extend([float(tag_probs[j, idx]) for j, idx in enumerate(tag_ids)])

        return sentiment_labels, sentiment_scores, tag_labels, tag_scores

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        title_col: str = "review_title",
        text_col: str = "review_text",
    ) -> pd.DataFrame:
        if title_col not in df.columns or text_col not in df.columns:
            raise ValueError(
                f"Input CSV must contain '{title_col}' and '{text_col}' columns."
            )

        output_df = df.copy()
        output_df["combined_text"] = self._combine_text(output_df[title_col], output_df[text_col])
        output_df = output_df[output_df["combined_text"].str.strip() != ""].copy()

        texts = output_df["combined_text"].tolist()
        sent_labels, sent_scores, tag_labels, tag_scores = self._predict_batch(texts)

        output_df["predicted_sentiment"] = sent_labels
        output_df["sentiment_confidence"] = sent_scores
        output_df["predicted_tag"] = tag_labels
        output_df["tag_confidence"] = tag_scores

        return output_df


st.set_page_config(page_title="Trustpilot Review Classifier", layout="wide")
st.title("Trustpilot Review Classifier")
st.write("Upload a CSV file to generate sentiment and tag predictions.")

@st.cache_resource
def load_classifier():
    return ReviewClassifier()

classifier = load_classifier()

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
title_col = st.text_input("Title column name", value="review_title")
text_col = st.text_input("Text column name", value="review_text")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Input Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run Prediction"):
            with st.spinner("Generating predictions..."):
                result_df = classifier.predict_dataframe(
                    df,
                    title_col=title_col,
                    text_col=text_col,
                )

            st.subheader("Prediction Output")
            st.dataframe(result_df.head(20), use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download output CSV",
                data=csv_data,
                file_name="reviews_with_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error: {e}")
