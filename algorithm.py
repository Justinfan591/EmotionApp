#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified to enhance efficiency, integrate with UI, and adjust for AWS deployment.
"""

import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import nltk
import matplotlib.pyplot as plt
from scipy.special import expit
import gc
from tqdm import tqdm

# Ensure NLTK data is downloaded
nltk.data.path.append('./nltk_data/')
nltk.download('punkt', download_dir='./nltk_data/', quiet=True)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Algorithm:
    def __init__(self):
        # Initialize the tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.model = RobertaForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.model.to(device)  # Move model to GPU if available
        self.labels = self.model.config.id2label

    def split_text_into_sentences(self, text):
        # Use NLTK to split text into sentences
        sentences = nltk.tokenize.sent_tokenize(text)
        return sentences

    def score_emotions(self, text_list):
        scores_list = []
        batch_size = 16  # Adjust based on your hardware capabilities
        total_batches = (len(text_list) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(text_list), batch_size), desc="Scoring emotions", leave=False):
            batch_texts = text_list[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = inputs.to(device)  # Move inputs to GPU
            with torch.no_grad():
                logits = self.model(**inputs).logits
                scores = expit(logits.cpu().numpy())
            for sample_scores in scores:
                emotion_scores = {self.labels[idx]: score for idx, score in enumerate(sample_scores)}
                scores_list.append(emotion_scores)
            # Clear variables to free memory
            del inputs, logits, scores
            torch.cuda.empty_cache()
            gc.collect()

        return scores_list

    def plot_emotion_graph(self, text, progress_callback=None):
        # Define emotions of interest
        emotions_of_interest = ['surprise', 'anger', 'sadness', 'fear', 'joy']
        # Split text into sentences
        sentences = self.split_text_into_sentences(text)
        num_sentences = len(sentences)
        # Score emotions with progress callback
        emotion_scores = []
        batch_size = 16

        for i in range(0, num_sentences, batch_size):
            batch_sentences = sentences[i:i+batch_size]
            batch_scores = self.score_emotions(batch_sentences)
            emotion_scores.extend(batch_scores)
            if progress_callback:
                progress = min((i + batch_size) / num_sentences, 1.0)
                progress_callback(progress)

        # Prepare data for plotting
        emotion_trends = {emotion: [] for emotion in emotions_of_interest}
        for scores in emotion_scores:
            for emotion in emotions_of_interest:
                emotion_trends[emotion].append(scores.get(emotion, 0))

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        x_values = range(len(emotion_scores))
        for emotion, scores in emotion_trends.items():
            ax.plot(x_values, scores, label=emotion)
        ax.set_xlabel('Sentence Index')
        ax.set_ylabel('Emotion Score')
        ax.set_title('Emotion Analysis Over Text')
        ax.legend()
        plt.tight_layout()
        return fig
