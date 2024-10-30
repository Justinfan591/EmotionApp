#!/usr/bin/env python3
"""
Updated to reflect the original algorithm, processing text slices and plotting specified emotions.
"""

import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re
import matplotlib.pyplot as plt
from scipy.special import expit
import gc
from tqdm import tqdm

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Algorithm:
    def __init__(self):
        # Initialize the tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.model = RobertaForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.model.to(device)  # Move model to GPU if available
        self.labels = self.model.config.id2label
        # Define emotions of interest
        self.emotions_of_interest = ['sadness', 'anger', 'fear', 'joy']  # Excluding 'surprise'

    def split_text_into_chunks(self, text_slice):
        # Remove HTML tags and split text into sentences
        sentences = re.sub('<[^>]+>', '', str(text_slice)).split('. ')
        return sentences

    def score_emotions(self, text_list):
        return_list = []
        for text in text_list:
            try:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                inputs = inputs.to(device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    logits = logits.cpu().numpy()[0]
                    scores = expit(logits)
                emotion_scores = [{'score': scores[i], 'label': self.labels[i]} for i in range(len(scores))]
            except Exception as e:
                emotion_scores = [{'score': np.nan, 'label': self.labels[i]} for i in range(len(self.labels))]
            return_list.append(emotion_scores)
            # Clear variables to free memory
            del inputs, logits, scores
            torch.cuda.empty_cache()
            gc.collect()
        return return_list

    def plot_emotion_graph(self, text, progress_callback=None):
        emots = []

        text_length = len(text)
        num_percentiles = 100

        for percentile in tqdm(range(num_percentiles), desc="Processing text slices", leave=False):
            # Calculate start and end indices for the text slice
            bot = (text_length * percentile) // num_percentiles
            top = bot + 10000  # Fixed slice length of 10,000 characters

            text_slice = text[bot:top]

            # Split the text slice into chunks (sentences)
            chunks = self.split_text_into_chunks(text_slice)

            # Score emotions for the chunks
            chunks_scored = self.score_emotions(chunks)

            # Aggregate emotion scores
            this_chunk = {}
            for emotion in self.emotions_of_interest:
                try:
                    scores = [
                        [subchunk for subchunk in chunk if subchunk['label'] == emotion][0]['score']
                        for chunk in chunks_scored
                    ]
                    avg_score = np.nanmean(scores) if scores else 0
                except Exception as e:
                    avg_score = 0
                this_chunk[emotion] = avg_score
            emots.append(this_chunk)

            # Update progress
            if progress_callback:
                progress = (percentile + 1) / num_percentiles
                progress_callback(progress)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        x_values = range(1, num_percentiles + 1)

        for emotion in self.emotions_of_interest:
            emotion_scores = [emotion_score[emotion] for emotion_score in emots]
            ax.plot(x_values, emotion_scores, label=emotion)

        ax.set_xlabel('Percentile of Text')
        ax.set_ylabel('Average Emotion Score')
        ax.set_title('Emotion Analysis Over Text')
        ax.legend()
        plt.tight_layout()
        return fig
