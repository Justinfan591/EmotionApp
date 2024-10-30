#!/usr/bin/env python3
"""
Updated to reflect the original algorithm, processing text in chunks and plotting specified emotions.
"""

import os
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

    def split_text_into_chunks(self, text, chunk_size=10000):
        # Remove HTML tags and split text into chunks of specified size
        text = re.sub('<[^>]+>', '', str(text))
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks

    def score_emotions(self, text_list):
        return_list = []
        for j, text in enumerate(tqdm(text_list, desc="Scoring emotions", leave=False)):
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
        # Define emotions of interest
        emotions_of_interest = ['sadness', 'anger', 'fear', 'joy']

        # Split text into chunks
        chunks = self.split_text_into_chunks(text)

        num_chunks = len(chunks)
        emots = []

        for idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks", leave=False)):
            # Process progress
            if progress_callback:
                progress = (idx + 1) / num_chunks
                progress_callback(progress)

            # Score emotions for the chunk
            chunks_scored = self.score_emotions([chunk])

            # Aggregate emotion scores
            this_chunk = {}
            for emotion in emotions_of_interest:
                try:
                    scores = [subchunk['score'] for sublist in chunks_scored for subchunk in sublist if subchunk['label'] == emotion]
                    avg_score = np.nanmean(scores) if scores else 0
                except Exception as e:
                    avg_score = 0
                this_chunk[emotion] = avg_score
            emots.append(this_chunk)

        # Prepare data for plotting
        x_values = range(len(emots))
        fig, ax = plt.subplots(figsize=(12, 6))

        for emotion in emotions_of_interest:
            emotion_scores = [emotion_score[emotion] for emotion_score in emots]
            ax.plot(x_values, emotion_scores, label=emotion)

        ax.set_xlabel('Chunk Index')
        ax.set_ylabel('Emotion Score')
        ax.set_title('Emotion Analysis Over Text')
        ax.legend()
        plt.tight_layout()
        return fig
