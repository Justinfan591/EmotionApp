#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:59:44 2024
Modified to enhance efficiency and integrate with UI by Justin Fan

@author: samsun.knight, justin.fan


"""


import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from matplotlib import pyplot as plt
from scipy.special import expit
import gc

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')


class Algorithm:
    def __init__(self):
        # Initialize the sentiment analyzer and model
        self.sid = SentimentIntensityAnalyzer()
        self.tokenizer = RobertaTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.model = RobertaForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.labels = self.model.config.id2label

    def split_text_into_chunks(self, text):
        # Split the text into chunks by sentence
        return re.sub('<[^>]+>', '', str(text)).split('. ')

    def score_emotions(self, text_list, progress_callback_chunk=None):
        return_list = []
        for j, text in tqdm(enumerate(text_list), total=len(text_list), desc="Processing chunks"):
            try:
                inputs = self.tokenizer(text, return_tensors='pt')
                with torch.no_grad():
                    logits = np.array(self.model(**inputs).logits[0], dtype=np.float64)
                    scores = expit(logits)
                this_text_dicts = [{'score': scores[score_ind], 'label': self.labels[score_ind]} for score_ind in range(28)]
            except:
                this_text_dicts = [{'score': np.nan, 'label': self.labels[score_ind]} for score_ind in range(28)]
            return_list.append(this_text_dicts)

            # Update the progress bar for the current chunk
            if progress_callback_chunk:
                progress_callback_chunk(j + 1, len(text_list))

        return return_list

    def plot_emotion_graph(self, text, progress_callback_chunk=None, progress_callback_overall=None):
        emots_no_neutral = ['surprise', 'anger', 'sadness', 'fear', 'joy']
        emots = []

        for percentile in tqdm(range(100), desc="Processing text in chunks"):
            bot = ((len(text)) * (percentile)) // 100
            top = bot + 10000

            chunks = self.split_text_into_chunks(text[bot:top])

            chunks_scored = self.score_emotions(chunks, progress_callback_chunk)

            this_chunk = {}
            for emot in emots_no_neutral:
                try:
                    scores = np.array([subchunk['score'] for chunk in chunks_scored for subchunk in chunk if subchunk['label'] == emot])
                    avg_score = np.nanmean(scores) if scores.size > 0 else np.nan
                except IndexError:
                    avg_score = np.nan

                this_chunk[emot] = avg_score

            emots.append(this_chunk)

            # Update the overall progress bar
            if progress_callback_overall:
                progress_callback_overall(percentile + 1, 100)

            # Memory management: clear unused variables and force garbage collection
            del chunks, chunks_scored, this_chunk
            gc.collect()

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        for emot in ['sadness', 'anger', 'fear', 'joy']:
            line, = ax.plot(range(1, 101), [emotion_score[emot] for emotion_score in emots])
            line.set_label(emot)

        fig.legend()
        return fig
