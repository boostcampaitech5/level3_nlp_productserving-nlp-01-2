from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from summary.summary import summary
from classification.classification import classification
import joblib
import pandas as pd
from googletrans import Translator  # googletrans==3.1.0a0

import pickle
from classification.bert import predict_genre
from transformers import BertTokenizerFast, BertForSequenceClassification


import os
import torch
import gluonnlp as nlp
import numpy as np
#kobert
from utils import get_tokenizer
from pytorch_kobert import get_pytorch_kobert_model
from BERTClf import BERTDataset


## 요약 모델
MODEL_NAME = "knlpscience/pegasus-ft"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)

## 토픽 모델
with open('classification/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('gyubinc/bert-book32-gyubin', num_labels=32)

## 감정 모델
mood_model, vocab = get_pytorch_kobert_model(cachedir=".cache")
mood_tokenizer = get_tokenizer()
mood_tokenizer = nlp.data.BERTSPTokenizer(mood_tokenizer, vocab, lower=False)


## 번역
def google_trans(text: str, src='ko', tgt='en') -> str:
    translator = Translator()
    target = translator.translate(text, src=src, dest=tgt)
    return target.text

def predict(mood_model, mood_tokenizer, predict_sentence):
    device = next(mood_model.parameters()).device

    dataset_another = [[predict_sentence, '0']]

    another_test = BERTDataset(dataset_another, 0, 1, mood_tokenizer, vocab, 256, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=32, num_workers=5)
    
    mood_model.eval()
    for token_ids, valid_length, segment_ids, label in test_dataloader:
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length.long().to(device)
        label = label.long().to(device)

        out = mood_model(token_ids, valid_length, segment_ids)
        
        test_eval=[]
        for logits in out:
            logits = logits.detach().cpu().numpy()

            label2emotion = ['Fear', 'Surprise', 'Angry', 'Sadness', 'Neutral', 'Happiness', 'Disgust']
            test_eval.append(label2emotion[np.argmax(logits)])
    print(f'감정:{test_eval[0]}')

    return test_eval[0] 


def work(passage):
    music_mood = predict(mood_model, mood_tokenizer, passage)                          # 감정
    passage = google_trans(passage)                                                    # 번역
    music_summary = summary(pegasus_tokenizer, pegasus_model, passage)                 # 요약
    music_genre = predict_genre(passage, bert_model, bert_tokenizer, le, k=1)          # 토픽
                                                   
    return music_summary, music_genre, music_mood