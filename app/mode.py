import pickle
from classification.bert import predict_genre
from transformers import BertTokenizerFast, BertForSequenceClassification


with open('classification/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('gyubinc/bert-book32-gyubin', num_labels=32)


if __name__ == "__main__":
    # predict_genre()
    text = "He died yesterday"

    # 텍스트의 장르 예측
    music_genre = predict_genre(text, model, tokenizer, le, k=1)  # 'le'를 사용하셔야 합니다.
    print(music_genre)