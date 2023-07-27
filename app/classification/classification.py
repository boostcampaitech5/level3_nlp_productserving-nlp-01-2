import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder



def classification(data, clf, text):
    text = list(text)
    genre = pd.DataFrame(data['Genre'])

    feat = ['Genre']
    for x in feat:
        le = LabelEncoder()
        le.fit(list(genre[x].values))
        genre[x] = le.transform(list(genre[x]))

    vectorizer = TfidfVectorizer(min_df=2, max_features=70000, strip_accents='unicode',lowercase =True,
                                analyzer='word', token_pattern=r'\w+', use_idf=True, 
                                smooth_idf=True, sublinear_tf=True, stop_words = 'english')
    vectors = vectorizer.fit_transform(data['everything'])

    

    
    text[0] = text[0].lower()
    s = (vectorizer.transform(text))
    d = (clf.predict(s))
    genre_answer = le.inverse_transform(d)[0]
    return genre_answer

if __name__ == "__main__":
    data = pd.read_csv('classification/vector_data.csv',encoding = "ISO-8859-1")
    clf = joblib.load('classification/best.pkl')
    text = 'I am happy'
    print(classification(data, clf, text))