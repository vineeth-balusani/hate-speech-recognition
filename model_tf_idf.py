import pandas as pd
import joblib
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


stopwords = set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", \
                 "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
                 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', \
                 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
                 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
                 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
                 'after', \
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', \
                 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                 'more', \
                 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
                 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
                 're', \
                 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', \
                 "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', \
                 "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
                 "weren't", \
                 'won', "won't", 'wouldn', "wouldn't"])


def clean_text(sentence):
    sentence = re.sub(r"@\S+", "", str(sentence))
    sentence = re.sub(r'[RT]+', '', sentence)
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
    return sentence.strip()


def predict(string):
    clf = joblib.load('model_lr_tfidf.pkl')
    count_vect = joblib.load('tf_idf_vect.pkl')
    review_text = clean_text(string)
    test_vect = count_vect.transform(([review_text]))
    pred = clf.predict(test_vect)
    if pred[0] == 0:
        prediction = "Hate Tweet"
    elif pred[0] == 1:
        prediction = "Offensive"
    else:
        prediction = ' Neither hate nor Offensive'
    return prediction


#########################

data = pd.read_csv(r"Data.csv")

preprocessed_tweets = []
for sentence in data['tweet'].values:
    preprocessed_tweets.append(clean_text(str(sentence)))

tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
tf_idf_vectorizer.fit(preprocessed_tweets)
joblib.dump(tf_idf_vectorizer, 'tf_idf_vect.pkl')
X = tf_idf_vectorizer.transform(preprocessed_tweets)
Y = data['class'].values
clf = SGDClassifier(alpha=0.0001, average=False, class_weight='balanced',
                    early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                    l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=1000,
                    n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
                    random_state=None, shuffle=True, tol=0.001,
                    validation_fraction=0.1, verbose=0, warm_start=False)
clf.fit(X, Y)
joblib.dump(clf, 'model_lr_tfidf.pkl')
