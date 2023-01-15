import flask
from flask import Flask, jsonify, request
import joblib
from bs4 import BeautifulSoup
import re
import model_tf_idf

app = Flask(__name__)

###################################################


stopwords = model_tf_idf.stopwords


###################################################
@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('model_lr_tfidf.pkl')
    count_vect = joblib.load('tf_idf_vect.pkl')
    to_predict_list = request.form.to_dict()
    review_tweet = model_tf_idf.clean_text(to_predict_list['tweet'])
    pred = clf.predict(count_vect.transform([review_tweet]))

    if pred[0] == 0:
        prediction = "Hate Tweet"
    elif pred[0] == 1:
        prediction = "Offensive"
    else:
        prediction = ' Neither hate nor Offensive'
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
