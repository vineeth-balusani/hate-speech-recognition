import streamlit as st
import joblib
import pandas as pd
import model_tf_idf
import lime
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

st.title('Hate Speech Recognition')


###################################################


stopwords = model_tf_idf.stopwords


sentence = st.text_input('Input your tweet here:')

if sentence:
    clf = joblib.load('model_lr_tfidf.pkl')
    count_vect = joblib.load('tf_idf_vect.pkl')
    review_tweet = model_tf_idf.clean_text(sentence)
    pred = clf.predict(count_vect.transform([review_tweet]))

    if pred[0] == 0:
        prediction = "Hate Tweet"
    elif pred[0] == 1:
        prediction = "Offensive"
    else:
        prediction = ' Neither hate nor Offensive'

    st.write(prediction)

if st.checkbox('Explanation'):
    try:
        explainer = LimeTextExplainer(class_names=['Hate', 'Offensive', 'Normal tweet'])


        def pred_fn(text):
            text_transformed = count_vect.transform(text)
            return clf.predict_proba(text_transformed)


        explanation = explainer.explain_instance(sentence, classifier_fn=pred_fn, top_labels=3)
        explanation.save_to_file('lime_report.html')
        HtmlFile = open("lime_report.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=1000)
    except:
        st.write("Enter your tweet to obtain the explanation")


if st.checkbox('Show class labels'):
    st.write(pd.DataFrame({
        'class Label': ['Hate Tweet', 'Offensive', 'Neither'],
        'Output': ['Hate speech is towards detecting the more prevalent forms of hate speech',
                   'Offensive Tweet consider words that are sexist and derogatory',
                   'Tweets with overall positive sentiment and are more like to belong to neither class', ]
    }))
