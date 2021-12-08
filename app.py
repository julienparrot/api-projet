from flask import Flask, render_template, request, url_for
from sklearn import externals
from joblib import load
import pandas as pd
import regex as re
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from stop_words import get_stop_words

df = pd.read_csv("static/labels.csv", usecols=['class', 'tweet'])
df['tweet'] = df['tweet'].apply(lambda tweet: re.sub('[^A-Za-z]+', ' ', tweet.lower()))
warnings.filterwarnings("ignore")

clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)

df2=df.head(100)
clf = clf.fit(X=df2['tweet'], y=df2['class'])

from joblib import dump

dump((clf), 'model.joblib')

clf = load('model.joblib')

app = Flask(__name__)

@app.route("/")
def form():
    return render_template("form.html")

@app.route("/predict", methods=['POST'])
def predict():
    result = request.form
    tweet = result['tweet']
    probas = clf.predict_proba([tweet])[0]
    if probas[0] > probas [1] and probas[0] > probas[2]:
        return render_template('form.html', prediction_text='propos haineux')
    elif probas[1] > probas [0] and probas[1] > probas[2]:
        return render_template('form.html', prediction_text='langage vulgaire')
    else:
        return render_template('form.html', prediction_text='RAS')

if __name__ == "__main__":
    app.run(threaded=True, port=5000)