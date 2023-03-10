# from flask import Flask, render_template, request
# import pickle
# import re
# model = pickle.load(open("nb_classifier.pkl", "rb"))

# app = Flask(__name__, template_folder='templates')

# @app.route("/predict", methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         tweet = request.form['tweet']  # get the tweet input from the user
#         tweet = request.form['tweet']  # get the tweet input from the user
#         category = model.predict(tweet)  # use your ML model to predict the category
#         return render_template('predict.html', category=category)
#     return render_template('predict.html')
#
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, flash, redirect, url_for, request
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize

def tokenizer(text):
    tokens = word_tokenize(text)
    return tokens
# Load the trained model and any pre-processing steps
model = pickle.load(open("nb_classifier.pkl", "rb"))

app = Flask(__name__, template_folder='/Users/farhatshaikh01/Documents/GitHub/Twitter-category-classification-using-NLP/templates')
#app = Flask(__name__, static_url_path='/static')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        # Preprocess the user input using the same steps used in training
        tweet = re.sub(r'http\S+', '', tweet)  # remove URLs
        tweet = re.sub(r'@[^\s]+', '', tweet)  # remove usernames
        tweet = tweet.lower()  # convert to lowercase
        tokens = tokenizer(tweet)  # tokenize the tweet
        
        # Make the prediction using the trained model
        category = model.predict(tokens)[0]
        return render_template('predict.html', category=category)
    
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)

