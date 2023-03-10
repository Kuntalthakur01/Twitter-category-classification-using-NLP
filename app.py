from flask import Flask, render_template, flash, redirect, url_for, request
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
import numpy as np

# Load the trained model and any pre-processing steps
model = pickle.load(open("nb_classifier.pkl", "rb"))
vocabulary = list(model.feature_log_prob_.argsort(axis=1)[:,-1000:][0])

app = Flask(__name__, template_folder='templates')
app.secret_key = '18071208'

def tokenizer(text):
    tokens = word_tokenize(text)
    return tokens

@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            tweet = request.form['tweet']
            # Preprocess the user input using the same steps used in training
            tweet = re.sub(r'http\S+', '', tweet)  # remove URLs
            tweet = re.sub(r'@[^\s]+', '', tweet)  # remove usernames
            tweet = tweet.lower()  # convert to lowercase
            tokens = tokenizer(tweet)  # tokenize the tweet

            # Convert the tokens to a one-hot encoding
            features = np.zeros((1, len(vocabulary)))
            for token in tokens:
                if token in vocabulary:
                    index = vocabulary.index(token)
                    features[0][index] = 1
            
            # Make the prediction using the trained model
            category = model.predict(features)[0]
            print(category)
            return render_template('predict.html', category=category)
        except KeyError:
            flash('Invalid input. Please enter a valid tweet.')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
