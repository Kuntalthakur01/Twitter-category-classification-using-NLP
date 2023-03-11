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

# Create a dictionary to map predicted categories to their corresponding labels
category_labels = {
    0: "arts_&_culture",
    1: "business_&_entrepreneurs",
    2: "celebrity_&_pop_culture",
    3: "diaries_&_daily_life",
    4: "family",
    5: "fashion_&_style",
    6: "film_tv_&_video",
    7: "fitness_&_health",
    8: "food_&_dining",
    9: "gaming",
    10: "learning_&_educational",
    11: "music",
    12: "news_&_social_concern",
    13: "other_hobbies",
    14: "relationships",
    15: "science_&_technology",
    16: "sports",
    17: "travel_&_adventure",
    18: "youth_&_student_life"
}

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
            category_label = category_labels[category] # Map the predicted category to its corresponding label
            return render_template('predict.html', category=category_label)
        except KeyError:
            flash('Invalid input. Please enter a valid tweet.')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
