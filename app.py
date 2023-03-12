from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch  

app = Flask(__name__)
MODEL = "cardiffnlp/tweet-topic-21-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
class_mapping = model.config.id2label

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

@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        tweet = request.form["tweet"]
        inputs = tokenizer.encode_plus(tweet, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        category_id = int(torch.argmax(logits, axis=1).detach().cpu().numpy()[0])
        category = category_labels[category_id]
        return render_template('predict.html', tweet=tweet, category=category, probabilities=probabilities)
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run()
2{
3    "name": "Twitter-category-classification-using-NLP",
4    "version": "1.0.0",
5    "description": "",
6    "scripts": {
7        "build": "echo \"No build specified\" && exit 1",
8        "test": "echo \"No test specified\" && exit 1",
9        "start": "node index.js"
10    },
11    "repository": {
12        "type": "git",
13        "url": "git+https://github.com/Kuntalthakur01/Twitter-category-classification-using-NLP.git"
14    },
15    "author": "",
16    "license": "ISC",
17    "bugs": {
18        "url": "https://github.com/Kuntalthakur01/Twitter-category-classification-using-NLP/issues"
19    },
20    "homepage": "https://github.com/Kuntalthakur01/Twitter-category-classification-using-NLP#readme"
21}