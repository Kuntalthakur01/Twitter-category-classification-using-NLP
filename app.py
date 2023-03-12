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
