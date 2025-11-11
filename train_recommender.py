<<<<<<< HEAD
# train_recommender.py
import argparse
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils_dataset import ATTRIBUTES

def load_style_dataset(json_path):
    with open(json_path, "r") as f:
        examples = json.load(f)
    X = []
    y = []
    for rec in examples:
        ctx = rec.get("context", {})  # optional
        # default context keys
        mood = ctx.get("mood", rec.get("mood", "neutral"))
        location = ctx.get("location", rec.get("location", "home"))
        event = ctx.get("event", rec.get("event", "casual"))
        company = ctx.get("company", rec.get("company", "friends"))
        weather = ctx.get("weather", rec.get("weather", "mild"))
        safety = int(ctx.get("safety_score", rec.get("safety_score", 5)))
        self_expr = int(ctx.get("self_expression", rec.get("self_expression", 5)))
        # feature vector: encode as indices
        feat = [
            ["confident","happy","neutral","anxious","low"].index(mood) if mood in ["confident","happy","neutral","anxious","low"] else 2,
            ["home","office","cafe","public_transport","night_out"].index(location) if location in ["home","office","cafe","public_transport","night_out"] else 0,
            ["work","meeting","casual","date","party"].index(event) if event in ["work","meeting","casual","date","party"] else 2,
            ["alone","friends","mixed","strangers"].index(company) if company in ["alone","friends","mixed","strangers"] else 1,
            ["hot","mild","cold","rainy"].index(weather) if weather in ["hot","mild","cold","rainy"] else 1,
            safety, self_expr
        ]
        X.append(feat)
        y.append(rec["style_label"])
    return np.array(X), np.array(y)

def train(args):
    X, y = load_style_dataset(args.style_json)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.12, random_state=42)
    model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Val acc:", acc)
    joblib.dump({"model": model, "le": le}, "models/outfit_recommender.pkl")
    print("Saved models/outfit_recommender.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_json", required=True, help="JSON with examples having style_label and optional context")
    args = parser.parse_args()
    train(args)
=======
# train_recommender.py
import argparse
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils_dataset import ATTRIBUTES

def load_style_dataset(json_path):
    with open(json_path, "r") as f:
        examples = json.load(f)
    X = []
    y = []
    for rec in examples:
        ctx = rec.get("context", {})  # optional
        # default context keys
        mood = ctx.get("mood", rec.get("mood", "neutral"))
        location = ctx.get("location", rec.get("location", "home"))
        event = ctx.get("event", rec.get("event", "casual"))
        company = ctx.get("company", rec.get("company", "friends"))
        weather = ctx.get("weather", rec.get("weather", "mild"))
        safety = int(ctx.get("safety_score", rec.get("safety_score", 5)))
        self_expr = int(ctx.get("self_expression", rec.get("self_expression", 5)))
        # feature vector: encode as indices
        feat = [
            ["confident","happy","neutral","anxious","low"].index(mood) if mood in ["confident","happy","neutral","anxious","low"] else 2,
            ["home","office","cafe","public_transport","night_out"].index(location) if location in ["home","office","cafe","public_transport","night_out"] else 0,
            ["work","meeting","casual","date","party"].index(event) if event in ["work","meeting","casual","date","party"] else 2,
            ["alone","friends","mixed","strangers"].index(company) if company in ["alone","friends","mixed","strangers"] else 1,
            ["hot","mild","cold","rainy"].index(weather) if weather in ["hot","mild","cold","rainy"] else 1,
            safety, self_expr
        ]
        X.append(feat)
        y.append(rec["style_label"])
    return np.array(X), np.array(y)

def train(args):
    X, y = load_style_dataset(args.style_json)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.12, random_state=42)
    model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Val acc:", acc)
    joblib.dump({"model": model, "le": le}, "models/outfit_recommender.pkl")
    print("Saved models/outfit_recommender.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_json", required=True, help="JSON with examples having style_label and optional context")
    args = parser.parse_args()
    train(args)
>>>>>>> f8020b9cda9abc7043eb62e00f483d48837e0e8e
