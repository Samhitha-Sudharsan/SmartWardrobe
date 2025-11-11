<<<<<<< HEAD
# inference_helpers.py
import torch
import joblib
import os
from utils_dataset import style_text_from_attrs, ATTRIBUTES
from torchvision import models, transforms
import numpy as np

def build_inference_model(num_classes_map, device="cpu", pretrained=False):
    import torch.nn as nn
    model = models.resnet50(pretrained=pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Identity()
    heads = nn.ModuleDict()
    for head, n in num_classes_map.items():
        heads[head] = nn.Linear(in_feats, n)
    class MultiHeadModel(nn.Module):
        def __init__(self, backbone, heads):
            super().__init__()
            self.backbone = backbone
            self.heads = heads
        def forward(self, x):
            feat = self.backbone(x)
            out = {}
            for k,v in self.heads.items():
                out[k] = v(feat)
            return out
    return MultiHeadModel(model, heads)

def load_classifier(device="cpu"):
    enc_path = "models/attr_encoders.pkl"
    state_path = "models/clothing_model.pt"
    if not os.path.exists(enc_path) or not os.path.exists(state_path):
        return None
    encoders = joblib.load(enc_path)
    num_map = {k: len(encoders[k].classes_) for k in encoders}
    model = build_inference_model(num_map, device=device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return {"model": model, "encoders": encoders, "transform": transform, "device": device}

def predict_attrs_pytorch(pil_image, clf):
    import torch, torch.nn.functional as F
    t = clf["transform"](pil_image).unsqueeze(0).to(clf["device"])
    with torch.no_grad():
        outs = clf["model"](t)
    preds = {}
    for head, out in outs.items():
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        label = clf["encoders"][head].inverse_transform([idx])[0]
        preds[head] = label
    return preds

def load_recommender():
    p = "models/outfit_recommender.pkl"
    if not os.path.exists(p):
        return None
    obj = joblib.load(p)
    return obj  # {'model':..., 'le':LabelEncoder}

def predict_style_from_context(factors, recommender_obj, topk=1):
    # build feature vector same as training
    mood_map = ["confident","happy","neutral","anxious","low"]
    location_map = ["home","office","cafe","public_transport","night_out"]
    event_map = ["work","meeting","casual","date","party"]
    company_map = ["alone","friends","mixed","strangers"]
    weather_map = ["hot","mild","cold","rainy"]
    feat = [
        mood_map.index(factors.get("mood","neutral")) if factors.get("mood","neutral") in mood_map else 2,
        location_map.index(factors.get("location","home")) if factors.get("location","home") in location_map else 0,
        event_map.index(factors.get("event","casual")) if factors.get("event","casual") in event_map else 2,
        company_map.index(factors.get("company","friends")) if factors.get("company","friends") in company_map else 1,
        weather_map.index(factors.get("weather","mild")) if factors.get("weather","mild") in weather_map else 1,
        int(factors.get("safety_score",5)),
        int(factors.get("self_expression",5))
    ]
    # predict probabilities
    probs = recommender_obj["model"].predict_proba([feat])[0]
    idxs = probs.argsort()[::-1][:topk]
    labels = recommender_obj["le"].inverse_transform(idxs)
    return list(zip(labels, probs[idxs]))
=======
# inference_helpers.py
import torch
import joblib
import os
from utils_dataset import style_text_from_attrs, ATTRIBUTES
from torchvision import models, transforms
import numpy as np

def build_inference_model(num_classes_map, device="cpu", pretrained=False):
    import torch.nn as nn
    model = models.resnet50(pretrained=pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Identity()
    heads = nn.ModuleDict()
    for head, n in num_classes_map.items():
        heads[head] = nn.Linear(in_feats, n)
    class MultiHeadModel(nn.Module):
        def __init__(self, backbone, heads):
            super().__init__()
            self.backbone = backbone
            self.heads = heads
        def forward(self, x):
            feat = self.backbone(x)
            out = {}
            for k,v in self.heads.items():
                out[k] = v(feat)
            return out
    return MultiHeadModel(model, heads)

def load_classifier(device="cpu"):
    enc_path = "models/attr_encoders.pkl"
    state_path = "models/clothing_model.pt"
    if not os.path.exists(enc_path) or not os.path.exists(state_path):
        return None
    encoders = joblib.load(enc_path)
    num_map = {k: len(encoders[k].classes_) for k in encoders}
    model = build_inference_model(num_map, device=device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return {"model": model, "encoders": encoders, "transform": transform, "device": device}

def predict_attrs_pytorch(pil_image, clf):
    import torch, torch.nn.functional as F
    t = clf["transform"](pil_image).unsqueeze(0).to(clf["device"])
    with torch.no_grad():
        outs = clf["model"](t)
    preds = {}
    for head, out in outs.items():
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        label = clf["encoders"][head].inverse_transform([idx])[0]
        preds[head] = label
    return preds

def load_recommender():
    p = "models/outfit_recommender.pkl"
    if not os.path.exists(p):
        return None
    obj = joblib.load(p)
    return obj  # {'model':..., 'le':LabelEncoder}

def predict_style_from_context(factors, recommender_obj, topk=1):
    # build feature vector same as training
    mood_map = ["confident","happy","neutral","anxious","low"]
    location_map = ["home","office","cafe","public_transport","night_out"]
    event_map = ["work","meeting","casual","date","party"]
    company_map = ["alone","friends","mixed","strangers"]
    weather_map = ["hot","mild","cold","rainy"]
    feat = [
        mood_map.index(factors.get("mood","neutral")) if factors.get("mood","neutral") in mood_map else 2,
        location_map.index(factors.get("location","home")) if factors.get("location","home") in location_map else 0,
        event_map.index(factors.get("event","casual")) if factors.get("event","casual") in event_map else 2,
        company_map.index(factors.get("company","friends")) if factors.get("company","friends") in company_map else 1,
        weather_map.index(factors.get("weather","mild")) if factors.get("weather","mild") in weather_map else 1,
        int(factors.get("safety_score",5)),
        int(factors.get("self_expression",5))
    ]
    # predict probabilities
    probs = recommender_obj["model"].predict_proba([feat])[0]
    idxs = probs.argsort()[::-1][:topk]
    labels = recommender_obj["le"].inverse_transform(idxs)
    return list(zip(labels, probs[idxs]))
>>>>>>> f8020b9cda9abc7043eb62e00f483d48837e0e8e
