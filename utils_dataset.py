<<<<<<< HEAD
# utils_dataset.py
import os
import random
import json
from PIL import Image
from torchvision import transforms

ATTRIBUTES = {
    "category": ["tshirt", "shirt", "blouse", "dress", "pullover", "jacket", "coat", "jeans", "trouser", "skirt", "shorts", "sneaker", "boot", "sandal"],
    "color": ["white", "black", "blue", "navy", "red", "pink", "green", "yellow", "beige", "brown", "grey", "purple"],
    "pattern": ["plain", "striped", "floral", "polka", "checked", "graphic"],
    "sleeve": ["short", "long", "sleeveless", "three-quarter"],
    "fit": ["slim", "regular", "oversized"],
    "bottom_style": ["ripped jeans", "skinny jeans", "straight jeans", "chinos", "tailored trousers", "pleated skirt", "mini skirt"]
}

def style_text_from_attrs(attrs):
    """Produce a human-friendly style phrase from predicted attributes."""
    parts = []
    # top part
    if attrs.get("pattern", "plain") != "plain":
        parts.append(f"{attrs.get('pattern')}")
    parts.append(f"{attrs.get('color')} {attrs.get('category')}")
    if attrs.get("sleeve"):
        parts.append(f"{attrs.get('sleeve')}-sleeve")
    if attrs.get("fit") and attrs.get("category") not in ["jeans", "trouser", "skirt"]:
        parts.append(attrs.get("fit"))
    top_phrase = " ".join(parts).strip()
    # bottom
    bottom = attrs.get("bottom_style")
    if bottom:
        style = f"{top_phrase} with {attrs.get('color','blue')} {bottom}"
    else:
        style = top_phrase
    # shoes
    if attrs.get("footwear"):
        style = f"{style} + {attrs.get('footwear')}"
    return style

# basic torchvision transforms
def get_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def get_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

# small utility to create synthetic labeled examples (image placeholder)
def create_synthetic_label_csv(out_csv="synthetic_labels.json", n=2000, seed=42):
    random.seed(seed)
    examples = []
    for i in range(n):
        attrs = {
            "category": random.choice(ATTRIBUTES["category"]),
            "color": random.choice(ATTRIBUTES["color"]),
            "pattern": random.choice(ATTRIBUTES["pattern"]),
            "sleeve": random.choice(ATTRIBUTES["sleeve"]),
            "fit": random.choice(ATTRIBUTES["fit"]),
            "bottom_style": random.choice(ATTRIBUTES["bottom_style"]),
            "footwear": random.choice(["white sneakers","black boots","brown loafers","sandals","none"])
        }
        # style label: a short textual description the recommender will learn to predict
        style_label = style_text_from_attrs(attrs)
        examples.append({"image_path": None, "attrs": attrs, "style_label": style_label})
    with open(out_csv, "w") as f:
        json.dump(examples, f, indent=2)
    print("Wrote", out_csv)
    return out_csv
=======
# utils_dataset.py
import os
import random
import json
from PIL import Image
from torchvision import transforms

ATTRIBUTES = {
    "category": ["tshirt", "shirt", "blouse", "dress", "pullover", "jacket", "coat", "jeans", "trouser", "skirt", "shorts", "sneaker", "boot", "sandal"],
    "color": ["white", "black", "blue", "navy", "red", "pink", "green", "yellow", "beige", "brown", "grey", "purple"],
    "pattern": ["plain", "striped", "floral", "polka", "checked", "graphic"],
    "sleeve": ["short", "long", "sleeveless", "three-quarter"],
    "fit": ["slim", "regular", "oversized"],
    "bottom_style": ["ripped jeans", "skinny jeans", "straight jeans", "chinos", "tailored trousers", "pleated skirt", "mini skirt"]
}

def style_text_from_attrs(attrs):
    """Produce a human-friendly style phrase from predicted attributes."""
    parts = []
    # top part
    if attrs.get("pattern", "plain") != "plain":
        parts.append(f"{attrs.get('pattern')}")
    parts.append(f"{attrs.get('color')} {attrs.get('category')}")
    if attrs.get("sleeve"):
        parts.append(f"{attrs.get('sleeve')}-sleeve")
    if attrs.get("fit") and attrs.get("category") not in ["jeans", "trouser", "skirt"]:
        parts.append(attrs.get("fit"))
    top_phrase = " ".join(parts).strip()
    # bottom
    bottom = attrs.get("bottom_style")
    if bottom:
        style = f"{top_phrase} with {attrs.get('color','blue')} {bottom}"
    else:
        style = top_phrase
    # shoes
    if attrs.get("footwear"):
        style = f"{style} + {attrs.get('footwear')}"
    return style

# basic torchvision transforms
def get_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def get_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

# small utility to create synthetic labeled examples (image placeholder)
def create_synthetic_label_csv(out_csv="synthetic_labels.json", n=2000, seed=42):
    random.seed(seed)
    examples = []
    for i in range(n):
        attrs = {
            "category": random.choice(ATTRIBUTES["category"]),
            "color": random.choice(ATTRIBUTES["color"]),
            "pattern": random.choice(ATTRIBUTES["pattern"]),
            "sleeve": random.choice(ATTRIBUTES["sleeve"]),
            "fit": random.choice(ATTRIBUTES["fit"]),
            "bottom_style": random.choice(ATTRIBUTES["bottom_style"]),
            "footwear": random.choice(["white sneakers","black boots","brown loafers","sandals","none"])
        }
        # style label: a short textual description the recommender will learn to predict
        style_label = style_text_from_attrs(attrs)
        examples.append({"image_path": None, "attrs": attrs, "style_label": style_label})
    with open(out_csv, "w") as f:
        json.dump(examples, f, indent=2)
    print("Wrote", out_csv)
    return out_csv
>>>>>>> f8020b9cda9abc7043eb62e00f483d48837e0e8e
