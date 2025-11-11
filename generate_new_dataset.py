<<<<<<< HEAD
import json, random
from utils_dataset import ATTRIBUTES, style_text_from_attrs

data = []
for _ in range(4000):
    attrs = {
        "category": random.choice(ATTRIBUTES["category"]),
        "color": random.choice(ATTRIBUTES["color"]),
        "pattern": random.choice(ATTRIBUTES["pattern"]),
        "sleeve": random.choice(ATTRIBUTES["sleeve"]),
        "fit": random.choice(ATTRIBUTES["fit"]),
        "bottom_style": random.choice(ATTRIBUTES["bottom_style"]),
        "footwear": random.choice(["white sneakers", "black boots", "sandals", "heels", "loafers"])
    }
    ctx = {
        "mood": random.choice(["confident", "happy", "neutral", "anxious", "low"]),
        "event": random.choice(["work", "meeting", "casual", "date", "party"]),
        "company": random.choice(["friends", "alone", "mixed", "strangers"]),
        "weather": random.choice(["hot", "mild", "cold", "rainy"]),
        "safety_score": random.randint(1, 10),
        "self_expression": random.randint(1, 10)
    }
    style_label = style_text_from_attrs(attrs)
    data.append({
        "image_path": None,
        "attrs": attrs,
        "style_label": style_label,
        "context": ctx
    })

with open("boot_style.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Generated", len(data), "diverse style examples with context.")
=======
import json, random
from utils_dataset import ATTRIBUTES, style_text_from_attrs

data = []
for _ in range(4000):
    attrs = {
        "category": random.choice(ATTRIBUTES["category"]),
        "color": random.choice(ATTRIBUTES["color"]),
        "pattern": random.choice(ATTRIBUTES["pattern"]),
        "sleeve": random.choice(ATTRIBUTES["sleeve"]),
        "fit": random.choice(ATTRIBUTES["fit"]),
        "bottom_style": random.choice(ATTRIBUTES["bottom_style"]),
        "footwear": random.choice(["white sneakers", "black boots", "sandals", "heels", "loafers"])
    }
    ctx = {
        "mood": random.choice(["confident", "happy", "neutral", "anxious", "low"]),
        "event": random.choice(["work", "meeting", "casual", "date", "party"]),
        "company": random.choice(["friends", "alone", "mixed", "strangers"]),
        "weather": random.choice(["hot", "mild", "cold", "rainy"]),
        "safety_score": random.randint(1, 10),
        "self_expression": random.randint(1, 10)
    }
    style_label = style_text_from_attrs(attrs)
    data.append({
        "image_path": None,
        "attrs": attrs,
        "style_label": style_label,
        "context": ctx
    })

with open("boot_style.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Generated", len(data), "diverse style examples with context.")
>>>>>>> f8020b9cda9abc7043eb62e00f483d48837e0e8e
