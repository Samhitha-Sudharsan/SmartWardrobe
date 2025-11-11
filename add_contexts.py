<<<<<<< HEAD
import json, random

with open("boot_style.json") as f:
    data = json.load(f)

for item in data:
    top = item["attrs"]["category"]
    ctx = {}
    if top in ["shirt", "blouse", "jacket"]:
        ctx["event"] = random.choice(["work", "meeting"])
        ctx["mood"] = random.choice(["confident", "happy"])
    elif top in ["dress", "skirt"]:
        ctx["event"] = random.choice(["date", "party"])
        ctx["mood"] = random.choice(["happy", "neutral"])
    elif top in ["tshirt", "shorts"]:
        ctx["event"] = random.choice(["casual", "party"])
        ctx["mood"] = random.choice(["happy", "low"])
    else:
        ctx["event"] = random.choice(["casual"])
        ctx["mood"] = random.choice(["neutral"])
    ctx["company"] = random.choice(["friends", "alone"])
    ctx["weather"] = random.choice(["hot", "mild", "cold"])
    ctx["safety_score"] = random.randint(1, 10)
    ctx["self_expression"] = random.randint(1, 10)
    item["context"] = ctx

with open("boot_style.json", "w") as f:
    json.dump(data, f, indent=2)

print("Updated contexts intelligently.")
=======
import json, random

with open("boot_style.json") as f:
    data = json.load(f)

for item in data:
    top = item["attrs"]["category"]
    ctx = {}
    if top in ["shirt", "blouse", "jacket"]:
        ctx["event"] = random.choice(["work", "meeting"])
        ctx["mood"] = random.choice(["confident", "happy"])
    elif top in ["dress", "skirt"]:
        ctx["event"] = random.choice(["date", "party"])
        ctx["mood"] = random.choice(["happy", "neutral"])
    elif top in ["tshirt", "shorts"]:
        ctx["event"] = random.choice(["casual", "party"])
        ctx["mood"] = random.choice(["happy", "low"])
    else:
        ctx["event"] = random.choice(["casual"])
        ctx["mood"] = random.choice(["neutral"])
    ctx["company"] = random.choice(["friends", "alone"])
    ctx["weather"] = random.choice(["hot", "mild", "cold"])
    ctx["safety_score"] = random.randint(1, 10)
    ctx["self_expression"] = random.randint(1, 10)
    item["context"] = ctx

with open("boot_style.json", "w") as f:
    json.dump(data, f, indent=2)

print("Updated contexts intelligently.")
>>>>>>> f8020b9cda9abc7043eb62e00f483d48837e0e8e
