<<<<<<< HEAD
import json
from collections import Counter

with open("boot_style.json") as f:
    data = json.load(f)

styles = [d["style_label"] for d in data]
print("Total items:", len(styles))
print("Unique styles:", len(set(styles)))
print("Most common styles:")
for s, c in Counter(styles).most_common(5):
    print(f"{s[:80]}... → {c} times")
=======
import json
from collections import Counter

with open("boot_style.json") as f:
    data = json.load(f)

styles = [d["style_label"] for d in data]
print("Total items:", len(styles))
print("Unique styles:", len(set(styles)))
print("Most common styles:")
for s, c in Counter(styles).most_common(5):
    print(f"{s[:80]}... → {c} times")
>>>>>>> f8020b9cda9abc7043eb62e00f483d48837e0e8e
