<<<<<<< HEAD
# dataset_image_attrs.py
import json
import os
from torch.utils.data import Dataset
from PIL import Image

class FashionAttrsDataset(Dataset):
    """
    Expects a JSON list: [{"image_path": "...", "attrs": {"category":..., "color":..., ...}}]
    For synthetic bootstrapping, image_path can be None (we'll generate a solid color image).
    """
    def __init__(self, json_path, transforms=None, img_size=224):
        with open(json_path, "r") as f:
            self.items = json.load(f)
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img_path = rec.get("image_path")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            # create placeholder image with dominant color from attrs
            color = rec.get("attrs", {}).get("color", "blue")
            # map color name to RGB roughly
            mapc = {"white":(255,255,255),"black":(5,5,5),"blue":(20,90,200),"navy":(10,30,120),"red":(200,30,30),
                    "pink":(255,150,180),"green":(30,160,60),"yellow":(240,220,50),"beige":(230,200,170),"brown":(120,80,40),
                     "grey":(140,140,140),"purple":(130,50,160)}
            rgb = mapc.get(color, (100,100,100))
            img = Image.new("RGB", (self.img_size, self.img_size), rgb)
        label_attrs = rec.get("attrs", {})
        # flatten labels to multi-hot / integer labels for heads
        return img, label_attrs
=======
# dataset_image_attrs.py
import json
import os
from torch.utils.data import Dataset
from PIL import Image

class FashionAttrsDataset(Dataset):
    """
    Expects a JSON list: [{"image_path": "...", "attrs": {"category":..., "color":..., ...}}]
    For synthetic bootstrapping, image_path can be None (we'll generate a solid color image).
    """
    def __init__(self, json_path, transforms=None, img_size=224):
        with open(json_path, "r") as f:
            self.items = json.load(f)
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img_path = rec.get("image_path")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            # create placeholder image with dominant color from attrs
            color = rec.get("attrs", {}).get("color", "blue")
            # map color name to RGB roughly
            mapc = {"white":(255,255,255),"black":(5,5,5),"blue":(20,90,200),"navy":(10,30,120),"red":(200,30,30),
                    "pink":(255,150,180),"green":(30,160,60),"yellow":(240,220,50),"beige":(230,200,170),"brown":(120,80,40),
                     "grey":(140,140,140),"purple":(130,50,160)}
            rgb = mapc.get(color, (100,100,100))
            img = Image.new("RGB", (self.img_size, self.img_size), rgb)
        label_attrs = rec.get("attrs", {})
        # flatten labels to multi-hot / integer labels for heads
        return img, label_attrs
>>>>>>> f8020b9cda9abc7043eb62e00f483d48837e0e8e
