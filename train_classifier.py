<<<<<<< HEAD
# train_classifier.py
import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from utils_dataset import get_train_transforms, get_val_transforms, ATTRIBUTES
from dataset_image_attrs import FashionAttrsDataset
from sklearn.preprocessing import LabelEncoder
import joblib

def build_model(num_classes_map, device="cpu", pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_feats = model.fc.in_features
    # remove fc
    model.fc = nn.Identity()
    # heads
    heads = nn.ModuleDict()
    for head_name, n_classes in num_classes_map.items():
        heads[head_name] = nn.Linear(in_feats, n_classes)
    # wrap model
    class MultiHeadModel(nn.Module):
        def __init__(self, backbone, heads):
            super().__init__()
            self.backbone = backbone
            self.heads = heads
        def forward(self, x):
            feat = self.backbone(x)
            out = {}
            for k, head in self.heads.items():
                out[k] = head(feat)
            return out
    return MultiHeadModel(model, heads).to(device)

def collate_fn(batch):
    imgs, attrs = zip(*batch)
    return imgs, attrs

def encode_attrs(attrs_list, encoders):
    # returns dict head_name -> tensor of labels
    import torch
    out = {}
    for head, le in encoders.items():
        labels = []
        for a in attrs_list:
            val = a.get(head, None)
            if val is None:
                labels.append(0)
            else:
                labels.append(int(le.transform([val])[0]))
        out[head] = torch.tensor(labels, dtype=torch.long)
    return out

def train(args):
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    # read JSON dataset
    train_json = args.train_json
    val_json = args.val_json or args.train_json
    # create encoders for each attribute
    encoders = {}
    num_classes_map = {}
    for head, choices in ATTRIBUTES.items():
        le = LabelEncoder()
        le.fit(choices)
        encoders[head] = le
        num_classes_map[head] = len(le.classes_)
    # instantiate dataset
    train_ds = FashionAttrsDataset(train_json, transforms=get_train_transforms(args.img_size), img_size=args.img_size)
    val_ds = FashionAttrsDataset(val_json, transforms=get_val_transforms(args.img_size), img_size=args.img_size)
    # dataloaders
    def pil_to_tensor(img, transforms):
        return transforms(img)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build model
    model = build_model(num_classes_map, device=device, pretrained=args.pretrained)
    # losses & optim
    criterions = {h: nn.CrossEntropyLoss() for h in num_classes_map}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # training loop
    best_loss = 1e9
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for imgs, attrs in train_loader:
            imgs_t = torch.stack([pil_to_tensor(im, get_train_transforms(args.img_size)) for im in imgs]).to(device)
            y = encode_attrs(attrs, encoders)
            for k in y:
                y[k] = y[k].to(device)
            optimizer.zero_grad()
            outs = model(imgs_t)
            loss = 0
            for head in outs:
                loss_head = criterions[head](outs[head], y[head])
                loss = loss + loss_head
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n += 1
        avg_loss = total_loss / n
        print(f"[Epoch {epoch+1}] train_loss={avg_loss:.4f}")
        # validation
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            vn = 0
            for imgs, attrs in val_loader:
                imgs_t = torch.stack([pil_to_tensor(im, get_val_transforms(args.img_size)) for im in imgs]).to(device)
                y = encode_attrs(attrs, encoders)
                for k in y:
                    y[k] = y[k].to(device)
                outs = model(imgs_t)
                loss = 0
                for head in outs:
                    loss += criterions[head](outs[head], y[head]).item()
                vloss += loss
                vn += 1
            vavg = vloss / max(1, vn)
            print(f"[Epoch {epoch+1}] val_loss={vavg:.4f}")
            if vavg < best_loss:
                best_loss = vavg
                # save model state dict + encoders
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/clothing_model.pt")
                joblib.dump(encoders, "models/attr_encoders.pkl")
                print("Saved best model to models/clothing_model.pt and encoders to models/attr_encoders.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True, help="train annotations json")
    parser.add_argument("--val_json", required=False, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    train(args)
=======
# train_classifier.py
import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from utils_dataset import get_train_transforms, get_val_transforms, ATTRIBUTES
from dataset_image_attrs import FashionAttrsDataset
from sklearn.preprocessing import LabelEncoder
import joblib

def build_model(num_classes_map, device="cpu", pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_feats = model.fc.in_features
    # remove fc
    model.fc = nn.Identity()
    # heads
    heads = nn.ModuleDict()
    for head_name, n_classes in num_classes_map.items():
        heads[head_name] = nn.Linear(in_feats, n_classes)
    # wrap model
    class MultiHeadModel(nn.Module):
        def __init__(self, backbone, heads):
            super().__init__()
            self.backbone = backbone
            self.heads = heads
        def forward(self, x):
            feat = self.backbone(x)
            out = {}
            for k, head in self.heads.items():
                out[k] = head(feat)
            return out
    return MultiHeadModel(model, heads).to(device)

def collate_fn(batch):
    imgs, attrs = zip(*batch)
    return imgs, attrs

def encode_attrs(attrs_list, encoders):
    # returns dict head_name -> tensor of labels
    import torch
    out = {}
    for head, le in encoders.items():
        labels = []
        for a in attrs_list:
            val = a.get(head, None)
            if val is None:
                labels.append(0)
            else:
                labels.append(int(le.transform([val])[0]))
        out[head] = torch.tensor(labels, dtype=torch.long)
    return out

def train(args):
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    # read JSON dataset
    train_json = args.train_json
    val_json = args.val_json or args.train_json
    # create encoders for each attribute
    encoders = {}
    num_classes_map = {}
    for head, choices in ATTRIBUTES.items():
        le = LabelEncoder()
        le.fit(choices)
        encoders[head] = le
        num_classes_map[head] = len(le.classes_)
    # instantiate dataset
    train_ds = FashionAttrsDataset(train_json, transforms=get_train_transforms(args.img_size), img_size=args.img_size)
    val_ds = FashionAttrsDataset(val_json, transforms=get_val_transforms(args.img_size), img_size=args.img_size)
    # dataloaders
    def pil_to_tensor(img, transforms):
        return transforms(img)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build model
    model = build_model(num_classes_map, device=device, pretrained=args.pretrained)
    # losses & optim
    criterions = {h: nn.CrossEntropyLoss() for h in num_classes_map}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # training loop
    best_loss = 1e9
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for imgs, attrs in train_loader:
            imgs_t = torch.stack([pil_to_tensor(im, get_train_transforms(args.img_size)) for im in imgs]).to(device)
            y = encode_attrs(attrs, encoders)
            for k in y:
                y[k] = y[k].to(device)
            optimizer.zero_grad()
            outs = model(imgs_t)
            loss = 0
            for head in outs:
                loss_head = criterions[head](outs[head], y[head])
                loss = loss + loss_head
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n += 1
        avg_loss = total_loss / n
        print(f"[Epoch {epoch+1}] train_loss={avg_loss:.4f}")
        # validation
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            vn = 0
            for imgs, attrs in val_loader:
                imgs_t = torch.stack([pil_to_tensor(im, get_val_transforms(args.img_size)) for im in imgs]).to(device)
                y = encode_attrs(attrs, encoders)
                for k in y:
                    y[k] = y[k].to(device)
                outs = model(imgs_t)
                loss = 0
                for head in outs:
                    loss += criterions[head](outs[head], y[head]).item()
                vloss += loss
                vn += 1
            vavg = vloss / max(1, vn)
            print(f"[Epoch {epoch+1}] val_loss={vavg:.4f}")
            if vavg < best_loss:
                best_loss = vavg
                # save model state dict + encoders
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/clothing_model.pt")
                joblib.dump(encoders, "models/attr_encoders.pkl")
                print("Saved best model to models/clothing_model.pt and encoders to models/attr_encoders.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True, help="train annotations json")
    parser.add_argument("--val_json", required=False, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    train(args)
>>>>>>> f8020b9cda9abc7043eb62e00f483d48837e0e8e
