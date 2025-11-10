# train_effv2_meta_auto_val.py
import os, re, json, random
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from antispoof_model import EffV2WithMeta, build_img_transform, parse_meta_line, IMG_SIZE

DATA_ROOT = Path(r"C:\Users\USUARIO\datasetproject\dataset-muestra\dataset-muestra\data")  # train/ y test/
RUNS_DIR  = Path(r"C:\Users\USUARIO\datasetproject\dataset-muestra\dataset-muestra\runs\effv2_meta")
BATCH_SIZE = 32         # sube a 48–64 si te da la VRAM
EPOCHS = 15
LR = 1.5e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0         # Windows
VAL_RATIO = 0.10
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def find_label_same_dir(img_path: Path):
    p = img_path.with_suffix(".txt")
    if p.exists(): return p
    c = list(img_path.parent.glob(img_path.stem + "_*.txt"))
    return c[0] if c else None

def find_label_in_labels_dir(img_path: Path, labels_dir: Path):
    p = labels_dir / (img_path.stem + ".txt")
    if p.exists(): return p
    c = list(labels_dir.glob(img_path.stem + "_*.txt"))
    return c[0] if c else None

class MixedCelebaSpoof(Dataset):
    def __init__(self, split: str, root: Path, augment=False):
        self.samples = []
        self.tfm = build_img_transform(IMG_SIZE)
        self.split = split
        self.augment = augment and split=="train"
        # A) train/<ID>/<live|spoof>/*
        id_dirs = [p for p in (root/split).iterdir() if p.is_dir() and p.name.lower() not in ("live","spoof")]
        if id_dirs:
            for id_dir in id_dirs:
                for cls,lbl in [("live",1),("spoof",0)]:
                    d = id_dir/cls
                    if not d.exists(): continue
                    for img in d.rglob("*"):
                        if img.suffix.lower() not in (".png",".jpg",".jpeg",".bmp"): continue
                        lab = find_label_same_dir(img)
                        if lab is None: continue
                        self.samples.append((img,lab,lbl))
        else:
            # B) train/<live|spoof>/{images|labels}
            for cls,lbl in [("live",1),("spoof",0)]:
                img_dir = root/split/cls/"images"
                lab_dir = root/split/cls/"labels"
                if not img_dir.exists(): continue
                for img in img_dir.rglob("*"):
                    if img.suffix.lower() not in (".png",".jpg",".jpeg",".bmp"): continue
                    lab = find_label_in_labels_dir(img, lab_dir) if lab_dir.exists() else None
                    if lab is None: continue
                    self.samples.append((img,lab,lbl))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        img_path, txt_path, lbl = self.samples[i]
        im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im is None:
            im = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h0, w0 = im.shape[:2]
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            meta = parse_meta_line(f.readline()).tolist()
        x,y,w,h,conf = meta
        if w0>0 and h0>0:
            meta = np.array([x/w0, y/h0, w/w0, h/h0, conf], dtype=np.float32)
        im = build_img_transform(IMG_SIZE)(im)
        meta = torch.from_numpy(np.array(meta, dtype=np.float32))
        lbl = torch.tensor(lbl, dtype=torch.long)
        return im, meta, lbl

def stratified_split(ds, val_ratio=0.1, seed=42):
    live  = [i for i,(_,_,l) in enumerate(ds.samples) if l==1]
    spoof = [i for i,(_,_,l) in enumerate(ds.samples) if l==0]
    rng = random.Random(seed)
    rng.shuffle(live); rng.shuffle(spoof)
    nlv = int(len(live)*val_ratio); nsp = int(len(spoof)*val_ratio)
    val_idx = set(live[:nlv] + spoof[:nsp])
    train_idx = [i for i in range(len(ds)) if i not in val_idx]
    return train_idx, list(val_idx)

def run_epoch(model, loader, crit, opt, scaler, device, train=True, desc="train"):
    if train: model.train()
    else: model.eval()
    losses, y_true, y_pred = [], [], []
    for imgs, metas, labels in tqdm(loader, desc=desc, ncols=100):
        imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
        if train:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(True):
                logits = model(imgs, metas)
                loss = crit(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            with torch.no_grad():
                logits = model(imgs, metas)
                loss = crit(logits, labels)
        losses.append(float(loss.item()))
        y_pred += logits.argmax(1).detach().cpu().tolist()
        y_true += labels.detach().cpu().tolist()
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    return float(np.mean(losses)), acc, f1

@torch.no_grad()
def eval_loader(model, loader, crit, device, desc="val"):
    return run_epoch(model, loader, crit, None, torch.cuda.amp.GradScaler(False), device, train=False, desc=desc)

def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    full_train = MixedCelebaSpoof("train", DATA_ROOT, augment=True)
    test_ds    = MixedCelebaSpoof("test",  DATA_ROOT, augment=False)
    print(f"[DATA] train_total={len(full_train)} | test={len(test_ds)}")

    tr_idx, va_idx = stratified_split(full_train, VAL_RATIO, SEED)
    tr_ds = Subset(full_train, tr_idx)
    va_ds = Subset(full_train, va_idx)

    trL = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    vaL = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    teL = DataLoader(test_ds,batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = EffV2WithMeta(pretrained=True).to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(True)

    # warmup
    for p in model.backbone.parameters(): p.requires_grad = False
    warm = 2; best = -1.0; hist=[]
    for ep in range(1, EPOCHS+1):
        if ep == warm+1:
            for p in model.backbone.parameters(): p.requires_grad = True
        tr_loss, tr_acc, tr_f1 = run_epoch(model, trL, crit, opt, scaler, device, True, "train")
        va_loss, va_acc, va_f1 = eval_loader(model, vaL, crit, device, "val")
        sch.step()
        print(f"[EP {ep:02d}] tr:loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
              f"va:loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f}")
        hist.append({"ep":ep,"train":{"loss":tr_loss,"acc":tr_acc,"f1":tr_f1},
                          "val":{"loss":va_loss,"acc":va_acc,"f1":va_f1}})
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), RUNS_DIR/"best.pt")
            print(f"[SAVE] best.pt (val_acc={best:.4f})")

    # test final
    if (RUNS_DIR/"best.pt").exists():
        model.load_state_dict(torch.load(RUNS_DIR/"best.pt", map_location=device))
    te_loss, te_acc, te_f1 = eval_loader(model, teL, crit, device, "test")
    print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f} f1={te_f1:.4f}")
    with (RUNS_DIR/"history.json").open("w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

if __name__ == "__main__":
    main()
