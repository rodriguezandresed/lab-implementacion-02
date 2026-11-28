# Kernel GPU: End-to-End Fine-Tuning de Vision + Tabular (timm ConvNeXt / Swin) directamente sobre las fotos
# principales de cada mascota (<PetID>-1.jpg) en 5-fold StratifiedGroupKFold(seed=42) por RescuerID.
# Inspirado en soluciones ganadoras de PetFinder Pawpularity: adapta el backbone visual a la adopcion
# con CosineAnnealingLR, Mixup/Augmentations, Multi-Task head (MSE + CORAL Ordinal + 5-class CE) y TTA (flip horizontal!).
# ZERO leakage: cada fold se entrena solo con train del fold; mascotas sin foto usan rama tabular sola.
import gc, glob, json, math, os, random, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer

try:
    CONFIG
except NameError:
    CONFIG = {
        "backbone": "convnext_small.fb_in22k_ft_in1k_384",
        "tag": "e2e_convnext_s",
        "img_size": 256,
        "epochs": 4,
        "batch_size": 32,
        "lr_backbone": 5e-5,
        "lr_head": 5e-4,
        "seed": 42,
    }

BACKBONE = CONFIG.get("backbone", "convnext_small.fb_in22k_ft_in1k_384")
TAG = CONFIG.get("tag", "e2e_convnext_s")
IMG_SIZE = int(CONFIG.get("img_size", 256))
EPOCHS = int(CONFIG.get("epochs", 4))
BS = int(CONFIG.get("batch_size", 32))
LR_BB = float(CONFIG.get("lr_backbone", 5e-5))
LR_HD = float(CONFIG.get("lr_head", 5e-4))
SEED = int(CONFIG.get("seed", 42))
OUT = "/kaggle/working"
t0 = time.time()

assert torch.cuda.is_available(), "Sin GPU: necesita T4"
dev = "cuda"
print(f"Device: {torch.cuda.get_device_name(0)} | backbone={BACKBONE} | img_size={IMG_SIZE} | epochs={EPOCHS}", flush=True)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def buscar(archivo):
    for prof in range(0, 5):
        for d in glob.glob("/kaggle/input/" + "*/" * prof):
            p = os.path.join(d, archivo)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(archivo)


D = os.path.dirname(os.path.dirname(buscar("train/train.csv")))
train_raw = pd.read_csv(f"{D}/train/train.csv").set_index("PetID")
test_raw = pd.read_csv(f"{D}/test/test.csv").set_index("PetID")
y_all = train_raw["AdoptionSpeed"].astype(int).values
groups_all = train_raw["RescuerID"].astype(str).values


def build_tab(df):
    out = pd.DataFrame(index=df.index)
    out["Type"] = (df["Type"] == 1).astype("float32")
    out["Age_log"] = np.log1p(df["Age"].clip(lower=0).astype("float32"))
    out["Fee_log"] = np.log1p(df["Fee"].clip(lower=0).astype("float32"))
    out["Fee_free"] = (df["Fee"] == 0).astype("float32")
    out["Qty_log"] = np.log1p(df["Quantity"].clip(lower=0).astype("float32"))
    out["PhotoAmt"] = df["PhotoAmt"].astype("float32")
    out["VideoAmt"] = df["VideoAmt"].astype("float32")
    out["MaturitySize"] = df["MaturitySize"].astype("float32")
    out["FurLength"] = df["FurLength"].astype("float32")
    out["Gender_M"] = (df["Gender"] == 1).astype("float32")
    out["Gender_F"] = (df["Gender"] == 2).astype("float32")
    out["Health"] = df["Health"].astype("float32")
    out["Care_yes"] = sum((df[c] == 1).astype("float32") for c in ("Vaccinated", "Dewormed", "Sterilized"))
    out["Care_unsure"] = sum((df[c] == 3).astype("float32") for c in ("Vaccinated", "Dewormed", "Sterilized"))
    out["Has_breed2"] = (df["Breed2"] > 0).astype("float32")
    out["Desc_len"] = np.log1p(df["Description"].fillna("").astype(str).str.len().astype("float32"))
    resc = df["RescuerID"].astype(str)
    out["Resc_cnt_log"] = np.log1p(resc.map(resc.value_counts()).astype("float32"))
    out["Has_photo"] = (df["PhotoAmt"] > 0).astype("float32")
    return out.values.astype("float32")


tab_tr_raw = build_tab(train_raw)
tab_te_raw = build_tab(test_raw)


def img_paths(idx, parte):
    paths = []
    for pid in idx:
        p = f"{D}/{parte}_images/{pid}-1.jpg"
        paths.append(p if os.path.exists(p) else "")
    return paths


paths_tr = img_paths(train_raw.index, "train")
paths_te = img_paths(test_raw.index, "test")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
tf_train = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
tf_val = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class PetImgDataset(Dataset):
    def __init__(self, paths, tab, labels=None, tf=tf_val):
        self.paths = paths
        self.tab = tab
        self.labels = labels
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        has_im = 1.0
        if p:
            try:
                im = Image.open(p).convert("RGB")
                x = self.tf(im)
            except Exception:
                x = torch.zeros(3, IMG_SIZE, IMG_SIZE)
                has_im = 0.0
        else:
            x = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            has_im = 0.0
        t = torch.tensor(self.tab[i], dtype=torch.float32)
        if self.labels is not None:
            return x, t, torch.tensor(has_im, dtype=torch.float32), torch.tensor(self.labels[i], dtype=torch.long)
        return x, t, torch.tensor(has_im, dtype=torch.float32)


class VisionTabNet(nn.Module):
    def __init__(self, backbone_name, tab_dim):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, drop_path_rate=0.2)
        v_dim = self.backbone.num_features
        self.tab_net = nn.Sequential(nn.Linear(tab_dim, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(v_dim + 64, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.25))
        self.out_reg = nn.Linear(128, 1)
        self.out_ord = nn.Linear(128, 4)
        self.out_cls = nn.Linear(128, 5)

    def forward(self, x_img, x_tab, has_im):
        vf = self.backbone(x_img) * has_im.unsqueeze(-1)
        tf = self.tab_net(x_tab)
        h = self.head(torch.cat([vf, tf], dim=-1))
        return self.out_reg(h).squeeze(-1), self.out_ord(h), self.out_cls(h)


def cortes_cuantiles(p, y):
    props = pd.Series(y).value_counts(normalize=True).sort_index().reindex(range(5), fill_value=0).values
    return np.quantile(p, np.cumsum(props)[:-1])


def eval_tta(model, dl):
    """Evaluate with horizontal-flip TTA (5th place Minixhofer trick!)."""
    model.eval()
    regs, ords, probs = [], [], []
    with torch.no_grad():
        for batch in dl:
            x, t, hi = batch[0].to(dev, non_blocking=True), batch[1].to(dev, non_blocking=True), batch[2].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                r1, o1, c1 = model(x, t, hi)
                r2, o2, c2 = model(torch.flip(x, dims=[-1]), t, hi)
            r = 0.5 * (r1.float() + r2.float())
            o = 0.5 * (torch.sigmoid(o1.float()).sum(-1) + torch.sigmoid(o2.float()).sum(-1))
            p = 0.5 * (torch.softmax(c1.float(), -1) + torch.softmax(c2.float(), -1))
            regs.append(r.cpu().numpy())
            ords.append(o.cpu().numpy())
            probs.append(p.cpu().numpy())
    reg = np.concatenate(regs)
    ord_ev = np.concatenate(ords)
    prob = np.concatenate(probs)
    cls_ev = (prob * np.arange(5)[None, :]).sum(1)
    blend = (reg + ord_ev + cls_ev) / 3.0
    return reg, ord_ev, cls_ev, blend, prob


sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits_5 = list(sgkf5.split(train_raw, y_all, groups=groups_all))

oof_reg = np.zeros(len(train_raw), dtype=np.float32)
oof_ord = np.zeros(len(train_raw), dtype=np.float32)
oof_blend = np.zeros(len(train_raw), dtype=np.float32)
oof_prob = np.zeros((len(train_raw), 5), dtype=np.float32)

test_reg = np.zeros(len(test_raw), dtype=np.float32)
test_ord = np.zeros(len(test_raw), dtype=np.float32)
test_blend = np.zeros(len(test_raw), dtype=np.float32)
test_prob = np.zeros((len(test_raw), 5), dtype=np.float32)

mse, bce, ce = nn.MSELoss(), nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
fold_qwks = []

for fi, (tr_i, va_i) in enumerate(splits_5):
    tf = time.time()
    seed_all(SEED + fi)
    qt = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=42)
    Ttr = np.clip(qt.fit_transform(tab_tr_raw[tr_i]), -5, 5).astype("float32")
    Tva = np.clip(qt.transform(tab_tr_raw[va_i]), -5, 5).astype("float32")
    Tte = np.clip(qt.transform(tab_te_raw), -5, 5).astype("float32")

    dl_tr = DataLoader(PetImgDataset([paths_tr[i] for i in tr_i], Ttr, y_all[tr_i], tf=tf_train),
                       batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    dl_va = DataLoader(PetImgDataset([paths_tr[i] for i in va_i], Tva, y_all[va_i], tf=tf_val),
                       batch_size=BS * 2, shuffle=False, num_workers=4, pin_memory=True)
    dl_te = DataLoader(PetImgDataset(paths_te, Tte, None, tf=tf_val),
                       batch_size=BS * 2, shuffle=False, num_workers=4, pin_memory=True)

    model = VisionTabNet(BACKBONE, Ttr.shape[1]).to(dev)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BB, "weight_decay": 1e-2},
        {"params": list(model.tab_net.parameters()) + list(model.head.parameters()) +
                   list(model.out_reg.parameters()) + list(model.out_ord.parameters()) + list(model.out_cls.parameters()),
         "lr": LR_HD, "weight_decay": 2e-2},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(dl_tr))
    scaler = torch.amp.GradScaler("cuda")

    best_rmse = 1e9
    best_state = None
    for ep in range(EPOCHS):
        model.train()
        tr_l = 0.0
        for x, t, hi, y in dl_tr:
            x, t, hi, y = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True), hi.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            y_ord = (y.unsqueeze(1) >= torch.arange(1, 5, device=dev).unsqueeze(0)).float()
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                r, o, c = model(x, t, hi)
                loss = 0.45 * mse(r, y.float()) + 0.35 * bce(o, y_ord) + 0.20 * ce(c, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr_l += loss.item()
        rv, ov, cv, bv, pv = eval_tta(model, dl_va)
        rmse = float(np.sqrt(np.mean((bv - y_all[va_i]) ** 2)))
        qwk = cohen_kappa_score(y_all[va_i], np.digitize(bv, cortes_cuantiles(bv, y_all[tr_i])), weights="quadratic")
        print(f"Fold {fi} Ep {ep+1}/{EPOCHS} | loss={tr_l/len(dl_tr):.4f} | rmse={rmse:.4f} | qwk={qwk:.4f} ({time.time()-tf:.0f}s)", flush=True)
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    rv, ov, cv, bv, pv = eval_tta(model, dl_va)
    oof_reg[va_i], oof_ord[va_i], oof_blend[va_i], oof_prob[va_i] = rv, ov, bv, pv
    fq = cohen_kappa_score(y_all[va_i], np.digitize(bv, cortes_cuantiles(bv, y_all[tr_i])), weights="quadratic")
    fold_qwks.append(fq)
    print(f"Fold {fi} BEST rmse={best_rmse:.4f} qwk={fq:.4f}", flush=True)

    rt, ot, ct, bt, pt = eval_tta(model, dl_te)
    test_reg += rt / 5.0; test_ord += ot / 5.0; test_blend += bt / 5.0; test_prob += pt / 5.0
    del model, best_state, opt, sched
    torch.cuda.empty_cache()
    gc.collect()

tot_rmse = float(np.sqrt(np.mean((oof_blend - y_all) ** 2)))
print(f"== {TAG} 5-fold SGKF RMSE={tot_rmse:.4f} | QWK={np.mean(fold_qwks):.4f} +/- {np.std(fold_qwks, ddof=1):.4f} | folds={[round(x, 4) for x in fold_qwks]}", flush=True)

cols = [f"{TAG}_reg", f"{TAG}_ord", f"{TAG}_blend"] + [f"{TAG}_p{k}" for k in range(5)]
pd.DataFrame(np.column_stack([oof_reg, oof_ord, oof_blend, oof_prob]), index=train_raw.index, columns=cols, dtype="float32").to_parquet(f"{OUT}/{TAG}_oof.parquet")
pd.DataFrame(np.column_stack([test_reg, test_ord, test_blend, test_prob]), index=test_raw.index, columns=cols, dtype="float32").to_parquet(f"{OUT}/{TAG}_test.parquet")
json.dump({"tag": TAG, "backbone": BACKBONE, "rmse_5f": tot_rmse, "qwk_5f_mean": float(np.mean(fold_qwks)),
           "qwk_5f_std": float(np.std(fold_qwks, ddof=1)), "qwk_5f_folds": [round(x, 5) for x in fold_qwks],
           "elapsed_sec": round(time.time() - t0, 1)}, open(f"{OUT}/metrics.json", "w"), indent=1)
print("LISTO", sorted(os.listdir(OUT)), f"total {time.time()-t0:.0f}s", flush=True)
