# Kernel GPU: Fine-tuning end-to-end de DeBERTa-v3 (o mDeBERTa-v3) en 5-fold StratifiedGroupKFold por RescuerID.
# Multi-task head: Regression (SmoothL1/MSE) + Ordinal CORAL (4 logits) + 5-class CE.
# ZERO leakage: cada fold se entrena solo en train del fold; guarda deberta_oof.parquet y deberta_test.parquet.
import glob, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

try:
    CONFIG
except NameError:
    CONFIG = {"model_name": "microsoft/deberta-v3-base", "tag": "deberta_v3_base", "epochs": 3, "lr": 2e-5, "max_len": 256}

MODEL_NAME = CONFIG.get("model_name", "microsoft/deberta-v3-base")
TAG = CONFIG.get("tag", "deberta_v3_base")
EPOCHS = int(CONFIG.get("epochs", 3))
LR = float(CONFIG.get("lr", 2e-5))
MAX_LEN = int(CONFIG.get("max_len", 256))
BATCH_SIZE = int(CONFIG.get("batch_size", 16))
GRAD_ACCUM = int(CONFIG.get("grad_accum", 2))
SEED = int(CONFIG.get("seed", 42))
OUT = "/kaggle/working"
t0 = time.time()

assert torch.cuda.is_available(), "Sin GPU: necesita T4"
dev = "cuda"
print("torch", torch.__version__, torch.cuda.get_device_name(0), "| model:", MODEL_NAME, "| tag:", TAG, flush=True)


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def buscar(archivo, required=True):
    for prof in range(0, 5):
        for d in glob.glob("/kaggle/input/" + "*/" * prof):
            p = os.path.join(d, archivo)
            if os.path.exists(p):
                return p
    if required:
        raise FileNotFoundError(archivo)
    return None


D = os.path.dirname(os.path.dirname(buscar("train/train.csv")))
train = pd.read_csv(f"{D}/train/train.csv").set_index("PetID")
test = pd.read_csv(f"{D}/test/test.csv").set_index("PetID")
breeds = pd.read_csv(f"{D}/breed_labels.csv").set_index("BreedID")["BreedName"].to_dict()
colors = pd.read_csv(f"{D}/color_labels.csv").set_index("ColorID")["ColorName"].to_dict()
states = pd.read_csv(f"{D}/state_labels.csv").set_index("StateID")["StateName"].to_dict()

# Optional Vision metadata labels + Sentiment text if stats-v2 is attached
p_mtr = buscar("metadata_txt_train.parquet", required=False)
p_mte = buscar("metadata_txt_test.parquet", required=False)
meta_tr = pd.read_parquet(p_mtr) if p_mtr else None
meta_te = pd.read_parquet(p_mte) if p_mte else None

# Compute rescuer_count separately within train and test (zero leakage)
rc_tr = train["RescuerID"].astype(str).map(train["RescuerID"].astype(str).value_counts())
rc_te = test["RescuerID"].astype(str).map(test["RescuerID"].astype(str).value_counts())

TYPE_MAP = {1: "Dog", 2: "Cat"}
GENDER_MAP = {1: "Male", 2: "Female", 3: "Mixed"}
SIZE_MAP = {1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large", 0: "Unknown"}
FUR_MAP = {1: "Short", 2: "Medium", 3: "Long", 0: "Unknown"}
YNU_MAP = {1: "Yes", 2: "No", 3: "Not Sure"}
HEALTH_MAP = {1: "Healthy", 2: "Minor Injury", 3: "Serious Injury", 0: "Unknown"}


def build_texts(df, rc, meta_txt=None):
    texts = []
    for pet_id, r in df.iterrows():
        b1 = breeds.get(r["Breed1"], "Unknown")
        b2 = breeds.get(r["Breed2"], "")
        breed_str = f"{b1} / {b2}" if b2 and b2 != b1 else b1
        cs = [colors.get(r[c], "") for c in ("Color1", "Color2", "Color3")]
        col_str = ", ".join(c for c in cs if c) or "Unknown"
        st_str = states.get(r["State"], "Unknown")
        name_str = str(r["Name"]).strip() if pd.notna(r["Name"]) and str(r["Name"]).strip() else "None"
        d_str = str(r["Description"]).strip() if pd.notna(r["Description"]) else ""
        tags = ""
        if meta_txt is not None and pet_id in meta_txt.index:
            raw_t = str(meta_txt.loc[pet_id, "meta_labels_text"])
            # deduplicate top vision tags
            uniq_t = list(dict.fromkeys(raw_t.replace("_", " ").split()))[:12]
            if uniq_t:
                tags = " | Photo tags: " + ", ".join(uniq_t)
        s = (
            f"{TYPE_MAP.get(r['Type'], 'Pet')} | Breed: {breed_str} | Age: {int(r['Age'])}m | "
            f"Gender: {GENDER_MAP.get(r['Gender'], 'Unk')} | Size: {SIZE_MAP.get(r['MaturitySize'], 'Unk')} | "
            f"Fur: {FUR_MAP.get(r['FurLength'], 'Unk')} | Color: {col_str} | "
            f"Vacc: {YNU_MAP.get(r['Vaccinated'], '?')}, Deworm: {YNU_MAP.get(r['Dewormed'], '?')}, "
            f"Steril: {YNU_MAP.get(r['Sterilized'], '?')}, Health: {HEALTH_MAP.get(r['Health'], '?')} | "
            f"Qty: {int(r['Quantity'])} | Fee: RM {int(r['Fee'])} | Photos: {int(r['PhotoAmt'])} | "
            f"State: {st_str} | Rescuer posts: {int(rc.loc[pet_id])} | Name: {name_str}{tags} | "
            f"Desc: {d_str}"
        )
        texts.append(s)
    return texts


train_texts = build_texts(train, rc_tr, meta_tr)
test_texts = build_texts(test, rc_te, meta_te)
print("Sample prompt:", train_texts[0][:300], flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


class PetDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        item = {"text": self.texts[i]}
        if self.labels is not None:
            item["label"] = int(self.labels[i])
        return item


def collate_fn(batch):
    txts = [b["text"] for b in batch]
    enc = tok(txts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    if "label" in batch[0]:
        enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return enc


class PetTransformer(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.cfg = AutoConfig.from_pretrained(model_name)
        self.cfg.hidden_dropout_prob = 0.15
        self.cfg.attention_probs_dropout_prob = 0.15
        self.backbone = AutoModel.from_pretrained(model_name, config=self.cfg, dtype=torch.float32)
        h = self.cfg.hidden_size
        self.drop = nn.Dropout(0.2)
        self.head_reg = nn.Linear(h, 1)
        self.head_ord = nn.Linear(h, 4)
        self.head_cls = nn.Linear(h, 5)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.backbone.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids
        out = self.backbone(**kwargs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        x = self.drop(pooled)
        return self.head_reg(x).squeeze(-1), self.head_ord(x), self.head_cls(x)


def cortes_cuantiles(p, y):
    props = pd.Series(y).value_counts(normalize=True).sort_index().reindex(range(5), fill_value=0).values
    return np.quantile(p, np.cumsum(props)[:-1])


def eval_preds(model, dl):
    model.eval()
    regs, ords, probs = [], [], []
    with torch.no_grad():
        for batch in dl:
            ids = batch["input_ids"].to(dev, non_blocking=True)
            mask = batch["attention_mask"].to(dev, non_blocking=True)
            tt = batch.get("token_type_ids")
            if tt is not None:
                tt = tt.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                r, o, c = model(ids, mask, tt)
            regs.append(r.float().cpu().numpy())
            ords.append(torch.sigmoid(o.float()).sum(dim=-1).cpu().numpy())
            probs.append(torch.softmax(c.float(), dim=-1).cpu().numpy())
    reg = np.concatenate(regs)
    ord_ev = np.concatenate(ords)
    prob = np.concatenate(probs)
    cls_ev = (prob * np.arange(5)[None, :]).sum(axis=1)
    blend = (reg + ord_ev + cls_ev) / 3.0
    return reg, ord_ev, cls_ev, blend, prob


y_all = train["AdoptionSpeed"].values
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(sgkf.split(train, y_all, groups=train["RescuerID"]))

oof_reg = np.zeros(len(train), dtype=np.float32)
oof_ord = np.zeros(len(train), dtype=np.float32)
oof_cls = np.zeros(len(train), dtype=np.float32)
oof_blend = np.zeros(len(train), dtype=np.float32)
oof_prob = np.zeros((len(train), 5), dtype=np.float32)

test_reg = np.zeros(len(test), dtype=np.float32)
test_ord = np.zeros(len(test), dtype=np.float32)
test_cls = np.zeros(len(test), dtype=np.float32)
test_blend = np.zeros(len(test), dtype=np.float32)
test_prob = np.zeros((len(test), 5), dtype=np.float32)

dl_test = DataLoader(PetDataset(test_texts), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, collate_fn=collate_fn)

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss()

fold_qwks = []
for fold_i, (tr_idx, va_idx) in enumerate(splits):
    t_f = time.time()
    seed_everything(SEED + fold_i)
    dl_tr = DataLoader(
        PetDataset([train_texts[i] for i in tr_idx], y_all[tr_idx]),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn, drop_last=False
    )
    dl_va = DataLoader(
        PetDataset([train_texts[i] for i in va_idx], y_all[va_idx]),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, collate_fn=collate_fn
    )
    model = PetTransformer(MODEL_NAME).to(dev)
    no_decay = ["bias", "LayerNorm.weight"]
    opt_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(opt_params, lr=LR)
    total_steps = math.ceil(len(dl_tr) / GRAD_ACCUM) * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")

    best_rmse = 1e9
    best_state = None
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        tr_loss = 0.0
        for step, batch in enumerate(dl_tr):
            ids = batch["input_ids"].to(dev, non_blocking=True)
            mask = batch["attention_mask"].to(dev, non_blocking=True)
            tt = batch.get("token_type_ids")
            if tt is not None:
                tt = tt.to(dev, non_blocking=True)
            y = batch["labels"].to(dev, non_blocking=True)
            y_f = y.float()
            y_ord = (y.unsqueeze(1) >= torch.arange(1, 5, device=dev).unsqueeze(0)).float()
            with torch.autocast("cuda", dtype=torch.float16):
                r, o, c = model(ids, mask, tt)
                loss = 0.5 * mse_loss(r, y_f) + 0.3 * bce_loss(o, y_ord) + 0.2 * ce_loss(c, y)
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            tr_loss += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(dl_tr):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
        r_va, o_va, c_va, b_va, p_va = eval_preds(model, dl_va)
        rmse = float(np.sqrt(np.mean((b_va - y_all[va_idx]) ** 2)))
        qwk = cohen_kappa_score(y_all[va_idx], np.digitize(b_va, cortes_cuantiles(b_va, y_all[tr_idx])), weights="quadratic")
        print(f"Fold {fold_i} Ep {ep+1}/{EPOCHS} | loss={tr_loss/len(dl_tr):.4f} | val_rmse={rmse:.4f} | val_qwk={qwk:.4f} ({time.time()-t_f:.0f}s)", flush=True)
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    r_va, o_va, c_va, b_va, p_va = eval_preds(model, dl_va)
    oof_reg[va_idx], oof_ord[va_idx], oof_cls[va_idx], oof_blend[va_idx], oof_prob[va_idx] = r_va, o_va, c_va, b_va, p_va
    qwk_best = cohen_kappa_score(y_all[va_idx], np.digitize(b_va, cortes_cuantiles(b_va, y_all[tr_idx])), weights="quadratic")
    fold_qwks.append(qwk_best)
    print(f"Fold {fold_i} BEST rmse={best_rmse:.4f} qwk={qwk_best:.4f}", flush=True)

    r_te, o_te, c_te, b_te, p_te = eval_preds(model, dl_test)
    test_reg += r_te / 5.0
    test_ord += o_te / 5.0
    test_cls += c_te / 5.0
    test_blend += b_te / 5.0
    test_prob += p_te / 5.0

    del model, best_state, opt, sched
    torch.cuda.empty_cache()

print(f"== {TAG} 5-fold Grouped QWK: {np.mean(fold_qwks):.4f} +/- {np.std(fold_qwks, ddof=1):.4f} | folds={[round(x, 4) for x in fold_qwks]}", flush=True)

cols = [f"{TAG}_reg", f"{TAG}_ord", f"{TAG}_cls", f"{TAG}_blend"] + [f"{TAG}_p{k}" for k in range(5)]
oof_df = pd.DataFrame(np.column_stack([oof_reg, oof_ord, oof_cls, oof_blend, oof_prob]), index=train.index, columns=cols, dtype="float32")
test_df = pd.DataFrame(np.column_stack([test_reg, test_ord, test_cls, test_blend, test_prob]), index=test.index, columns=cols, dtype="float32")
oof_df.to_parquet(f"{OUT}/{TAG}_oof.parquet")
test_df.to_parquet(f"{OUT}/{TAG}_test.parquet")
json.dump({"tag": TAG, "model_name": MODEL_NAME, "fold_qwks": fold_qwks, "mean_qwk": float(np.mean(fold_qwks)), "std_qwk": float(np.std(fold_qwks, ddof=1))},
          open(f"{OUT}/metrics.json", "w"), indent=1)
print("LISTO", sorted(os.listdir(OUT)), f"total {time.time() - t0:.0f}s", flush=True)
