# Kernel GPU de preprocesamiento de TEXTO SOTA multilingue (Ingles + Malayo + Chino):
# 1) BAAI/bge-m3 (1024 dims) sobre Description sola -> txt_emb_bgem3_{train,test}.parquet
# 2) BAAI/bge-m3 (1024 dims) sobre perfil completo serializado (raza, edad, salud, estado, nombre, descripcion)
#    -> txt_emb_bgem3_full_{train,test}.parquet
# 3) intfloat/multilingual-e5-large (1024 dims) sobre Description -> txt_emb_e5l_{train,test}.parquet
import glob, os, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

OUT = "/kaggle/working"
t0 = time.time()
assert torch.cuda.is_available(), "Sin GPU: el kernel tiene que correr con acelerador T4"
dev = "cuda"
print("torch", torch.__version__, dev, torch.cuda.get_device_name(0), flush=True)


def buscar(archivo):
    for prof in range(0, 5):
        for d in glob.glob("/kaggle/input/" + "*/" * prof):
            if os.path.exists(os.path.join(d, archivo)):
                return d.rstrip("/")
    raise FileNotFoundError(archivo)


D = buscar("train/train.csv")
train = pd.read_csv(f"{D}/train/train.csv").set_index("PetID")
test = pd.read_csv(f"{D}/test/test.csv").set_index("PetID")
breeds = pd.read_csv(f"{D}/breed_labels.csv").set_index("BreedID")["BreedName"].to_dict()
colors = pd.read_csv(f"{D}/color_labels.csv").set_index("ColorID")["ColorName"].to_dict()
states = pd.read_csv(f"{D}/state_labels.csv").set_index("StateID")["StateName"].to_dict()

all_df = pd.concat([train.drop(columns=["AdoptionSpeed"]), test])
desc = all_df["Description"].fillna("").astype(str).str.strip()

TYPE_MAP = {1: "Dog", 2: "Cat"}
GENDER_MAP = {1: "Male", 2: "Female", 3: "Mixed"}
SIZE_MAP = {1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large", 0: "Unknown"}
FUR_MAP = {1: "Short", 2: "Medium", 3: "Long", 0: "Unknown"}
YNU_MAP = {1: "Yes", 2: "No", 3: "Not Sure"}
HEALTH_MAP = {1: "Healthy", 2: "Minor Injury", 3: "Serious Injury", 0: "Unknown"}


def serializar(r):
    b1 = breeds.get(r["Breed1"], "Unknown")
    b2 = breeds.get(r["Breed2"], "")
    breed_str = f"{b1} / {b2}" if b2 and b2 != b1 else b1
    cs = [colors.get(r[c], "") for c in ("Color1", "Color2", "Color3")]
    col_str = ", ".join(c for c in cs if c) or "Unknown"
    st_str = states.get(r["State"], "Unknown")
    name_str = str(r["Name"]) if pd.notna(r["Name"]) and str(r["Name"]).strip() else "Unnamed"
    d_str = str(r["Description"]).strip() if pd.notna(r["Description"]) else ""
    return (
        f"{TYPE_MAP.get(r['Type'], 'Pet')} | Breed: {breed_str} | Age: {int(r['Age'])} months | "
        f"Gender: {GENDER_MAP.get(r['Gender'], 'Unknown')} | Size: {SIZE_MAP.get(r['MaturitySize'], 'Unknown')} | "
        f"Fur: {FUR_MAP.get(r['FurLength'], 'Unknown')} | Color: {col_str} | "
        f"Vaccinated: {YNU_MAP.get(r['Vaccinated'], 'Unknown')}, Dewormed: {YNU_MAP.get(r['Dewormed'], 'Unknown')}, "
        f"Sterilized: {YNU_MAP.get(r['Sterilized'], 'Unknown')}, Health: {HEALTH_MAP.get(r['Health'], 'Unknown')} | "
        f"Qty: {int(r['Quantity'])} | Fee: RM {int(r['Fee'])} | Photos: {int(r['PhotoAmt'])} | "
        f"State: {st_str} | Name: {name_str} | Description: {d_str}"
    )


full_text = pd.Series([serializar(r) for _, r in all_df.iterrows()], index=all_df.index)
print("Ejemplo serializado:", full_text.iloc[0][:240], flush=True)


def encode_series(model_name, series, prefix="", pooling="cls", max_len=512, batch_size=64, skip_empty=True):
    t1 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    net = AutoModel.from_pretrained(model_name).to(dev).eval().half()
    sub = series[series != ""] if skip_empty else series
    orden = sub.str.len().sort_values().index
    vecs = []
    with torch.no_grad():
        for k in range(0, len(orden), batch_size):
            batch_txt = [prefix + x for x in sub.loc[orden[k:k + batch_size]]]
            enc = tok(batch_txt, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
            out = net(**enc)
            h = out.last_hidden_state.float()
            if pooling == "cls":
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            v = torch.nn.functional.normalize(v, dim=-1)
            vecs.append(v.cpu().numpy())
    V = np.concatenate(vecs).astype("float32")
    emb = pd.DataFrame(V, index=orden, columns=[f"txt_emb_{i}" for i in range(V.shape[1])]).reindex(series.index)
    del net
    torch.cuda.empty_cache()
    print(f"{model_name} ({pooling}, skip_empty={skip_empty}): {emb.shape} en {time.time() - t1:.0f}s", flush=True)
    return emb


# 1) BGE-M3 on Description
e_bge = encode_series("BAAI/bge-m3", desc, prefix="", pooling="cls", max_len=512, batch_size=64, skip_empty=True)
e_bge.loc[train.index].to_parquet(f"{OUT}/txt_emb_bgem3_train.parquet")
e_bge.loc[test.index].to_parquet(f"{OUT}/txt_emb_bgem3_test.parquet")
del e_bge

# 2) BGE-M3 on full serialized profile
e_bge_full = encode_series("BAAI/bge-m3", full_text, prefix="", pooling="cls", max_len=512, batch_size=64, skip_empty=False)
e_bge_full.loc[train.index].to_parquet(f"{OUT}/txt_emb_bgem3_full_train.parquet")
e_bge_full.loc[test.index].to_parquet(f"{OUT}/txt_emb_bgem3_full_test.parquet")
del e_bge_full

# 3) multilingual-e5-large on Description
e_e5l = encode_series("intfloat/multilingual-e5-large", desc, prefix="query: ", pooling="mean", max_len=512, batch_size=64, skip_empty=True)
e_e5l.loc[train.index].to_parquet(f"{OUT}/txt_emb_e5l_train.parquet")
e_e5l.loc[test.index].to_parquet(f"{OUT}/txt_emb_e5l_test.parquet")
del e_e5l

print("LISTO", sorted(os.listdir(OUT)), f"total {time.time() - t0:.0f}s", flush=True)
