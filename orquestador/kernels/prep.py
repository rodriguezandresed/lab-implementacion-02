# Kernel de preprocesamiento (lo sube orquestador.py; CONFIG se inyecta arriba de este archivo).
#
# CONFIG["modelo"] elige que se calcula. Cada modelo es un kernel aparte en Kaggle (lab2-prep-<modelo>),
# asi un embedding se calcula una sola vez y lo reusan todos los experimentos.
#   stats     -> img_stats_{train,test}.parquet (calidad de la foto 1, = img-v2 del equipo)
#                txt_stats_{train,test}.parquet (8 estadisticas de la descripcion, = texto-v2 del equipo)
#   effb0     -> img_effb0_first_{train,test}.parquet: EfficientNetB0 keras de la foto 1 (= preprocesado/img del
#                equipo, con el arreglo del PetID). Es la imagen del baseline EXP-026.
#   clip-l14, dinov2-b, dinov2-l, siglip-b
#             -> img_<modelo>_first_{train,test}.parquet: embedding de la foto 1
#                img_<modelo>_mean_{train,test}.parquet:  promedio de los embeddings (normalizados) de todas las fotos
# Las columnas de imagen siempre se llaman img_feat_1..D, asi el modelo las trata igual que las de EfficientNet.
import glob, os, re, time
import numpy as np
import pandas as pd

try:
    CONFIG
except NameError:
    CONFIG = {"modelo": "stats"}
MODELO = CONFIG["modelo"]
OUT = "/kaggle/working"


def buscar(archivo):
    """Carpeta de /kaggle/input que contiene `archivo` (Kaggle monta los inputs en rutas distintas segun como se crea el kernel)."""
    for prof in range(0, 5):
        for d in glob.glob("/kaggle/input/" + "*/" * prof):
            if os.path.exists(os.path.join(d, archivo)):
                return d.rstrip("/")
    raise FileNotFoundError(archivo)


D = buscar("train/train.csv")
print("competencia en", D, "| modelo", MODELO, flush=True)
train = pd.read_csv(f"{D}/train/train.csv").set_index("PetID")
test = pd.read_csv(f"{D}/test/test.csv").set_index("PetID")


def fotos(parte, solo_primera):
    fs = sorted(f for f in os.listdir(f"{D}/{parte}_images") if f.endswith(".jpg"))
    if solo_primera:
        fs = [f for f in fs if f.endswith("-1.jpg")]
    pet = [f.rsplit("-", 1)[0] for f in fs]  # no usar strip("-1.jpg"): borra los 1 del PetID
    num = [int(f.rsplit("-", 1)[1][:-4]) for f in fs]
    return pd.DataFrame({"path": [f"{D}/{parte}_images/{f}" for f in fs], "PetID": pet, "n": num})


# ------------------------------------------------------------------ stats (sin modelo)
if MODELO == "stats":
    import cv2

    def calidad(path):
        im = cv2.imread(path)
        if im is None:
            return {}
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        return {"img_nitidez": cv2.Laplacian(g, cv2.CV_64F).var(), "img_brillo": g.mean(), "img_contraste": g.std(),
                "img_ancho": w, "img_alto": h, "img_aspecto": w / h}

    for parte in ("train", "test"):
        f = fotos(parte, True)
        q = pd.DataFrame([calidad(p) for p in f["path"]], index=f["PetID"].values)
        q.to_parquet(f"{OUT}/img_stats_{parte}.parquet")
        print("img_stats", parte, q.shape, flush=True)

    MALAYO = {"dan", "yang", "untuk", "ini", "itu", "boleh", "tidak", "saya", "dia", "ada", "sangat", "anak",
              "kucing", "anjing", "dengan", "nak", "jantan", "betina", "sudah", "sihat", "comel", "manja", "akan",
              "lagi", "sila", "hubungi", "percuma", "mahu", "kami", "dari", "pada", "dalam", "tak", "dah"}

    def stats(t):
        letras = sum(c.isalpha() for c in t)
        palabras = re.findall(r"[A-Za-z]+", t)
        return {
            "txt_largo": len(t),
            "txt_palabras": len(t.split()),
            "txt_frac_chino": len(re.findall(r"[一-鿿]", t)) / len(t) if t else np.nan,
            "txt_frac_malayo": sum(w.lower() in MALAYO for w in palabras) / len(palabras) if palabras else np.nan,
            "txt_frac_mayus": sum(c.isupper() for c in t) / letras if letras else np.nan,
            "txt_exclamaciones": t.count("!"),
            "txt_telefono": int(bool(re.search(r"\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}", t))),
            "txt_precio": int(bool(re.search(r"\b(rm\s?\d+|fee|fees|ringgit)\b|\$\s?\d+", t, re.I))),
        }

    desc = pd.concat([train["Description"], test["Description"]]).fillna("").astype(str).str.strip()
    st = pd.DataFrame([stats(t) for t in desc], index=desc.index)
    st.loc[train.index].to_parquet(f"{OUT}/txt_stats_train.parquet")
    st.loc[test.index].to_parquet(f"{OUT}/txt_stats_test.parquet")
    print("txt_stats", st.shape, flush=True)

# ------------------------------------------------------------------ EfficientNetB0 (baseline, keras, igual al equipo)
elif MODELO == "effb0":
    # Mismos pesos ImageNet de keras que el notebook del equipo, pero con backend torch: el TensorFlow de la imagen
    # de Kaggle no toma la GPU (la corrida del 2026-11-25 paso 85+ min en CPU). Con torch corre en la T4.
    os.environ["KERAS_BACKEND"] = "torch"
    import torch
    assert torch.cuda.is_available(), "Sin GPU: el kernel tiene que correr con acelerador T4"
    import keras
    from keras.utils import load_img, img_to_array
    from keras.applications.efficientnet import EfficientNetB0, preprocess_input
    print("keras", keras.__version__, "backend", keras.backend.backend(), "| GPU", torch.cuda.get_device_name(0), flush=True)

    model = EfficientNetB0(weights="imagenet", include_top=False, pooling="max")
    for parte in ("train", "test"):
        f = fotos(parte, True)
        t0, filas = time.time(), []
        for k, p in enumerate(f["path"]):
            img = preprocess_input(np.expand_dims(img_to_array(load_img(p)), axis=0))
            with torch.no_grad():
                filas.append(keras.ops.convert_to_numpy(model(img, training=False))[0])
            if k % 2000 == 0:
                print(parte, k, "de", len(f), f"{time.time() - t0:.0f}s", flush=True)
        e = pd.DataFrame(np.array(filas, dtype="float32"), index=f["PetID"].values,
                         columns=[f"img_feat_{i + 1}" for i in range(1280)])
        e.to_parquet(f"{OUT}/img_effb0_first_{parte}.parquet")
        print("effb0", parte, e.shape, flush=True)

# ------------------------------------------------------------------ emociones multilingues (texto, zero-shot NLI)
elif MODELO.startswith("emo-"):
    # Mismas 7 emociones que el modelo de la catedra (DistilRoBERTa, solo ingles), pero con un NLI multilingue:
    # para cada descripcion y emocion se pregunta si "This text expresses {emocion}." se deduce del texto, y se
    # hace softmax de los logits de entailment entre las 7 (como el pipeline zero-shot con multi_label=False).
    # Las hipotesis en ingles funcionan en cualquier idioma porque el modelo es cross-lingual.
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    EMO_HF = {"emo-mdeberta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"}
    EMOCIONES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    HIPOTESIS = {"anger": "This text expresses anger.", "disgust": "This text expresses disgust.",
                 "fear": "This text expresses fear.", "joy": "This text expresses joy.",
                 "neutral": "This text is neutral and expresses no emotion.", "sadness": "This text expresses sadness.",
                 "surprise": "This text expresses surprise."}
    assert torch.cuda.is_available(), "Sin GPU: el kernel tiene que correr con acelerador T4"
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(EMO_HF[MODELO])
    nli = AutoModelForSequenceClassification.from_pretrained(EMO_HF[MODELO]).to(dev).eval()
    ENT = {k.lower(): v for k, v in nli.config.label2id.items()}["entailment"]
    print("torch", torch.__version__, torch.cuda.get_device_name(0), "| entailment =", ENT, flush=True)

    desc = pd.concat([train["Description"], test["Description"]]).fillna("").astype(str).str.strip()
    con_texto = desc[desc != ""]
    orden = con_texto.str.len().sort_values().index  # lotes de largo parecido: menos padding
    logits = pd.DataFrame(np.nan, index=orden, columns=EMOCIONES)
    B, t0 = 64, time.time()
    with torch.no_grad():
        for emo in EMOCIONES:
            res = []
            for k in range(0, len(orden), B):
                t = con_texto.loc[orden[k:k + B]].tolist()
                enc = tok(t, [HIPOTESIS[emo]] * len(t), truncation="only_first", max_length=256,
                          padding=True, return_tensors="pt").to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = nli(**enc).logits.float()
                if not torch.isfinite(lg).all():  # DeBERTa a veces desborda en fp16: repetir ese lote en fp32
                    lg = nli(**enc).logits.float()
                res.append(lg[:, ENT].cpu().numpy())
            logits[emo] = np.concatenate(res)
            print(emo, "listo", f"{time.time() - t0:.0f}s", flush=True)
    prob = np.exp(logits.values - logits.values.max(axis=1, keepdims=True))
    prob = pd.DataFrame(prob / prob.sum(axis=1, keepdims=True), index=orden,
                        columns=[f"emo_{e}" for e in EMOCIONES]).reindex(desc.index)  # sin descripcion -> NaN
    nombre = MODELO[len("emo-"):]
    prob.loc[train.index].to_parquet(f"{OUT}/emo_{nombre}_train.parquet")
    prob.loc[test.index].to_parquet(f"{OUT}/emo_{nombre}_test.parquet")
    print(prob.describe().T.round(3).to_string(), flush=True)
    # control: las 2 descripciones con mas probabilidad de cada emocion
    for c in prob.columns:
        print("==", c)
        for i in prob[c].nlargest(2).index:
            print("  ", round(prob.loc[i, c], 3), desc[i][:140].replace(chr(10), " "))

# ------------------------------------------------------------------ embeddings de texto (= texto-v2 del equipo, celda 3)
elif MODELO.startswith("txt-"):
    import torch
    from transformers import AutoTokenizer, AutoModel
    TXT_HF = {"txt-e5": "intfloat/multilingual-e5-base", "txt-e5l": "intfloat/multilingual-e5-large"}
    assert torch.cuda.is_available(), "Sin GPU: el kernel tiene que correr con acelerador T4"
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(TXT_HF[MODELO])
    enc_model = AutoModel.from_pretrained(TXT_HF[MODELO]).to(dev).eval().half()
    print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
    desc = pd.concat([train["Description"], test["Description"]]).fillna("").astype(str).str.strip()
    con_texto = desc[desc != ""]
    orden = con_texto.str.len().sort_values().index
    vecs, B, t0 = [], 64, time.time()
    with torch.no_grad():
        for k in range(0, len(orden), B):
            t = ["query: " + x for x in con_texto.loc[orden[k:k + B]]]
            enc = tok(t, truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
            h = enc_model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            vecs.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy())
    V = np.concatenate(vecs).astype("float32")
    emb = pd.DataFrame(V, index=orden, columns=[f"txt_emb_{i}" for i in range(V.shape[1])]).reindex(desc.index)
    nombre = MODELO[len("txt-"):]
    emb.loc[train.index].to_parquet(f"{OUT}/txt_emb_{nombre}_train.parquet")
    emb.loc[test.index].to_parquet(f"{OUT}/txt_emb_{nombre}_test.parquet")
    print(nombre, emb.shape, "sin descripcion:", int(emb.isna().all(axis=1).sum()), f"{time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ modelos de Hugging Face (torch)
else:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    HF = {"clip-l14": "openai/clip-vit-large-patch14",
          "dinov2-s": "facebook/dinov2-small",  # ViT-S/14, 384 dims
          "dinov2-b": "facebook/dinov2-base",
          "dinov2-l": "facebook/dinov2-large",
          "siglip-b": "google/siglip-base-patch16-224"}
    assert torch.cuda.is_available(), "Sin GPU: el kernel tiene que correr con acelerador T4"
    dev = "cuda"
    print("torch", torch.__version__, dev, torch.cuda.get_device_name(0), flush=True)
    proc = AutoImageProcessor.from_pretrained(HF[MODELO])
    if MODELO.startswith("clip"):
        from transformers import CLIPVisionModelWithProjection
        net = CLIPVisionModelWithProjection.from_pretrained(HF[MODELO])
        salida = lambda o: o.image_embeds  # proyeccion al espacio compartido texto-imagen (768)
    elif MODELO.startswith("siglip"):
        from transformers import SiglipVisionModel
        net = SiglipVisionModel.from_pretrained(HF[MODELO])
        salida = lambda o: o.pooler_output
    else:
        net = AutoModel.from_pretrained(HF[MODELO])
        salida = lambda o: o.pooler_output  # token CLS normalizado
    net = net.to(dev).eval()
    if dev == "cuda":
        net = net.half()

    class Fotos(torch.utils.data.Dataset):
        def __init__(self, paths):
            self.paths = list(paths)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            try:
                im = Image.open(self.paths[i]).convert("RGB")
                return proc(images=im, return_tensors="pt")["pixel_values"][0], True
            except Exception:
                s = proc.crop_size if hasattr(proc, "crop_size") and proc.crop_size else proc.size
                h = s.get("height", s.get("shortest_edge", 224))
                return torch.zeros(3, h, s.get("width", h)), False

    for parte in ("train", "test"):
        f = fotos(parte, CONFIG.get("solo_primera", False))
        dl = torch.utils.data.DataLoader(Fotos(f["path"]), batch_size=64, num_workers=4)
        t0, vecs, ok = time.time(), [], []
        with torch.no_grad():
            for k, (x, bien) in enumerate(dl):
                x = x.to(dev).half() if dev == "cuda" else x.to(dev)
                v = torch.nn.functional.normalize(salida(net(pixel_values=x)).float(), dim=-1)
                vecs.append(v.cpu().numpy())
                ok.append(bien.numpy())
                if k % 100 == 0:
                    print(parte, k * 64, "de", len(f), f"{time.time() - t0:.0f}s", flush=True)
        V = np.concatenate(vecs).astype("float32")
        ok = np.concatenate(ok)
        ilegibles = int((~ok).sum())
        f, V = f[ok].reset_index(drop=True), V[ok]
        cols = [f"img_feat_{i + 1}" for i in range(V.shape[1])]
        todo = pd.DataFrame(V, columns=cols)
        todo["PetID"] = f["PetID"].values
        primera = todo[f["n"].values == 1].set_index("PetID")[cols]
        primera.to_parquet(f"{OUT}/img_{MODELO}_first_{parte}.parquet")
        media = todo.groupby("PetID")[cols].mean()
        media.to_parquet(f"{OUT}/img_{MODELO}_mean_{parte}.parquet")
        print(MODELO, parte, "fotos", len(f), "| primera", primera.shape, "| media", media.shape,
              f"| {time.time() - t0:.0f}s | fotos ilegibles {ilegibles}", flush=True)

print("LISTO", sorted(os.listdir(OUT)), flush=True)
