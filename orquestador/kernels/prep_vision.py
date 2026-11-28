# Kernel GPU de preprocesamiento de IMAGEN (DINOv2-Large, DINOv2-Small, SigLIP-SO400M, EVA02, etc.)
# Genera:
#   - img_<modelo>_first_{train,test}.parquet (embedding L2-norm de foto 1)
#   - img_<modelo>_mean_{train,test}.parquet  (promedio L2-renormalizado de todas las fotos)
#   - img_<modelo>_extra_{train,test}.parquet (diversidad intra-mascota: norm_of_mean, cos_first_mean, std, n_imgs)
# Si es SigLIP: ademas calcula 12 prompts zero-shot visuales + similitud coseno imagen-descripcion!
import glob, os, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

try:
    CONFIG
except NameError:
    CONFIG = {"modelo": "dinov2-l", "max_fotos": 0}

MODELO = CONFIG["modelo"]
MAX_FOTOS = int(CONFIG.get("max_fotos", 0))
OUT = "/kaggle/working"
t0 = time.time()

assert torch.cuda.is_available(), "Sin GPU: el kernel tiene que correr con acelerador T4"
dev = "cuda"
print("torch", torch.__version__, dev, torch.cuda.get_device_name(0), "| modelo:", MODELO, flush=True)


def buscar(archivo):
    for prof in range(0, 5):
        for d in glob.glob("/kaggle/input/" + "*/" * prof):
            if os.path.exists(os.path.join(d, archivo)):
                return d.rstrip("/")
    raise FileNotFoundError(archivo)


D = buscar("train/train.csv")
train = pd.read_csv(f"{D}/train/train.csv").set_index("PetID")
test = pd.read_csv(f"{D}/test/test.csv").set_index("PetID")


def listar_fotos(parte):
    fs = sorted(f for f in os.listdir(f"{D}/{parte}_images") if f.endswith(".jpg"))
    pet = [f.rsplit("-", 1)[0] for f in fs]
    num = [int(f.rsplit("-", 1)[1][:-4]) for f in fs]
    df = pd.DataFrame({"path": [f"{D}/{parte}_images/{f}" for f in fs], "PetID": pet, "n": num})
    if MAX_FOTOS > 0:
        df = df[df["n"] <= MAX_FOTOS].reset_index(drop=True)
    return df


HF = {
    "dinov2-s": "facebook/dinov2-small",
    "dinov2-b": "facebook/dinov2-base",
    "dinov2-l": "facebook/dinov2-large",
    "dinov2-g": "facebook/dinov2-giant",
    "convnext-l": "facebook/convnext-large-224-22k-1k",
    "siglip-so400m": "google/siglip-so400m-patch14-384",
    "siglip2-so400m": "google/siglip2-so400m-patch14-384",
    "siglip-b": "google/siglip-base-patch16-224",
    "siglip-b-384": "google/siglip-base-patch16-384",
}

IS_SIGLIP = MODELO.startswith("siglip")
if IS_SIGLIP:
    from transformers import AutoProcessor
    full_proc = AutoProcessor.from_pretrained(HF[MODELO])
    proc = full_proc.image_processor
    full_net = AutoModel.from_pretrained(HF[MODELO]).to(dev).eval().half()
    net = full_net.vision_model
    BATCH = 32 if "384" in HF[MODELO] else 64
else:
    proc = AutoImageProcessor.from_pretrained(HF[MODELO])
    net = AutoModel.from_pretrained(HF[MODELO]).to(dev).eval().half()
    BATCH = 64 if "large" in HF[MODELO] else 128


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


def extra_stats(f_df, V, idx):
    """Calcula diversidad visual intra-mascota a partir de embeddings L2-normalizados."""
    # Agrupar por PetID usando indices consecutivos
    order = np.argsort(f_df["PetID"].values, kind="mergesort")
    pets = f_df["PetID"].values[order]
    ns = f_df["n"].values[order]
    Vs = V[order]
    uniq, starts, counts = np.unique(pets, return_index=True, return_counts=True)
    rows = []
    for pet_id, st, cnt in zip(uniq, starts, counts):
        sub = Vs[st:st + cnt]
        sub_n = ns[st:st + cnt]
        mean_v = sub.mean(axis=0)
        norm_mean = float(np.linalg.norm(mean_v))
        mean_unit = mean_v / max(norm_mean, 1e-8)
        sims_to_mean = sub @ mean_unit
        first_idx = np.where(sub_n == 1)[0]
        cos_first_mean = float(sims_to_mean[first_idx[0]]) if len(first_idx) else float(sims_to_mean[0])
        rows.append({
            "PetID": pet_id,
            f"{MODELO}_norm_mean": norm_mean,
            f"{MODELO}_cos_first_mean": cos_first_mean,
            f"{MODELO}_cos_min_mean": float(sims_to_mean.min()),
            f"{MODELO}_cos_std_mean": float(sims_to_mean.std()) if cnt > 1 else 0.0,
            f"{MODELO}_emb_std_mean": float(sub.std(axis=0).mean()) if cnt > 1 else 0.0,
        })
    return pd.DataFrame(rows).set_index("PetID").reindex(idx).astype("float32")


# If SigLIP, precompute text embeddings for 12 visual aesthetic/context prompts + pet descriptions
if IS_SIGLIP:
    PARES = {
        "siglip_cute": ("a photo of a very cute, adorable, photogenic pet", "an ordinary, unattractive photo of a pet"),
        "siglip_baby": ("a photo of a very young baby puppy or kitten", "a photo of a full grown adult dog or cat"),
        "siglip_cage": ("a photo of an animal inside a metal cage or kennel", "a photo of an animal outside a cage in a home"),
        "siglip_blurry": ("a blurry, dark, noisy, low quality photo", "a crisp, sharp, well-lit, high quality photo"),
        "siglip_indoor": ("a photo of a pet in a clean cozy indoor home", "a photo of a stray pet outdoors on the street"),
        "siglip_person": ("a photo of a friendly person holding or petting an animal", "a photo of an animal alone with no people"),
        "siglip_multi": ("a photo of several puppies or kittens together", "a photo of a single lone animal"),
        "siglip_sick": ("a photo of a sick, skinny, injured or dirty stray animal", "a photo of a healthy, clean, well-groomed pet"),
        "siglip_purebred": ("a photo of a purebred fancy dog or cat", "a photo of a mixed breed local stray dog or cat"),
        "siglip_happy": ("a photo of a happy, playful, smiling pet looking at camera", "a photo of a scared, sad, hiding animal"),
        "siglip_closeup": ("a close-up portrait of a pet face looking at camera", "a distant full-body photo of an animal far away"),
        "siglip_fluffy": ("a photo of a fluffy long-haired pet", "a photo of a short-haired pet"),
    }
    frases = [f for pair in PARES.values() for f in pair]
    def _to_tensor(o):
        return o.pooler_output if hasattr(o, "pooler_output") else o
    with torch.no_grad():
        t_in = full_proc(text=frases, padding="max_length", truncation=True, return_tensors="pt").to(dev)
        prompt_emb = torch.nn.functional.normalize(_to_tensor(full_net.get_text_features(**t_in)).float(), dim=-1)  # (24, D)
        logit_scale = full_net.logit_scale.exp().float().item()
        logit_bias = full_net.logit_bias.float().item()
    print(f"SigLIP prompts encoded: {prompt_emb.shape}, scale={logit_scale:.2f}, bias={logit_bias:.2f}", flush=True)


for parte, df_meta in (("train", train), ("test", test)):
    f = listar_fotos(parte)
    dl = torch.utils.data.DataLoader(Fotos(f["path"]), batch_size=BATCH, num_workers=4, pin_memory=True)
    t_part, vecs, ok = time.time(), [], []
    with torch.no_grad():
        for k, (x, bien) in enumerate(dl):
            x = x.to(dev, non_blocking=True).half()
            out = net(pixel_values=x)
            feat = out.pooler_output.float()
            v = torch.nn.functional.normalize(feat, dim=-1)
            vecs.append(v.cpu().numpy())
            ok.append(bien.numpy())
            if k % 150 == 0:
                print(f"{MODELO} {parte}: {k * BATCH}/{len(f)} ({time.time() - t_part:.0f}s)", flush=True)
    V = np.concatenate(vecs).astype("float32")
    ok = np.concatenate(ok)
    f, V = f[ok].reset_index(drop=True), V[ok]
    cols = [f"img_feat_{i + 1}" for i in range(V.shape[1])]
    todo = pd.DataFrame(V, columns=cols)
    todo["PetID"] = f["PetID"].values

    primera = todo[f["n"].values == 1].set_index("PetID")[cols]
    primera.to_parquet(f"{OUT}/img_{MODELO}_first_{parte}.parquet")

    media = todo.groupby("PetID")[cols].mean()
    # L2 re-normalize mean embeddings for cleaner cosine geometry
    norms = np.linalg.norm(media.values, axis=1, keepdims=True)
    media_norm = pd.DataFrame(media.values / np.maximum(norms, 1e-8), index=media.index, columns=cols, dtype="float32")
    media_norm.to_parquet(f"{OUT}/img_{MODELO}_mean_{parte}.parquet")

    ex = extra_stats(f, V, df_meta.index)
    ex.to_parquet(f"{OUT}/img_{MODELO}_extra_{parte}.parquet")

    if IS_SIGLIP:
        # 1) Zero-shot visual scores on mean embedding & photo 1
        with torch.no_grad():
            V_mean_t = torch.tensor(media_norm.values, device=dev)
            sims = (V_mean_t @ prompt_emb.T) * logit_scale + logit_bias  # (N_pets, 24)
            # Pairwise softmax probability for each prompt pair
            probs = torch.softmax(sims.view(len(media_norm), len(PARES), 2), dim=-1)[:, :, 0].cpu().numpy()
            # Also raw sigmoid of positive prompt
            sigs = torch.sigmoid(sims.view(len(media_norm), len(PARES), 2)[:, :, 0]).cpu().numpy()
        zs_df = pd.DataFrame(probs, index=media_norm.index, columns=list(PARES.keys()), dtype="float32")
        for j, k_name in enumerate(PARES.keys()):
            zs_df[f"{k_name}_sig"] = sigs[:, j].astype("float32")

        # 2) Image-Text alignment between pet's description and its photo embedding
        descs = df_meta["Description"].fillna("").astype(str).str.strip()
        txt_vecs = []
        with torch.no_grad():
            for k in range(0, len(descs), 128):
                batch_t = descs.iloc[k:k + 128].tolist()
                t_in = full_proc(text=batch_t, padding="max_length", truncation=True, return_tensors="pt").to(dev)
                tv = torch.nn.functional.normalize(_to_tensor(full_net.get_text_features(**t_in)).float(), dim=-1)
                txt_vecs.append(tv.cpu().numpy())
        T_mat = np.concatenate(txt_vecs).astype("float32")
        T_df = pd.DataFrame(T_mat, index=df_meta.index)
        common_m = media_norm.index.intersection(df_meta.index)
        sim_mean = (media_norm.loc[common_m].values * T_df.loc[common_m].values).sum(axis=1)
        common_f = primera.index.intersection(df_meta.index)
        sim_first = (primera.loc[common_f].values * T_df.loc[common_f].values).sum(axis=1)
        zs_df = zs_df.reindex(df_meta.index)
        zs_df["siglip_img_txt_sim_mean"] = pd.Series(sim_mean, index=common_m).reindex(df_meta.index).astype("float32")
        zs_df["siglip_img_txt_sim_first"] = pd.Series(sim_first, index=common_f).reindex(df_meta.index).astype("float32")
        zs_df.to_parquet(f"{OUT}/img_siglip_zs_{parte}.parquet")
        print(f"siglip_zs {parte}: {zs_df.shape}", flush=True)

    print(f"{MODELO} {parte} listo: fotos={len(f)} | first={primera.shape} | mean={media_norm.shape} ({time.time() - t_part:.0f}s)", flush=True)

print("LISTO", sorted(os.listdir(OUT)), f"total {time.time() - t0:.0f}s", flush=True)
