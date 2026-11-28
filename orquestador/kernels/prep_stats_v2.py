# Kernel de preprocesamiento CPU: stats completas de TODAS las fotos, texto enriquecido,
# metadata completa de Google Vision (todas las fotos) y sentiment completo de Google NL.
import glob, json, os, re, time
from collections import Counter
import cv2
import numpy as np
import pandas as pd

OUT = "/kaggle/working"
t0 = time.time()


def buscar(archivo):
    for prof in range(0, 5):
        for d in glob.glob("/kaggle/input/" + "*/" * prof):
            if os.path.exists(os.path.join(d, archivo)):
                return d.rstrip("/")
    raise FileNotFoundError(archivo)


D = buscar("train/train.csv")
print("competencia en", D, flush=True)
train = pd.read_csv(f"{D}/train/train.csv").set_index("PetID")
test = pd.read_csv(f"{D}/test/test.csv").set_index("PetID")


# ------------------------------------------------------------------ 1. All-photo image quality stats
def foto_stats(path):
    try:
        fsize = os.path.getsize(path)
    except OSError:
        fsize = np.nan
    im = cv2.imread(path)
    if im is None:
        return None
    h, w = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    # Hasler & Susstrunk colorfulness metric
    b, gr, r = im[:, :, 0].astype(np.float32), im[:, :, 1].astype(np.float32), im[:, :, 2].astype(np.float32)
    rg = r - gr
    yb = 0.5 * (r + gr) - b
    colorfulness = float(np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    # Edge density (fraction of strong Sobel pixels)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge_mean = float(np.hypot(gx, gy).mean())
    return {
        "nitidez": lap,
        "brillo": float(g.mean()),
        "contraste": float(g.std()),
        "ancho": float(w),
        "alto": float(h),
        "aspecto": float(w / h) if h else np.nan,
        "area": float(w * h),
        "fsize": float(fsize),
        "sat_mean": float(hsv[:, :, 1].mean()),
        "sat_std": float(hsv[:, :, 1].std()),
        "colorfulness": colorfulness,
        "edge_mean": edge_mean,
    }


for parte, idx in (("train", train.index), ("test", test.index)):
    fs = sorted(f for f in os.listdir(f"{D}/{parte}_images") if f.endswith(".jpg"))
    rows = []
    for k, fn in enumerate(fs):
        pet_id, rest = fn.rsplit("-", 1)
        n = int(rest[:-4])
        st = foto_stats(f"{D}/{parte}_images/{fn}")
        if st is not None:
            st["PetID"] = pet_id
            st["n"] = n
            rows.append(st)
        if k % 15000 == 0:
            print(f"img_stats {parte}: {k}/{len(fs)} ({time.time() - t0:.0f}s)", flush=True)
    df_img = pd.DataFrame(rows)
    # Photo 1 (exact EXP-019 names + new photo-1 quality features)
    p1 = df_img[df_img["n"] == 1].set_index("PetID")
    out_img = pd.DataFrame(index=idx)
    out_img["img_nitidez"] = p1["nitidez"].reindex(idx)
    out_img["img_brillo"] = p1["brillo"].reindex(idx)
    out_img["img_contraste"] = p1["contraste"].reindex(idx)
    out_img["img_ancho"] = p1["ancho"].reindex(idx)
    out_img["img_alto"] = p1["alto"].reindex(idx)
    out_img["img_aspecto"] = p1["aspecto"].reindex(idx)
    out_img["img_p1_area"] = p1["area"].reindex(idx)
    out_img["img_p1_fsize"] = p1["fsize"].reindex(idx)
    out_img["img_p1_sat_mean"] = p1["sat_mean"].reindex(idx)
    out_img["img_p1_colorfulness"] = p1["colorfulness"].reindex(idx)
    out_img["img_p1_edge_mean"] = p1["edge_mean"].reindex(idx)

    # All-photo aggregations
    grp = df_img.groupby("PetID")
    out_img["img_n_files"] = grp["n"].count().reindex(idx).fillna(0).astype("float32")
    for col in ("nitidez", "brillo", "contraste", "ancho", "alto", "aspecto", "area", "fsize", "sat_mean", "colorfulness", "edge_mean"):
        out_img[f"img_all_{col}_mean"] = grp[col].mean().reindex(idx)
        out_img[f"img_all_{col}_std"] = grp[col].std().reindex(idx)
        out_img[f"img_all_{col}_max"] = grp[col].max().reindex(idx)
        out_img[f"img_all_{col}_min"] = grp[col].min().reindex(idx)
    out_img = out_img.astype("float32")
    out_img.to_parquet(f"{OUT}/img_stats_v2_{parte}.parquet")
    # Also save exact EXP-019 subset for backward compatibility with modelo.py
    out_img[["img_nitidez", "img_brillo", "img_contraste", "img_ancho", "img_alto", "img_aspecto"]].to_parquet(
        f"{OUT}/img_stats_{parte}.parquet"
    )
    print(f"img_stats_v2 {parte}: {out_img.shape} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ 2. Rich text stats
MALAYO = {
    "dan", "yang", "untuk", "ini", "itu", "boleh", "tidak", "saya", "dia", "ada", "sangat", "anak",
    "kucing", "anjing", "dengan", "nak", "jantan", "betina", "sudah", "sihat", "comel", "manja", "akan",
    "lagi", "sila", "hubungi", "percuma", "mahu", "kami", "dari", "pada", "dalam", "tak", "dah",
}


def texto_stats(t):
    letras = sum(c.isalpha() for c in t)
    digitos = sum(c.isdigit() for c in t)
    palabras = re.findall(r"[A-Za-z]+", t)
    ws = t.split()
    tl = t.lower()
    return {
        # Exact EXP-019 8 columns
        "txt_largo": float(len(t)),
        "txt_palabras": float(len(ws)),
        "txt_frac_chino": len(re.findall(r"[\u4e00-\u9fff]", t)) / len(t) if t else np.nan,
        "txt_frac_malayo": sum(w.lower() in MALAYO for w in palabras) / len(palabras) if palabras else np.nan,
        "txt_frac_mayus": sum(c.isupper() for c in t) / letras if letras else np.nan,
        "txt_exclamaciones": float(t.count("!")),
        "txt_telefono": float(bool(re.search(r"\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}", t))),
        "txt_precio": float(bool(re.search(r"\b(rm\s?\d+|fee|fees|ringgit)\b|\$\s?\d+", t, re.I))),
        # New rich domain/structural text stats
        "txt_vacia": float(len(t) == 0),
        "txt_digitos": float(digitos),
        "txt_frac_digitos": digitos / len(t) if t else np.nan,
        "txt_preguntas": float(t.count("?")),
        "txt_puntos": float(t.count(".")),
        "txt_comas": float(t.count(",")),
        "txt_lineas": float(t.count("\n") + (1 if t else 0)),
        "txt_avg_word_len": float(np.mean([len(w) for w in ws])) if ws else np.nan,
        "txt_unique_word_ratio": len(set(w.lower() for w in ws)) / len(ws) if ws else np.nan,
        "txt_kw_whatsapp": float(bool(re.search(r"what\s*s?\s*a*p+|sms|call|contact|pm\b|wechat|email|hubungi", tl))),
        "txt_kw_urgent": float(bool(re.search(r"urgent|asap|immediately|emergency|desperate|moving|relocat|leav", tl))),
        "txt_kw_free": float(bool(re.search(r"\bfree\b|percuma|no fee|foc\b", tl))),
        "txt_kw_vaccine": float(bool(re.search(r"vaccin|deworm|spay|neuter|steril|vet|health|sihat|vaksin", tl))),
        "txt_kw_friendly": float(bool(re.search(r"friendly|playful|sweet|gentle|active|manja|comel|cute|adorable|loving|affectionate|smart|obedient", tl))),
        "txt_kw_trained": float(bool(re.search(r"toilet|litter|house\s*train|potty|pee\s*pad|paper\s*train", tl))),
        "txt_kw_indoor": float(bool(re.search(r"indoor|house|apartment|condo|home", tl))),
        "txt_kw_stray": float(bool(re.search(r"stray|rescued|found|abandon|shelter|dump|street|roadside", tl))),
        "txt_kw_sick": float(bool(re.search(r"sick|injur|blind|deaf|parvo|flu|fungus|mange|wound|disabled|handicap", tl))),
        "txt_kw_litter": float(bool(re.search(r"litter|siblings|brothers|sisters|puppies|kittens|babies|anak", tl))),
        "txt_kw_adopt_cond": float(bool(re.search(r"serious adopter|responsible|commitment|agreement|visit|update|screening|keep in touch", tl))),
        "txt_has_url": float(bool(re.search(r"http|www\.|facebook|fb\.com|instagram|insta", tl))),
    }


desc = pd.concat([train["Description"], test["Description"]]).fillna("").astype(str).str.strip()
st = pd.DataFrame([texto_stats(t) for t in desc], index=desc.index).astype("float32")
st.loc[train.index].to_parquet(f"{OUT}/txt_stats_v2_train.parquet")
st.loc[test.index].to_parquet(f"{OUT}/txt_stats_v2_test.parquet")
exp019_cols = ["txt_largo", "txt_palabras", "txt_frac_chino", "txt_frac_malayo",
               "txt_frac_mayus", "txt_exclamaciones", "txt_telefono", "txt_precio"]
st.loc[train.index, exp019_cols].to_parquet(f"{OUT}/txt_stats_train.parquet")
st.loc[test.index, exp019_cols].to_parquet(f"{OUT}/txt_stats_test.parquet")
print(f"txt_stats_v2: {st.shape} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ 3. Google NL Sentiment JSON (full)
def parse_sentiment_dir(folder, idx):
    rows = []
    ent_texts = []
    for pet_id in idx:
        p = f"{folder}/{pet_id}.json"
        if not os.path.exists(p):
            rows.append({"sent_has_file": 0.0})
            ent_texts.append("")
            continue
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            rows.append({"sent_has_file": 0.0})
            ent_texts.append("")
            continue
        ds = j.get("documentSentiment", {}) or {}
        doc_score = float(ds.get("score", np.nan))
        doc_mag = float(ds.get("magnitude", np.nan))
        sents = j.get("sentences", []) or []
        s_scores = [float(s.get("sentiment", {}).get("score", 0.0)) for s in sents]
        s_mags = [float(s.get("sentiment", {}).get("magnitude", 0.0)) for s in sents]
        ents = j.get("entities", []) or []
        ent_names = [str(e.get("name", "")).strip() for e in ents if e.get("name")]
        ent_sals = [float(e.get("salience", 0.0)) for e in ents]
        ent_types = Counter(str(e.get("type", "")) for e in ents)
        lang = str(j.get("language", "unknown")).lower()
        rows.append({
            "sent_has_file": 1.0,
            "sent_doc_score": doc_score,
            "sent_doc_mag": doc_mag,
            "sent_score_x_mag": doc_score * doc_mag if np.isfinite(doc_score) and np.isfinite(doc_mag) else np.nan,
            "sent_n_sentences": float(len(sents)),
            "sent_mean_score": float(np.mean(s_scores)) if s_scores else np.nan,
            "sent_std_score": float(np.std(s_scores)) if len(s_scores) > 1 else np.nan,
            "sent_min_score": float(np.min(s_scores)) if s_scores else np.nan,
            "sent_max_score": float(np.max(s_scores)) if s_scores else np.nan,
            "sent_first_score": s_scores[0] if s_scores else np.nan,
            "sent_last_score": s_scores[-1] if s_scores else np.nan,
            "sent_mean_mag": float(np.mean(s_mags)) if s_mags else np.nan,
            "sent_max_mag": float(np.max(s_mags)) if s_mags else np.nan,
            "sent_n_entities": float(len(ents)),
            "sent_ent_salience_mean": float(np.mean(ent_sals)) if ent_sals else np.nan,
            "sent_ent_salience_max": float(np.max(ent_sals)) if ent_sals else np.nan,
            "sent_ent_person": float(ent_types.get("PERSON", 0)),
            "sent_ent_location": float(ent_types.get("LOCATION", 0)),
            "sent_ent_org": float(ent_types.get("ORGANIZATION", 0)),
            "sent_ent_other": float(ent_types.get("OTHER", 0)),
            "sent_lang_en": float(lang.startswith("en")),
            "sent_lang_zh": float(lang.startswith("zh")),
            "sent_lang_ms": float(lang in ("ms", "id")),
        })
        ent_texts.append(" ".join(ent_names))
    df_num = pd.DataFrame(rows, index=idx).astype("float32")
    df_txt = pd.DataFrame({"sent_entities_text": ent_texts}, index=idx)
    return df_num, df_txt


for parte, idx in (("train", train.index), ("test", test.index)):
    s_num, s_txt = parse_sentiment_dir(f"{D}/{parte}_sentiment", idx)
    s_num.to_parquet(f"{OUT}/sentiment_num_{parte}.parquet")
    s_txt.to_parquet(f"{OUT}/sentiment_txt_{parte}.parquet")
    print(f"sentiment {parte}: {s_num.shape} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ 4. Google Vision Metadata JSON (ALL photos!)
def parse_metadata_dir(folder, idx):
    files_by_pet = {}
    for fn in os.listdir(folder):
        if fn.endswith(".json"):
            pet_id = fn.rsplit("-", 1)[0]
            files_by_pet.setdefault(pet_id, []).append(f"{folder}/{fn}")
    rows = []
    label_texts = []
    for k, pet_id in enumerate(idx):
        paths = sorted(files_by_pet.get(pet_id, []))
        if not paths:
            rows.append({"meta_n_files": 0.0})
            label_texts.append("")
            continue
        all_scores, all_topicality, top1_scores = [], [], []
        crop_confs, crop_imps, crop_areas = [], [], []
        dom_scores, dom_pixs, dom_rs, dom_gs, dom_bs, n_colors = [], [], [], [], [], []
        labels = []
        n_labels_per_img = []
        for p in paths:
            try:
                j = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            ann = j.get("labelAnnotations", []) or []
            n_labels_per_img.append(len(ann))
            if ann:
                top1_scores.append(float(ann[0].get("score", np.nan)))
                for a in ann:
                    if "score" in a:
                        all_scores.append(float(a["score"]))
                    if "topicality" in a:
                        all_topicality.append(float(a["topicality"]))
                    d_str = str(a.get("description", "")).strip()
                    if d_str:
                        labels.append(d_str.replace(" ", "_"))
            ch = (j.get("cropHintsAnnotation", {}) or {}).get("cropHints", []) or []
            if ch:
                c0 = ch[0]
                if "confidence" in c0:
                    crop_confs.append(float(c0["confidence"]))
                if "importanceFraction" in c0:
                    crop_imps.append(float(c0["importanceFraction"]))
                verts = (c0.get("boundingPoly", {}) or {}).get("vertices", []) or []
                if len(verts) >= 3:
                    xs = [v.get("x", 0) for v in verts]
                    ys = [v.get("y", 0) for v in verts]
                    crop_areas.append(float((max(xs) - min(xs)) * (max(ys) - min(ys))))
            cols = ((j.get("imagePropertiesAnnotation", {}) or {}).get("dominantColors", {}) or {}).get("colors", []) or []
            n_colors.append(len(cols))
            if cols:
                c0 = cols[0]
                dom_scores.append(float(c0.get("score", np.nan)))
                dom_pixs.append(float(c0.get("pixelFraction", np.nan)))
                rgb = c0.get("color", {}) or {}
                dom_rs.append(float(rgb.get("red", np.nan)))
                dom_gs.append(float(rgb.get("green", np.nan)))
                dom_bs.append(float(rgb.get("blue", np.nan)))
        rows.append({
            "meta_n_files": float(len(paths)),
            "meta_n_labels_mean": float(np.mean(n_labels_per_img)) if n_labels_per_img else np.nan,
            "meta_n_labels_sum": float(np.sum(n_labels_per_img)) if n_labels_per_img else 0.0,
            "meta_score_mean": float(np.mean(all_scores)) if all_scores else np.nan,
            "meta_score_std": float(np.std(all_scores)) if len(all_scores) > 1 else np.nan,
            "meta_score_max": float(np.max(all_scores)) if all_scores else np.nan,
            "meta_top1_score_mean": float(np.mean(top1_scores)) if top1_scores else np.nan,
            "meta_top1_score_min": float(np.min(top1_scores)) if top1_scores else np.nan,
            "meta_topicality_mean": float(np.mean(all_topicality)) if all_topicality else np.nan,
            "meta_crop_conf_mean": float(np.mean(crop_confs)) if crop_confs else np.nan,
            "meta_crop_conf_max": float(np.max(crop_confs)) if crop_confs else np.nan,
            "meta_crop_imp_mean": float(np.mean(crop_imps)) if crop_imps else np.nan,
            "meta_crop_imp_min": float(np.min(crop_imps)) if crop_imps else np.nan,
            "meta_crop_area_mean": float(np.mean(crop_areas)) if crop_areas else np.nan,
            "meta_dom_score_mean": float(np.nanmean(dom_scores)) if dom_scores else np.nan,
            "meta_dom_score_max": float(np.nanmax(dom_scores)) if dom_scores else np.nan,
            "meta_dom_pixfrac_mean": float(np.nanmean(dom_pixs)) if dom_pixs else np.nan,
            "meta_dom_r_mean": float(np.nanmean(dom_rs)) if dom_rs else np.nan,
            "meta_dom_g_mean": float(np.nanmean(dom_gs)) if dom_gs else np.nan,
            "meta_dom_b_mean": float(np.nanmean(dom_bs)) if dom_bs else np.nan,
            "meta_n_colors_mean": float(np.mean(n_colors)) if n_colors else np.nan,
        })
        label_texts.append(" ".join(labels))
    df_num = pd.DataFrame(rows, index=idx).astype("float32")
    df_txt = pd.DataFrame({"meta_labels_text": label_texts}, index=idx)
    return df_num, df_txt


for parte, idx in (("train", train.index), ("test", test.index)):
    m_num, m_txt = parse_metadata_dir(f"{D}/{parte}_metadata", idx)
    m_num.to_parquet(f"{OUT}/metadata_num_{parte}.parquet")
    m_txt.to_parquet(f"{OUT}/metadata_txt_{parte}.parquet")
    print(f"metadata {parte}: {m_num.shape} ({time.time() - t0:.0f}s)", flush=True)

print("LISTO", sorted(os.listdir(OUT)), f"total {time.time() - t0:.0f}s", flush=True)
