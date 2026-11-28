# SOTA Multi-Modal Tabular + Vision + Text + Stacking Engine for PetFinder Adoption Prediction.
# 100% zero-leakage: all target-dependent features, SVD, TF-IDF, scalers, group stats, and thresholds
# are fit strictly inside each fold on X_train (with nested GroupKFold(5) by RescuerID for OOF meta-features).
import gc, glob, json, os, re, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from lightgbm import LGBMClassifier, LGBMRegressor

try:
    CONFIG
except NameError:
    CONFIG = {
        "exp": "EXP-036",
        "vision_models": ["dinov2-l"],
        "text_embs": ["bgem3", "bgem3_full", "e5l"],
        "use_tfidf": True,
        "use_tabular_sota": True,
        "use_stats_v2": True,
        "use_meta_sent": True,
        "use_siglip_zs": False,
        "use_deberta": [],
        "models": ["lgbm_rmse"],
        "n_svd_img": 24,
        "n_svd_txt": 12,
        "n_folds_sub": 10,
        "seeds": [42, 100, 2024],
        "keep_rescuer_id": False,
        "sub_cv": "sgkf",
    }

OUT = "/kaggle/working"
t0 = time.time()
print("CONFIG:", json.dumps(CONFIG, indent=1), flush=True)


def buscar(archivo, required=True):
    for prof in range(0, 6):
        for d in glob.glob("/kaggle/input/" + "*/" * prof):
            p = os.path.join(d, archivo)
            if os.path.exists(p):
                return p
    if required:
        raise FileNotFoundError(f"{archivo} not in /kaggle/input: {sorted(glob.glob('/kaggle/input/*/*'))[:60]}")
    return None


D = os.path.dirname(os.path.dirname(buscar("train/train.csv")))
train_raw = pd.read_csv(f"{D}/train/train.csv").set_index("PetID")
test_raw = pd.read_csv(f"{D}/test/test.csv").set_index("PetID")
breeds_df = pd.read_csv(f"{D}/breed_labels.csv")
states_df = pd.read_csv(f"{D}/state_labels.csv")
breed_name_map = breeds_df.set_index("BreedID")["BreedName"].to_dict()
state_name_map = states_df.set_index("StateID")["StateName"].to_dict()

y_all = train_raw["AdoptionSpeed"].astype(int)
groups_all = train_raw["RescuerID"].astype(str)


# ------------------------------------------------------------------ Malaysia State Demographics (Wikipedia / Kaggle #78040)
# Population (2018-2020), GDP per capita (MYR), Area (km^2)
MALAYSIA_DEMO = {
    "selangor": (6528400, 54995, 8104),
    "kuala lumpur": (1790000, 121300, 243),
    "pulau pinang": (1766800, 55243, 1048),
    "johor": (3742200, 37342, 19210),
    "perak": (2503500, 31668, 21035),
    "negeri sembilan": (1128900, 44450, 6686),
    "melaka": (922400, 49172, 1664),
    "kedah": (2163000, 22412, 9500),
    "pahang": (1664700, 36185, 36137),
    "terengganu": (1228300, 30901, 13035),
    "kelantan": (1856800, 14300, 15099),
    "sabah": (3898400, 25861, 73631),
    "sarawak": (2791700, 53358, 124450),
    "perlis": (253800, 24004, 821),
    "labuan": (99200, 77798, 91),
    "putrajaya": (90000, 121300, 49),
}


TYPOS = {
    "whatapp": "whatsapp", "whatssapp": "whatsapp", "whatssap": "whatsapp", "whatspp": "whatsapp",
    "whastapp": "whatsapp", "whatsap": "whatsapp", "whassap": "whatsapp", "watapps": "whatsapp",
    "watsapp": "whatsapp", "watsap": "whatsapp", "whatapps": "whatsapp",
    "sherpherd": "shepherd", "sherphed": "shepherd", "sherperd": "shepherd", "sherpard": "shepherd",
    "vacinated": "vaccinated", "vacinnated": "vaccinated", "deworm": "dewormed",
    "spay": "spayed", "neuter": "neutered", "sterilise": "sterilized", "sterilised": "sterilized",
}
TYPO_RE = re.compile(r"\b(" + "|".join(TYPOS.keys()) + r")\b", re.I)


def clean_desc(s):
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"(\d+)([A-Za-z]+)", r"\1 \2", s)
    s = TYPO_RE.sub(lambda m: TYPOS[m.group(0).lower()], s)
    return s


# ------------------------------------------------------------------ Base & Tabular Feature Engineering (target-free!)
GENERIC_NAMES = {
    "", "no name", "no name yet", "unnamed", "none", "nil", "puppy", "puppies", "kitten", "kittens",
    "kitty", "cat", "cats", "dog", "dogs", "baby", "babies", "boy", "girl", "male", "female",
    "stray", "rescued", "adopted", "cute", "lovely", "sweet", "doggy", "doggie", "pussy",
}


def build_base_frame(raw_df, parte):
    df = raw_df.copy()
    if "AdoptionSpeed" in df.columns:
        df = df.drop(columns=["AdoptionSpeed"])

    # Load catedra 7 emotions
    p_emo = buscar(f"{parte}.parquet", required=False)
    if p_emo:
        df = df.join(pd.read_parquet(p_emo))

    # Load stats (v2 or v1, gracefully optional if not attached yet)
    p_is2 = buscar(f"img_stats_v2_{parte}.parquet", required=False)
    p_is1 = buscar(f"img_stats_{parte}.parquet", required=False)
    if CONFIG.get("use_stats_v2", True) and p_is2:
        df = df.join(pd.read_parquet(p_is2))
    elif p_is1:
        df = df.join(pd.read_parquet(p_is1))

    p_ts2 = buscar(f"txt_stats_v2_{parte}.parquet", required=False)
    p_ts1 = buscar(f"txt_stats_{parte}.parquet", required=False)
    if CONFIG.get("use_stats_v2", True) and p_ts2:
        df = df.join(pd.read_parquet(p_ts2))
    elif p_ts1:
        df = df.join(pd.read_parquet(p_ts1))
    else:
        # Fast inline text stats (~1s) when stats kernel isn't attached yet
        MALAYO = {"dan", "yang", "untuk", "ini", "itu", "boleh", "tidak", "saya", "dia", "ada", "sangat", "anak",
                  "kucing", "anjing", "dengan", "nak", "jantan", "betina", "sudah", "sihat", "comel", "manja"}
        descs_s = df["Description"].fillna("").astype(str).str.strip()
        rows_t = []
        for t in descs_s:
            letras = sum(c.isalpha() for c in t)
            palabras = re.findall(r"[A-Za-z]+", t)
            tl = t.lower()
            rows_t.append({
                "txt_largo": float(len(t)),
                "txt_palabras": float(len(t.split())),
                "txt_frac_chino": len(re.findall(r"[\u4e00-\u9fff]", t)) / len(t) if t else np.nan,
                "txt_frac_malayo": sum(w.lower() in MALAYO for w in palabras) / len(palabras) if palabras else np.nan,
                "txt_frac_mayus": sum(c.isupper() for c in t) / letras if letras else np.nan,
                "txt_exclamaciones": float(t.count("!")),
                "txt_telefono": float(bool(re.search(r"\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}", t))),
                "txt_precio": float(bool(re.search(r"\b(rm\s?\d+|fee|fees|ringgit)\b|\$\s?\d+", t, re.I))),
                "txt_kw_whatsapp": float(bool(re.search(r"what\s*s?\s*a*p+|sms|call|contact|wechat|hubungi", tl))),
                "txt_kw_urgent": float(bool(re.search(r"urgent|asap|immediately|emergency|desperate|moving", tl))),
                "txt_kw_free": float(bool(re.search(r"\bfree\b|percuma|no fee|foc\b", tl))),
                "txt_kw_vaccine": float(bool(re.search(r"vaccin|deworm|spay|neuter|steril|vet|sihat|vaksin", tl))),
                "txt_kw_friendly": float(bool(re.search(r"friendly|playful|sweet|gentle|manja|comel|cute|adorable", tl))),
                "txt_kw_stray": float(bool(re.search(r"stray|rescued|found|abandon|shelter|dump|street", tl))),
                "txt_kw_adopt_cond": float(bool(re.search(r"serious adopter|responsible|commitment|agreement|visit|screening", tl))),
            })
        df = df.join(pd.DataFrame(rows_t, index=df.index, dtype="float32"))

    # Load Vision metadata + NL Sentiment numerical stats
    if CONFIG.get("use_meta_sent", True):
        for fname in (f"metadata_num_{parte}.parquet", f"sentiment_num_{parte}.parquet"):
            pf = buscar(fname, required=False)
            if pf:
                df = df.join(pd.read_parquet(pf))

    # Load vision extra diversity stats + SigLIP zero-shot/alignment
    for vm in CONFIG.get("vision_models", []):
        pe = buscar(f"img_{vm}_extra_{parte}.parquet", required=False)
        if pe:
            df = df.join(pd.read_parquet(pe))
    if CONFIG.get("use_siglip_zs", False):
        pz = buscar(f"img_siglip_zs_{parte}.parquet", required=False)
        if pz:
            df = df.join(pd.read_parquet(pz))

    # RescuerID count (always computed within dataset, zero leakage)
    resc = df["RescuerID"].astype(str)
    vc_resc = resc.value_counts()
    df["rescuer_count"] = resc.map(vc_resc).astype("float32")

    if CONFIG.get("use_tabular_sota", True):
        # 1) Malaysia state demographics
        st_names = df["State"].map(state_name_map).fillna("").str.lower()
        df["state_pop"] = st_names.map(lambda s: MALAYSIA_DEMO.get(s, (np.nan, np.nan, np.nan))[0]).astype("float32")
        df["state_gdp_pc"] = st_names.map(lambda s: MALAYSIA_DEMO.get(s, (np.nan, np.nan, np.nan))[1]).astype("float32")
        df["state_area"] = st_names.map(lambda s: MALAYSIA_DEMO.get(s, (np.nan, np.nan, np.nan))[2]).astype("float32")
        df["state_density"] = (df["state_pop"] / df["state_area"].clip(lower=1.0)).astype("float32")
        df["state_gdp_total"] = (df["state_pop"] * df["state_gdp_pc"] / 1e9).astype("float32")

        # 2) Breed features
        b1_name = df["Breed1"].map(breed_name_map).fillna("").str.lower()
        b2_name = df["Breed2"].map(breed_name_map).fillna("").str.lower()
        df["has_breed2"] = ((df["Breed2"] > 0) & (df["Breed2"] != df["Breed1"])).astype("float32")
        df["same_breed12"] = ((df["Breed2"] > 0) & (df["Breed2"] == df["Breed1"])).astype("float32")
        is_common_mix = b1_name.str.contains("mixed|domestic short|domestic medium|domestic long", regex=True)
        df["is_mixed_breed"] = (is_common_mix | (df["has_breed2"] > 0)).astype("float32")
        df["is_pure_breed"] = (1.0 - df["is_mixed_breed"]).astype("float32")
        df["breed1_is_dsh"] = (b1_name == "domestic short hair").astype("float32")
        df["breed1_is_mix_dog"] = (b1_name == "mixed breed").astype("float32")

        # 3) Age, Fee, Quantity, Color, Media interactions
        age = df["Age"].astype("float32")
        qty = df["Quantity"].astype("float32").clip(lower=1.0)
        fee = df["Fee"].astype("float32")
        df["age_log1p"] = np.log1p(age.clip(lower=0))
        df["age_years"] = np.floor(age / 12.0)
        df["age_mod12"] = age % 12.0
        df["age_is_exact_yr"] = ((age >= 12) & (age % 12 == 0)).astype("float32")
        df["age_is_baby"] = (age <= 3).astype("float32")
        df["age_is_young"] = ((age > 3) & (age <= 12)).astype("float32")
        df["age_over_1y"] = (age >= 12).astype("float32")
        df["age_over_3y"] = (age >= 36).astype("float32")
        df["age_x_qty"] = age * qty
        df["age_div_qty"] = age / qty
        df["fee_is_free"] = (fee == 0).astype("float32")
        df["fee_log1p"] = np.log1p(fee.clip(lower=0))
        df["fee_per_pet"] = fee / qty
        df["fee_per_age"] = fee / (age + 1.0)
        df["photo_per_qty"] = df["PhotoAmt"].astype("float32") / qty
        df["video_per_qty"] = df["VideoAmt"].astype("float32") / qty
        df["has_photo"] = (df["PhotoAmt"] > 0).astype("float32")
        df["has_video"] = (df["VideoAmt"] > 0).astype("float32")
        df["num_colors"] = ((df["Color1"] > 0).astype(int) + (df["Color2"] > 0).astype(int) + (df["Color3"] > 0).astype(int)).astype("float32")
        df["is_single_pet"] = (df["Quantity"] == 1).astype("float32")
        if "img_n_files" in df.columns:
            df["photo_amt_diff"] = df["PhotoAmt"].astype("float32") - df["img_n_files"].astype("float32")

        # 4) Health / Care composite scores (1=Yes, 2=No, 3=Not Sure)
        care_cols = ["Vaccinated", "Dewormed", "Sterilized"]
        df["care_yes_cnt"] = sum((df[c] == 1).astype("float32") for c in care_cols)
        df["care_no_cnt"] = sum((df[c] == 2).astype("float32") for c in care_cols)
        df["care_unsure_cnt"] = sum((df[c] == 3).astype("float32") for c in care_cols)
        df["care_all_yes"] = (df["care_yes_cnt"] == 3).astype("float32")
        df["care_combo"] = (df["Vaccinated"] * 16 + df["Dewormed"] * 4 + df["Sterilized"]).astype("float32")
        df["health_is_healthy"] = (df["Health"] == 1).astype("float32")
        df["type_x_gender"] = (df["Type"] * 10 + df["Gender"]).astype("float32")
        df["type_x_size"] = (df["Type"] * 10 + df["MaturitySize"]).astype("float32")
        df["type_x_fur"] = (df["Type"] * 10 + df["FurLength"]).astype("float32")
        df["type_x_pure"] = (df["Type"] * 2 + df["is_pure_breed"]).astype("float32")
        df["age_x_size"] = age * df["MaturitySize"].astype("float32")
        df["age_x_care_yes"] = age * df["care_yes_cnt"]
        if "txt_frac_malayo" in df.columns:
            # Cultural interaction: Malay language + Dog (13th place insight!)
            df["malay_x_dog"] = df["txt_frac_malayo"].fillna(0) * (df["Type"] == 1).astype("float32")
        if "sent_lang_ms" in df.columns:
            df["sent_ms_x_dog"] = df["sent_lang_ms"].fillna(0) * (df["Type"] == 1).astype("float32")

        # 5) Name features
        names = df["Name"].fillna("").astype(str).str.strip()
        names_l = names.str.lower()
        df["name_len"] = names.str.len().astype("float32")
        df["name_n_words"] = names.str.split().str.len().astype("float32")
        df["name_missing"] = (names == "").astype("float32")
        df["name_is_generic"] = names_l.isin(GENERIC_NAMES).astype("float32")
        df["name_has_digit"] = names.str.contains(r"\d", regex=True).astype("float32")
        df["name_is_upper"] = ((names.str.len() >= 2) & (names == names.str.upper())).astype("float32")
        df["name_has_and"] = names_l.str.contains(r"\band\b|&|/|,", regex=True).astype("float32")

        # 6) RescuerID group aggregations (within dataset only, zero target leakage!)
        df["rescuer_count_log1p"] = np.log1p(df["rescuer_count"]).astype("float32")
        # Binned rescuer count (5th place Minixhofer feature: shelter tier!)
        df["rescuer_tier"] = pd.cut(
            df["rescuer_count"], bins=[0, 1, 2, 4, 8, 15, 30, 60, 150, 10000], labels=False
        ).astype("float32")
        g_resc = df.groupby(resc)
        df["resc_nunique_type"] = resc.map(g_resc["Type"].nunique()).astype("float32")
        df["resc_nunique_breed1"] = resc.map(g_resc["Breed1"].nunique()).astype("float32")
        df["resc_nunique_state"] = resc.map(g_resc["State"].nunique()).astype("float32")
        df["resc_dog_ratio"] = resc.map((df["Type"] == 1).groupby(resc).mean()).astype("float32")
        df["resc_pure_ratio"] = resc.map(df["is_pure_breed"].groupby(resc).mean()).astype("float32")
        df["resc_free_ratio"] = resc.map(df["fee_is_free"].groupby(resc).mean()).astype("float32")
        df["resc_breed_div_cnt"] = (df["resc_nunique_breed1"] / df["rescuer_count"].clip(lower=1.0)).astype("float32")

        agg_cols = ["Age", "Fee", "PhotoAmt", "Quantity", "MaturitySize", "Sterilized", "Vaccinated", "care_yes_cnt", "txt_largo"]
        if "img_nitidez" in df.columns:
            agg_cols.append("img_nitidez")
        if "sent_doc_score" in df.columns:
            agg_cols.append("sent_doc_score")
        for col in agg_cols:
            s_col = df[col].astype("float32")
            g = s_col.groupby(resc)
            rmean = resc.map(g.mean()).astype("float32")
            rstd = resc.map(g.std()).astype("float32")
            rmin = resc.map(g.min()).astype("float32")
            rmax = resc.map(g.max()).astype("float32")
            df[f"resc_{col}_mean"] = rmean
            df[f"resc_{col}_std"] = rstd
            if col in ("Age", "Fee", "PhotoAmt", "Quantity", "txt_largo"):
                df[f"resc_{col}_min"] = rmin
                df[f"resc_{col}_max"] = rmax
                df[f"resc_{col}_diff"] = s_col - rmean
                df[f"resc_{col}_ratio"] = s_col / (rmean.abs() + 1.0)

    # Drop raw object text columns (keep categoricals clean)
    cat_cols = [
        "Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
        "FurLength", "MaturitySize", "Vaccinated", "Dewormed", "Sterilized", "Health", "State",
    ]
    if CONFIG.get("keep_rescuer_id", False):
        cat_cols.append("RescuerID")
    for c in cat_cols:
        df[c] = df[c].astype("category")
    drop_obj = [c for c in df.columns if df[c].dtype == "object"]
    df = df.drop(columns=drop_obj)
    return df, cat_cols


print("Building base tabular frames...", flush=True)
X_train_base, CAT_COLS = build_base_frame(train_raw, "train")
X_test_base, _ = build_base_frame(test_raw, "test")
print(f"Base frames: train={X_train_base.shape}, test={X_test_base.shape} ({time.time()-t0:.1f}s)", flush=True)


# ------------------------------------------------------------------ Load External Embedding Matrices (Indexed by PetID)
VISION_MATS = {}  # name -> (tr_df, te_df)
for vm in CONFIG.get("vision_models", []):
    for agg in ("mean", "first"):
        p_tr = buscar(f"img_{vm}_{agg}_train.parquet", required=False)
        p_te = buscar(f"img_{vm}_{agg}_test.parquet", required=False)
        if p_tr and p_te:
            VISION_MATS[f"{vm}_{agg}"] = (
                pd.read_parquet(p_tr).reindex(train_raw.index).astype("float32"),
                pd.read_parquet(p_te).reindex(test_raw.index).astype("float32"),
            )
            print(f"Loaded vision {vm}_{agg}: {VISION_MATS[f'{vm}_{agg}'][0].shape}", flush=True)

TEXT_MATS = {}
for tm in CONFIG.get("text_embs", []):
    p_tr = buscar(f"txt_emb_{tm}_train.parquet", required=False)
    p_te = buscar(f"txt_emb_{tm}_test.parquet", required=False)
    if p_tr and p_te:
        TEXT_MATS[tm] = (
            pd.read_parquet(p_tr).reindex(train_raw.index).astype("float32"),
            pd.read_parquet(p_te).reindex(test_raw.index).astype("float32"),
        )
        print(f"Loaded text emb {tm}: {TEXT_MATS[tm][0].shape}", flush=True)

# Text corpora for in-fold TF-IDF
DESC_TR = train_raw["Description"].map(clean_desc)
DESC_TE = test_raw["Description"].map(clean_desc)

def build_meta_corpus(raw_df, parte):
    parts = [
        raw_df["Name"].fillna("").astype(str),
        raw_df["Breed1"].map(breed_name_map).fillna("").astype(str),
        raw_df["Breed2"].map(breed_name_map).fillna("").astype(str),
    ]
    pm = buscar(f"metadata_txt_{parte}.parquet", required=False)
    if pm:
        parts.append(pd.read_parquet(pm).reindex(raw_df.index)["meta_labels_text"].fillna(""))
    ps = buscar(f"sentiment_txt_{parte}.parquet", required=False)
    if ps:
        parts.append(pd.read_parquet(ps).reindex(raw_df.index)["sent_entities_text"].fillna(""))
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out.str.strip()

META_TXT_TR = build_meta_corpus(train_raw, "train")
META_TXT_TE = build_meta_corpus(test_raw, "test")

# Optional fine-tuned DeBERTa OOF + test (same StratifiedGroupKFold(5, seed=42)!)
DEBERTA_TR_LIST, DEBERTA_TE_LIST = [], []
for dtag in CONFIG.get("use_deberta", []):
    po = buscar(f"{dtag}_oof.parquet", required=False)
    pt = buscar(f"{dtag}_test.parquet", required=False)
    if po and pt:
        DEBERTA_TR_LIST.append(pd.read_parquet(po).reindex(train_raw.index).astype("float32"))
        DEBERTA_TE_LIST.append(pd.read_parquet(pt).reindex(test_raw.index).astype("float32"))
        print(f"Loaded fine-tuned transformer {dtag}", flush=True)


# ------------------------------------------------------------------ In-Fold Leakage-Free Feature Generators
def add_fold_group_stats(Xtr, Xva, Xte):
    """Fit frequency encodings & category group stats strictly on Xtr (zero leakage!)."""
    if not CONFIG.get("use_tabular_sota", True):
        return Xtr, Xva, Xte
    tr_new, va_new, te_new = {}, {}, {}
    # Frequency encodings
    for col in ("Breed1", "Breed2", "Color1", "Color2", "State"):
        vc = Xtr[col].astype(str).value_counts(normalize=True)
        tr_new[f"freq_{col}"] = Xtr[col].astype(str).map(vc).fillna(0).astype("float32").values
        va_new[f"freq_{col}"] = Xva[col].astype(str).map(vc).fillna(0).astype("float32").values
        te_new[f"freq_{col}"] = Xte[col].astype(str).map(vc).fillna(0).astype("float32").values

    # Top-15 1st-place group aggregations + diffs & ratios (fit strictly on Xtr!)
    specs = [
        ("Breed1", ["Age", "Fee", "PhotoAmt", "MaturitySize", "care_yes_cnt"]),
        ("MaturitySize", ["Age", "Fee", "PhotoAmt"]),
        ("State", ["Age", "Fee", "Sterilized", "PhotoAmt", "rescuer_count"]),
        ("Type", ["Age", "Fee", "PhotoAmt", "txt_largo"]),
    ]
    for gcol, vcols in specs:
        gkey_tr = Xtr[gcol].astype(str)
        gkey_va = Xva[gcol].astype(str)
        gkey_te = Xte[gcol].astype(str)
        for vcol in vcols:
            if vcol not in Xtr.columns:
                continue
            s_tr = Xtr[vcol].astype("float32")
            s_va = Xva[vcol].astype("float32")
            s_te = Xte[vcol].astype("float32")
            gmean = s_tr.groupby(gkey_tr).mean()
            gstd = s_tr.groupby(gkey_tr).std()
            glob_m = float(s_tr.mean())
            m_tr = gkey_tr.map(gmean).fillna(glob_m).astype("float32").values
            m_va = gkey_va.map(gmean).fillna(glob_m).astype("float32").values
            m_te = gkey_te.map(gmean).fillna(glob_m).astype("float32").values
            tr_new[f"grp_{gcol}_{vcol}_diff"] = s_tr.values - m_tr
            va_new[f"grp_{gcol}_{vcol}_diff"] = s_va.values - m_va
            te_new[f"grp_{gcol}_{vcol}_diff"] = s_te.values - m_te
            tr_new[f"grp_{gcol}_{vcol}_ratio"] = s_tr.values / (np.abs(m_tr) + 1.0)
            va_new[f"grp_{gcol}_{vcol}_ratio"] = s_va.values / (np.abs(m_va) + 1.0)
            te_new[f"grp_{gcol}_{vcol}_ratio"] = s_te.values / (np.abs(m_te) + 1.0)
            if vcol in ("Age", "Fee", "Sterilized"):
                sd_tr = gkey_tr.map(gstd).fillna(0).astype("float32").values
                sd_va = gkey_va.map(gstd).fillna(0).astype("float32").values
                sd_te = gkey_te.map(gstd).fillna(0).astype("float32").values
                tr_new[f"grp_{gcol}_{vcol}_z"] = (s_tr.values - m_tr) / (sd_tr + 1.0)
                va_new[f"grp_{gcol}_{vcol}_z"] = (s_va.values - m_va) / (sd_va + 1.0)
                te_new[f"grp_{gcol}_{vcol}_z"] = (s_te.values - m_te) / (sd_te + 1.0)

    Xtr = pd.concat([Xtr, pd.DataFrame(tr_new, index=Xtr.index)], axis=1)
    Xva = pd.concat([Xva, pd.DataFrame(va_new, index=Xva.index)], axis=1)
    Xte = pd.concat([Xte, pd.DataFrame(te_new, index=Xte.index)], axis=1)
    return Xtr, Xva, Xte


def fit_nested_linear_oof(M_tr, y_tr, g_tr, M_va, M_te, prefix, alphas=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0), add_cls=True, C_cls=1.0):
    """Nested GroupKFold(5) multi-target Ridge via Cholesky normal equations (150x faster than RidgeCV)."""
    n_tr, n_va, n_te = M_tr.shape[0], M_va.shape[0], M_te.shape[0]
    is_sparse = hasattr(M_tr, "tocsr")
    if not is_sparse:
        ok_tr = np.isfinite(M_tr).all(axis=1)
        ok_va = np.isfinite(M_va).all(axis=1)
        ok_te = np.isfinite(M_te).all(axis=1)
    else:
        ok_tr = np.ones(n_tr, dtype=bool)
        ok_va = np.ones(n_va, dtype=bool)
        ok_te = np.ones(n_te, dtype=bool)

    Y_all = np.column_stack([y_tr.astype(np.float32), np.eye(5, dtype=np.float32)[y_tr]]) if add_cls else y_tr.astype(np.float32)[:, None]
    oof_r = np.full(n_tr, np.nan, dtype=np.float32)
    oof_p = np.full((n_tr, 5), np.nan, dtype=np.float32) if add_cls else None

    def _fit_dense_chol(X_f, Y_f, alpha):
        # Append bias column, solve (X^T X + alpha I) W = X^T Y in O(N d^2 + d^3)
        d = X_f.shape[1]
        mu_x = X_f.mean(axis=0, keepdims=True)
        mu_y = Y_f.mean(axis=0, keepdims=True)
        Xc = X_f - mu_x
        Yc = Y_f - mu_y
        G = (Xc.T @ Xc).astype(np.float64)
        G[np.diag_indices(d)] += float(alpha)
        W = np.linalg.solve(G, (Xc.T @ Yc).astype(np.float64)).astype(np.float32)
        return mu_x, mu_y, W

    def _pred_dense(X_p, params):
        mu_x, mu_y, W = params
        return (X_p - mu_x) @ W + mu_y

    # Pick best alpha once on inner GroupKFold for dense matrices (~0.3s total!)
    best_alpha = 30.0
    gkf = GroupKFold(n_splits=5)
    inner_splits = list(gkf.split(np.arange(n_tr), groups=g_tr))
    if not is_sparse:
        ia0, ib0 = inner_splits[0]
        ia0_ok, ib0_ok = ia0[ok_tr[ia0]], ib0[ok_tr[ib0]]
        if len(ia0_ok) >= 20 and len(ib0_ok) > 0:
            X0, Y0 = M_tr[ia0_ok], Y_all[ia0_ok, :1]
            mu_x0, mu_y0 = X0.mean(axis=0, keepdims=True), Y0.mean(axis=0, keepdims=True)
            Xc0, Yc0 = X0 - mu_x0, Y0 - mu_y0
            G0 = (Xc0.T @ Xc0).astype(np.float64)
            B0 = (Xc0.T @ Yc0).astype(np.float64)
            Xval0 = M_tr[ib0_ok] - mu_x0
            yval0 = y_tr[ib0_ok]
            best_mse = 1e18
            for a in alphas:
                Ga = G0.copy()
                Ga[np.diag_indices(Ga.shape[0])] += float(a)
                Wa = np.linalg.solve(Ga, B0).astype(np.float32)
                pred0 = (Xval0 @ Wa + mu_y0)[:, 0]
                mse0 = float(np.mean((pred0 - yval0) ** 2))
                if mse0 < best_mse:
                    best_mse, best_alpha = mse0, float(a)

    for ia, ib in inner_splits:
        ia_ok = ia[ok_tr[ia]]
        ib_ok = ib[ok_tr[ib]]
        if len(ia_ok) < 20 or len(ib_ok) == 0:
            continue
        if is_sparse:
            reg = Ridge(alpha=5.0).fit(M_tr[ia_ok], Y_all[ia_ok])
            P = reg.predict(M_tr[ib_ok])
        else:
            params = _fit_dense_chol(M_tr[ia_ok], Y_all[ia_ok], best_alpha)
            P = _pred_dense(M_tr[ib_ok], params)
        if P.ndim == 1:
            P = P[:, None]
        oof_r[ib_ok] = P[:, 0]
        if add_cls:
            pp = np.clip(P[:, 1:6], 0.0, 1.0)
            oof_p[ib_ok] = pp / np.maximum(pp.sum(axis=1, keepdims=True), 1e-6)

    # Fit ONCE on full fold train for BOTH va and te
    va_r = np.full(n_va, np.nan, dtype=np.float32)
    te_r = np.full(n_te, np.nan, dtype=np.float32)
    va_p = np.full((n_va, 5), np.nan, dtype=np.float32) if add_cls else None
    te_p = np.full((n_te, 5), np.nan, dtype=np.float32) if add_cls else None

    idx_ok_tr = np.where(ok_tr)[0]
    if is_sparse:
        model_full = Ridge(alpha=5.0).fit(M_tr[idx_ok_tr], Y_all[idx_ok_tr])
        pred_fn = lambda Xp: model_full.predict(Xp)
    else:
        params_full = _fit_dense_chol(M_tr[idx_ok_tr], Y_all[idx_ok_tr], best_alpha)
        pred_fn = lambda Xp: _pred_dense(Xp, params_full)

    if ok_va.any():
        Pv = pred_fn(M_va[ok_va])
        if Pv.ndim == 1:
            Pv = Pv[:, None]
        va_r[ok_va] = Pv[:, 0]
        if add_cls:
            pp = np.clip(Pv[:, 1:6], 0.0, 1.0)
            va_p[ok_va] = pp / np.maximum(pp.sum(axis=1, keepdims=True), 1e-6)
    if ok_te.any():
        Pt = pred_fn(M_te[ok_te])
        if Pt.ndim == 1:
            Pt = Pt[:, None]
        te_r[ok_te] = Pt[:, 0]
        if add_cls:
            pp = np.clip(Pt[:, 1:6], 0.0, 1.0)
            te_p[ok_te] = pp / np.maximum(pp.sum(axis=1, keepdims=True), 1e-6)

    out_tr = {f"{prefix}_ridge": oof_r}
    out_va = {f"{prefix}_ridge": va_r}
    out_te = {f"{prefix}_ridge": te_r}
    if add_cls:
        ev_w = np.arange(5, dtype=np.float32)[None, :]
        out_tr[f"{prefix}_cls_ev"] = np.nansum(oof_p * ev_w, axis=1)
        out_va[f"{prefix}_cls_ev"] = np.nansum(va_p * ev_w, axis=1)
        out_te[f"{prefix}_cls_ev"] = np.nansum(te_p * ev_w, axis=1)
        out_tr[f"{prefix}_cls_ev"][~ok_tr] = np.nan
        out_va[f"{prefix}_cls_ev"][~ok_va] = np.nan
        out_te[f"{prefix}_cls_ev"][~ok_te] = np.nan
        for k in range(5):
            out_tr[f"{prefix}_p{k}"] = oof_p[:, k]
            out_va[f"{prefix}_p{k}"] = va_p[:, k]
            out_te[f"{prefix}_p{k}"] = te_p[:, k]
    return out_tr, out_va, out_te


def fit_loro_knn_oof(M_tr, y_tr, g_tr, M_va, M_te, prefix, ks=(15, 50, 150), extra_vals=None):
    """Leave-One-Rescuer-Out cosine kNN target & visual-attribute features (100% zero leakage, ~1s via BLAS!)."""
    n_tr, n_va, n_te = M_tr.shape[0], M_va.shape[0], M_te.shape[0]
    ok_tr = np.isfinite(M_tr).all(axis=1)
    ok_va = np.isfinite(M_va).all(axis=1)
    ok_te = np.isfinite(M_te).all(axis=1)

    def _l2(X):
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        return (X / np.maximum(nrm, 1e-8)).astype(np.float32)

    idx_ok_tr = np.where(ok_tr)[0]
    V_ref = _l2(M_tr[idx_ok_tr])
    y_ref = y_tr[idx_ok_tr].astype(np.float32)
    g_ref = g_tr[idx_ok_tr]
    y_oh = np.eye(5, dtype=np.float32)[y_tr[idx_ok_tr]]
    age_ref = extra_vals[idx_ok_tr].astype(np.float32) if extra_vals is not None else None

    kmax = max(ks)
    def _query(Q_raw, q_ok, q_groups=None):
        n_q = Q_raw.shape[0]
        res = {}
        for k in ks:
            res[f"{prefix}_knn{k}_ymean"] = np.full(n_q, np.nan, dtype=np.float32)
            res[f"{prefix}_knn{k}_ystd"] = np.full(n_q, np.nan, dtype=np.float32)
            res[f"{prefix}_knn{k}_sim"] = np.full(n_q, np.nan, dtype=np.float32)
        res[f"{prefix}_knn50_p0"] = np.full(n_q, np.nan, dtype=np.float32)
        res[f"{prefix}_knn50_p4"] = np.full(n_q, np.nan, dtype=np.float32)
        if age_ref is not None:
            res[f"{prefix}_knn30_age"] = np.full(n_q, np.nan, dtype=np.float32)

        idx_q = np.where(q_ok)[0]
        if len(idx_q) == 0:
            return res
        VQ = _l2(Q_raw[idx_q])
        # Process in chunks of 1024 to keep RAM tiny
        for s in range(0, len(idx_q), 1024):
            e = min(s + 1024, len(idx_q))
            sims = VQ[s:e] @ V_ref.T  # (chunk, n_ref)
            if q_groups is not None:
                qg = q_groups[idx_q[s:e]]
                same = (qg[:, None] == g_ref[None, :])
                sims[same] = -1e9
            top_idx = np.argpartition(-sims, kmax, axis=1)[:, :kmax]
            top_sim = np.take_along_axis(sims, top_idx, axis=1)
            ord_s = np.argsort(-top_sim, axis=1)
            top_idx = np.take_along_axis(top_idx, ord_s, axis=1)
            top_sim = np.take_along_axis(top_sim, ord_s, axis=1)
            # Softmax-like temperature weights on cosine similarity
            w_all = np.exp((top_sim - top_sim[:, :1]) * 12.0)
            y_nb = y_ref[top_idx]
            sl = idx_q[s:e]
            for k in ks:
                wk = w_all[:, :k]
                wk = wk / np.maximum(wk.sum(axis=1, keepdims=True), 1e-8)
                ym = (wk * y_nb[:, :k]).sum(axis=1)
                yv = np.sqrt(np.maximum((wk * (y_nb[:, :k] - ym[:, None]) ** 2).sum(axis=1), 0.0))
                res[f"{prefix}_knn{k}_ymean"][sl] = ym
                res[f"{prefix}_knn{k}_ystd"][sl] = yv
                res[f"{prefix}_knn{k}_sim"][sl] = top_sim[:, :k].mean(axis=1)
            w50 = w_all[:, :50]
            w50 = w50 / np.maximum(w50.sum(axis=1, keepdims=True), 1e-8)
            res[f"{prefix}_knn50_p0"][sl] = (w50 * y_oh[top_idx[:, :50], 0]).sum(axis=1)
            res[f"{prefix}_knn50_p4"][sl] = (w50 * y_oh[top_idx[:, :50], 4]).sum(axis=1)
            if age_ref is not None:
                w30 = w_all[:, :30]
                w30 = w30 / np.maximum(w30.sum(axis=1, keepdims=True), 1e-8)
                res[f"{prefix}_knn30_age"][sl] = (w30 * age_ref[top_idx[:, :30]]).sum(axis=1)
        return res

    return _query(M_tr, ok_tr, g_tr), _query(M_va, ok_va, None), _query(M_te, ok_te, None)


def add_svd_block(M_tr, M_va, M_te, n_comp, prefix):
    is_sparse = hasattr(M_tr, "tocsr")
    if is_sparse:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        z_tr = svd.fit_transform(M_tr).astype("float32")
        z_va = svd.transform(M_va).astype("float32")
        z_te = svd.transform(M_te).astype("float32")
    else:
        ok_tr = np.isfinite(M_tr).all(axis=1)
        ok_va = np.isfinite(M_va).all(axis=1)
        ok_te = np.isfinite(M_te).all(axis=1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(M_tr[ok_tr])
        z_tr = np.full((M_tr.shape[0], n_comp), np.nan, dtype=np.float32)
        z_va = np.full((M_va.shape[0], n_comp), np.nan, dtype=np.float32)
        z_te = np.full((M_te.shape[0], n_comp), np.nan, dtype=np.float32)
        z_tr[ok_tr] = svd.transform(M_tr[ok_tr])
        if ok_va.any():
            z_va[ok_va] = svd.transform(M_va[ok_va])
        if ok_te.any():
            z_te[ok_te] = svd.transform(M_te[ok_te])
    out_tr = {f"{prefix}_svd_{i}": z_tr[:, i] for i in range(n_comp)}
    out_va = {f"{prefix}_svd_{i}": z_va[:, i] for i in range(n_comp)}
    out_te = {f"{prefix}_svd_{i}": z_te[:, i] for i in range(n_comp)}
    return out_tr, out_va, out_te


def build_fold_features(tr_idx, va_idx):
    idx_tr = train_raw.index[tr_idx]
    idx_va = train_raw.index[va_idx]
    idx_te = test_raw.index
    y_tr = y_all.values[tr_idx]
    g_tr = groups_all.values[tr_idx]

    Xtr = X_train_base.iloc[tr_idx].copy()
    Xva = X_train_base.iloc[va_idx].copy()
    Xte = X_test_base.copy()

    # 1) In-fold category frequency & group aggregations
    Xtr, Xva, Xte = add_fold_group_stats(Xtr, Xva, Xte)

    extra_tr, extra_va, extra_te = {}, {}, {}

    # 2) Vision embeddings: SVD + Nested RidgeCV + Nested 5-class LogReg
    n_svd_img = int(CONFIG.get("n_svd_img", 24))
    for vkey, (Vtr_all, Vte_all) in VISION_MATS.items():
        M_tr = Vtr_all.values[tr_idx]
        M_va = Vtr_all.values[va_idx]
        M_te = Vte_all.values
        if vkey.endswith("_mean"):
            d_tr, d_va, d_te = add_svd_block(M_tr, M_va, M_te, n_svd_img, f"v_{vkey}")
            extra_tr.update(d_tr); extra_va.update(d_va); extra_te.update(d_te)
        # Nested linear probes on BOTH mean and first!
        d_tr, d_va, d_te = fit_nested_linear_oof(
            M_tr, y_tr, g_tr, M_va, M_te, f"v_{vkey}",
            alphas=(3.0, 10.0, 30.0, 100.0, 300.0), add_cls=vkey.endswith("_mean"), C_cls=0.5
        )
        extra_tr.update(d_tr); extra_va.update(d_va); extra_te.update(d_te)

    # Combined multi-backbone linear probe + Leave-One-Rescuer-Out Cosine-kNN manifold features!
    mean_keys = [k for k in VISION_MATS if k.endswith("_mean")]
    if len(mean_keys) >= 2:
        M_tr = np.hstack([np.nan_to_num(VISION_MATS[k][0].values[tr_idx], 0.0) for k in mean_keys])
        M_va = np.hstack([np.nan_to_num(VISION_MATS[k][0].values[va_idx], 0.0) for k in mean_keys])
        M_te = np.hstack([np.nan_to_num(VISION_MATS[k][1].values, 0.0) for k in mean_keys])
        d_tr, d_va, d_te = fit_nested_linear_oof(
            M_tr, y_tr, g_tr, M_va, M_te, "v_joint",
            alphas=(10.0, 30.0, 100.0, 300.0, 1000.0), add_cls=True, C_cls=0.3
        )
        extra_tr.update(d_tr); extra_va.update(d_va); extra_te.update(d_te)
        if CONFIG.get("use_knn", False):
            # LORO kNN on joint vision manifold (strictly masking same RescuerID!)
            age_tr = np.log1p(train_raw["Age"].clip(lower=0).values[tr_idx])
            k_tr, k_va, k_te = fit_loro_knn_oof(M_tr, y_tr, g_tr, M_va, M_te, "v_joint", ks=(15, 50, 150), extra_vals=age_tr)
            extra_tr.update(k_tr); extra_va.update(k_va); extra_te.update(k_te)
            # Also per-backbone LORO kNN on top vision & text embeddings
            for k_single in mean_keys:
                Vtr_s, Vte_s = VISION_MATS[k_single]
                k_tr, k_va, k_te = fit_loro_knn_oof(Vtr_s.values[tr_idx], y_tr, g_tr, Vtr_s.values[va_idx], Vte_s.values, f"v_{k_single}", ks=(20, 80))
                extra_tr.update(k_tr); extra_va.update(k_va); extra_te.update(k_te)
            if "bgem3_full" in TEXT_MATS:
                Ttr_s, Tte_s = TEXT_MATS["bgem3_full"]
                k_tr, k_va, k_te = fit_loro_knn_oof(Ttr_s.values[tr_idx], y_tr, g_tr, Ttr_s.values[va_idx], Tte_s.values, "t_bgem3_full", ks=(20, 80))
                extra_tr.update(k_tr); extra_va.update(k_va); extra_te.update(k_te)

    # 3) Dense Multilingual Text Embeddings (BGE-M3, BGE-M3-Full, E5-Large)
    n_svd_txt = int(CONFIG.get("n_svd_txt", 12))
    for tkey, (Ttr_all, Tte_all) in TEXT_MATS.items():
        M_tr = Ttr_all.values[tr_idx]
        M_va = Ttr_all.values[va_idx]
        M_te = Tte_all.values
        d_tr, d_va, d_te = add_svd_block(M_tr, M_va, M_te, n_svd_txt, f"t_{tkey}")
        extra_tr.update(d_tr); extra_va.update(d_va); extra_te.update(d_te)
        d_tr, d_va, d_te = fit_nested_linear_oof(
            M_tr, y_tr, g_tr, M_va, M_te, f"t_{tkey}",
            alphas=(1.0, 3.0, 10.0, 30.0, 100.0), add_cls=(tkey == "bgem3_full"), C_cls=0.5
        )
        extra_tr.update(d_tr); extra_va.update(d_va); extra_te.update(d_te)

    # 4) Multi-View Sparse TF-IDF (Description word+char + Vision/Sentiment/Breed meta text)
    if CONFIG.get("use_tfidf", True):
        # Word TF-IDF on Description
        vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        Mw_tr = vec_w.fit_transform(DESC_TR.iloc[tr_idx])
        Mw_va = vec_w.transform(DESC_TR.iloc[va_idx])
        Mw_te = vec_w.transform(DESC_TE)
        d_tr, d_va, d_te = fit_nested_linear_oof(Mw_tr, y_tr, g_tr, Mw_va, Mw_te, "tfidf_desc_w", alphas=(1.0, 3.0, 10.0, 30.0), add_cls=True, C_cls=0.5)
        extra_tr.update(d_tr); extra_va.update(d_va); extra_te.update(d_te)
        d_tr, d_va, d_te = add_svd_block(Mw_tr, Mw_va, Mw_te, 16, "tfidf_desc_w")
        extra_tr.update(d_tr); extra_va.update(d_va); extra_te.update(d_te)

        # Char TF-IDF on Description (captures Malay/Chinese/typos/phone/RM!)
        vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=30000, sublinear_tf=True)
        Mc_tr = vec_c.fit_transform(DESC_TR.iloc[tr_idx])
        Mc_va = vec_c.transform(DESC_TR.iloc[va_idx])
        Mc_te = vec_c.transform(DESC_TE)
        d_tr, d_va, d_te = fit_nested_linear_oof(Mc_tr, y_tr, g_tr, Mc_va, Mc_te, "tfidf_desc_c", alphas=(1.0, 3.0, 10.0, 30.0), add_cls=False)
        extra_tr.update(d_tr); extra_va.update(d_va); extra_te.update(d_te)

        # Word TF-IDF on Vision Labels + Sentiment Entities + Breed + Name
        vec_m = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=15000, sublinear_tf=True)
        Mm_tr = vec_m.fit_transform(META_TXT_TR.iloc[tr_idx])
        Mm_va = vec_m.transform(META_TXT_TR.iloc[va_idx])
        Mm_te = vec_m.transform(META_TXT_TE)
        d_tr, d_va, d_te = fit_nested_linear_oof(Mm_tr, y_tr, g_tr, Mm_va, Mm_te, "tfidf_meta", alphas=(1.0, 3.0, 10.0, 30.0), add_cls=False)
        extra_tr.update(d_tr); extra_va.update(d_va); extra_te.update(d_te)
        d_tr, d_va, d_te = add_svd_block(Mm_tr, Mm_va, Mm_te, 12, "tfidf_meta")
        extra_tr.update(d_tr); extra_va.update(d_va); extra_te.update(d_te)

    # 5) Optional fine-tuned DeBERTa OOF (ONLY valid for exact 5-fold SGKF seed=42!)
    if DEBERTA_TR_LIST and CONFIG.get("_allow_deberta_in_fold", False):
        for df_dtr, df_dte in zip(DEBERTA_TR_LIST, DEBERTA_TE_LIST):
            for c in df_dtr.columns:
                extra_tr[c] = df_dtr[c].values[tr_idx]
                extra_va[c] = df_dtr[c].values[va_idx]
                extra_te[c] = df_dte[c].values

    Xtr = pd.concat([Xtr, pd.DataFrame(extra_tr, index=idx_tr, dtype="float32")], axis=1)
    Xva = pd.concat([Xva, pd.DataFrame(extra_va, index=idx_va, dtype="float32")], axis=1)
    Xte = pd.concat([Xte, pd.DataFrame(extra_te, index=idx_te, dtype="float32")], axis=1)
    return Xtr, Xva, Xte


# ------------------------------------------------------------------ Model Families
def fit_predict_model(mname, Xtr, ytr, Xva, Xte, seeds):
    va_preds, te_preds = [], []
    if mname == "lgbm_rmse":
        for s in seeds:
            m = LGBMRegressor(
                n_estimators=550, learning_rate=0.02, num_leaves=23, min_child_samples=35,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.45,
                reg_alpha=1.0, reg_lambda=5.0, random_state=s, n_jobs=-1, verbose=-1
            )
            m.fit(Xtr, ytr)
            va_preds.append(m.predict(Xva))
            te_preds.append(m.predict(Xte))
    elif mname == "lgbm_deep":
        for s in seeds:
            m = LGBMRegressor(
                n_estimators=650, learning_rate=0.015, num_leaves=31, min_child_samples=50,
                subsample=0.75, subsample_freq=1, colsample_bytree=0.35,
                reg_alpha=2.0, reg_lambda=10.0, random_state=s + 7, n_jobs=-1, verbose=-1
            )
            m.fit(Xtr, ytr)
            va_preds.append(m.predict(Xva))
            te_preds.append(m.predict(Xte))
    elif mname == "lgbm_huber":
        for s in seeds:
            m = LGBMRegressor(
                objective="huber", alpha=1.2,
                n_estimators=600, learning_rate=0.02, num_leaves=23, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.45,
                reg_alpha=1.0, reg_lambda=5.0, random_state=s + 13, n_jobs=-1, verbose=-1
            )
            m.fit(Xtr, ytr)
            va_preds.append(m.predict(Xva))
            te_preds.append(m.predict(Xte))
    elif mname == "lgbm_cls":
        ev_w = np.arange(5, dtype=np.float64)[None, :]
        for s in seeds:
            m = LGBMClassifier(
                objective="multiclass", num_class=5,
                n_estimators=550, learning_rate=0.02, num_leaves=23, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.45,
                reg_alpha=1.0, reg_lambda=6.0, random_state=s + 19, n_jobs=-1, verbose=-1
            )
            m.fit(Xtr, ytr)
            va_preds.append((m.predict_proba(Xva) * ev_w).sum(axis=1))
            te_preds.append((m.predict_proba(Xte) * ev_w).sum(axis=1))
    elif mname == "lgbm_ordinal":
        # Cumulative ordinal probabilities: E[y] = sum_{k=1..4} P(y >= k)
        for s in seeds:
            pv_sum = np.zeros(len(Xva), dtype=np.float64)
            pt_sum = np.zeros(len(Xte), dtype=np.float64)
            for thresh in (1, 2, 3, 4):
                yb = (ytr >= thresh).astype(int)
                m = LGBMClassifier(
                    objective="binary",
                    n_estimators=550, learning_rate=0.02, num_leaves=23, min_child_samples=40,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.45,
                    reg_alpha=1.0, reg_lambda=6.0, random_state=s + 31 * thresh, n_jobs=-1, verbose=-1
                )
                m.fit(Xtr, yb)
                pv_sum += m.predict_proba(Xva)[:, 1]
                pt_sum += m.predict_proba(Xte)[:, 1]
            va_preds.append(pv_sum)
            te_preds.append(pt_sum)
    elif mname == "catboost_cls":
        from catboost import CatBoostClassifier
        ev_w = np.arange(5, dtype=np.float64)[None, :]
        Xtr_c, Xva_c, Xte_c = Xtr.copy(), Xva.copy(), Xte.copy()
        cat_idx = []
        for i, c in enumerate(Xtr_c.columns):
            if str(Xtr_c[c].dtype) == "category":
                cat_idx.append(i)
                Xtr_c[c] = Xtr_c[c].astype(str)
                Xva_c[c] = Xva_c[c].astype(str)
                Xte_c[c] = Xte_c[c].astype(str)
        for s in seeds:
            m = CatBoostClassifier(
                loss_function="MultiClass", iterations=1100, learning_rate=0.03, depth=6, l2_leaf_reg=8.0,
                rsm=0.5, random_seed=s + 23, verbose=0, thread_count=-1
            )
            m.fit(Xtr_c, ytr, cat_features=cat_idx)
            va_preds.append((m.predict_proba(Xva_c) * ev_w).sum(axis=1))
            te_preds.append((m.predict_proba(Xte_c) * ev_w).sum(axis=1))
    elif mname == "xgb_cls":
        from xgboost import XGBClassifier
        ev_w = np.arange(5, dtype=np.float64)[None, :]
        Xtr_x, Xva_x, Xte_x = Xtr.copy(), Xva.copy(), Xte.copy()
        for c in Xtr_x.columns:
            if str(Xtr_x[c].dtype) == "category":
                cats = pd.Index(Xtr_x[c].astype(str).unique())
                Xtr_x[c] = pd.Categorical(Xtr_x[c].astype(str), categories=cats)
                Xva_x[c] = pd.Categorical(Xva_x[c].astype(str), categories=cats)
                Xte_x[c] = pd.Categorical(Xte_x[c].astype(str), categories=cats)
        for s in seeds:
            m = XGBClassifier(
                objective="multi:softprob", num_class=5,
                n_estimators=550, learning_rate=0.02, max_depth=5, min_child_weight=25,
                subsample=0.8, colsample_bytree=0.4, reg_alpha=1.5, reg_lambda=8.0,
                enable_categorical=True, max_cat_to_onehot=4, tree_method="hist",
                random_state=s + 29, n_jobs=-1, verbosity=0
            )
            m.fit(Xtr_x, ytr)
            va_preds.append((m.predict_proba(Xva_x) * ev_w).sum(axis=1))
            te_preds.append((m.predict_proba(Xte_x) * ev_w).sum(axis=1))
    elif mname == "xgb_ordinal":
        from xgboost import XGBClassifier
        Xtr_x, Xva_x, Xte_x = Xtr.copy(), Xva.copy(), Xte.copy()
        for c in Xtr_x.columns:
            if str(Xtr_x[c].dtype) == "category":
                cats = pd.Index(Xtr_x[c].astype(str).unique())
                Xtr_x[c] = pd.Categorical(Xtr_x[c].astype(str), categories=cats)
                Xva_x[c] = pd.Categorical(Xva_x[c].astype(str), categories=cats)
                Xte_x[c] = pd.Categorical(Xte_x[c].astype(str), categories=cats)
        for s in seeds:
            pv_sum = np.zeros(len(Xva_x), dtype=np.float64)
            pt_sum = np.zeros(len(Xte_x), dtype=np.float64)
            for thresh in (1, 2, 3, 4):
                yb = (ytr >= thresh).astype(int)
                m = XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=550, learning_rate=0.02, max_depth=5, min_child_weight=25,
                    subsample=0.8, colsample_bytree=0.4, reg_alpha=1.5, reg_lambda=8.0,
                    enable_categorical=True, max_cat_to_onehot=4, tree_method="hist",
                    random_state=s + 37 * thresh, n_jobs=-1, verbosity=0
                )
                m.fit(Xtr_x, yb)
                pv_sum += m.predict_proba(Xva_x)[:, 1]
                pt_sum += m.predict_proba(Xte_x)[:, 1]
            va_preds.append(pv_sum)
            te_preds.append(pt_sum)
    elif mname == "catboost_ordinal":
        from catboost import CatBoostClassifier
        Xtr_c, Xva_c, Xte_c = Xtr.copy(), Xva.copy(), Xte.copy()
        cat_idx = []
        for i, c in enumerate(Xtr_c.columns):
            if str(Xtr_c[c].dtype) == "category":
                cat_idx.append(i)
                Xtr_c[c] = Xtr_c[c].astype(str)
                Xva_c[c] = Xva_c[c].astype(str)
                Xte_c[c] = Xte_c[c].astype(str)
        for s in seeds:
            pv_sum = np.zeros(len(Xva_c), dtype=np.float64)
            pt_sum = np.zeros(len(Xte_c), dtype=np.float64)
            for thresh in (1, 2, 3, 4):
                yb = (ytr >= thresh).astype(int)
                m = CatBoostClassifier(
                    loss_function="Logloss", iterations=1100, learning_rate=0.025, depth=6, l2_leaf_reg=8.0,
                    rsm=0.5, subsample=0.8, random_seed=s + 41 * thresh, verbose=0, thread_count=-1
                )
                m.fit(Xtr_c, yb, cat_features=cat_idx)
                pv_sum += m.predict_proba(Xva_c)[:, 1]
                pt_sum += m.predict_proba(Xte_c)[:, 1]
            va_preds.append(pv_sum)
            te_preds.append(pt_sum)
    elif mname == "catboost_rmse":
        from catboost import CatBoostRegressor
        Xtr_c, Xva_c, Xte_c = Xtr.copy(), Xva.copy(), Xte.copy()
        cat_idx = []
        for i, c in enumerate(Xtr_c.columns):
            if str(Xtr_c[c].dtype) == "category":
                cat_idx.append(i)
                Xtr_c[c] = Xtr_c[c].astype(str)
                Xva_c[c] = Xva_c[c].astype(str)
                Xte_c[c] = Xte_c[c].astype(str)
        for s in seeds:
            m = CatBoostRegressor(
                iterations=1400, learning_rate=0.025, depth=6, l2_leaf_reg=8.0,
                rsm=0.5, subsample=0.8, random_seed=s, verbose=0, thread_count=-1
            )
            m.fit(Xtr_c, ytr, cat_features=cat_idx)
            va_preds.append(m.predict(Xva_c))
            te_preds.append(m.predict(Xte_c))
    elif mname == "xgb_rmse":
        from xgboost import XGBRegressor
        Xtr_x, Xva_x, Xte_x = Xtr.copy(), Xva.copy(), Xte.copy()
        for c in Xtr_x.columns:
            if str(Xtr_x[c].dtype) == "category":
                cats = pd.Index(Xtr_x[c].astype(str).unique())
                Xtr_x[c] = pd.Categorical(Xtr_x[c].astype(str), categories=cats)
                Xva_x[c] = pd.Categorical(Xva_x[c].astype(str), categories=cats)
                Xte_x[c] = pd.Categorical(Xte_x[c].astype(str), categories=cats)
        for s in seeds:
            m = XGBRegressor(
                n_estimators=650, learning_rate=0.02, max_depth=5, min_child_weight=25,
                subsample=0.8, colsample_bytree=0.4, reg_alpha=1.5, reg_lambda=8.0,
                enable_categorical=True, max_cat_to_onehot=4, tree_method="hist",
                random_state=s, n_jobs=-1, verbosity=0
            )
            m.fit(Xtr_x, ytr)
            va_preds.append(m.predict(Xva_x))
            te_preds.append(m.predict(Xte_x))
    elif mname == "mlp_ordinal":
        import torch
        import torch.nn as nn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # Convert to numeric matrix with QuantileTransformer (RankGauss) + StandardScaler
        def to_num(df_):
            out = df_.copy()
            for c in out.columns:
                if str(out[c].dtype) == "category":
                    out[c] = out[c].cat.codes.astype("float32")
            arr = out.values.astype("float32")
            return arr
        Atr, Ava, Ate = to_num(Xtr), to_num(Xva), to_num(Xte)
        med = np.nanmedian(Atr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        for A in (Atr, Ava, Ate):
            bad = ~np.isfinite(A)
            if bad.any():
                A[bad] = np.take(med, np.where(bad)[1])
        qt = QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=42)
        Atr = np.clip(qt.fit_transform(Atr), -5, 5).astype("float32")
        Ava = np.clip(qt.transform(Ava), -5, 5).astype("float32")
        Ate = np.clip(qt.transform(Ate), -5, 5).astype("float32")

        class ResMLP(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.in_proj = nn.Sequential(nn.Linear(d, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.45))
                self.block1 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.45), nn.Linear(256, 256), nn.LayerNorm(256))
                self.act = nn.SiLU()
                self.head_r = nn.Linear(256, 1)
                self.head_o = nn.Linear(256, 4)

            def forward(self, x):
                h = self.in_proj(x)
                h = self.act(h + self.block1(h))
                return self.head_r(h).squeeze(-1), self.head_o(h)

        X_tr_t = torch.tensor(Atr, device=dev)
        y_tr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
        y_ord_t = (y_tr_t.unsqueeze(1) >= torch.arange(1, 5, device=dev).unsqueeze(0)).float()
        X_va_t = torch.tensor(Ava, device=dev)
        X_te_t = torch.tensor(Ate, device=dev)
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            net = ResMLP(Atr.shape[1]).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=3e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=22)
            bs = 256
            net.train()
            for ep in range(22):
                perm = torch.randperm(len(X_tr_t), device=dev)
                for k in range(0, len(X_tr_t), bs):
                    bi = perm[k:k + bs]
                    xb = X_tr_t[bi]
                    # Gaussian feature noise + feature dropout for strong tabular regularization
                    xb = xb + 0.05 * torch.randn_like(xb)
                    r, o = net(xb)
                    loss = 0.6 * mse(r, y_tr_t[bi].float()) + 0.4 * bce(o, y_ord_t[bi])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
            net.eval()
            with torch.no_grad():
                rv, ov = net(X_va_t)
                rt, ot = net(X_te_t)
                pv = (0.5 * rv + 0.5 * torch.sigmoid(ov).sum(-1)).cpu().numpy()
                pt = (0.5 * rt + 0.5 * torch.sigmoid(ot).sum(-1)).cpu().numpy()
            va_preds.append(pv)
            te_preds.append(pt)
    else:
        raise ValueError(f"Unknown model {mname}")
    return np.mean(va_preds, axis=0), np.mean(te_preds, axis=0)


# ------------------------------------------------------------------ Thresholding & Metrics
def cortes_cuantiles(p, y):
    props = pd.Series(y).value_counts(normalize=True).sort_index().reindex(range(5), fill_value=0).values
    return np.quantile(p, np.cumsum(props)[:-1])


def opt_rounder_fit(p_tr, y_tr):
    init = cortes_cuantiles(p_tr, y_tr)
    def loss(th):
        th_s = np.sort(th)
        pred = np.digitize(p_tr, th_s)
        return -cohen_kappa_score(y_tr, pred, weights="quadratic")
    res = minimize(loss, init, method="Nelder-Mead", options={"maxiter": 400})
    return np.sort(res.x)


def eval_cv_preds(oof_dict, test_dict, splits, y_series):
    """Evaluate each model + weighted blend + rank average on 5-fold SGKF."""
    y_arr = y_series.values
    summary = {}
    for mname, oof in oof_dict.items():
        rmse = float(np.sqrt(np.mean((oof - y_arr) ** 2)))
        fq_q = []
        for tr_i, va_i in splits:
            p_va = oof[va_i]
            fq_q.append(cohen_kappa_score(y_arr[va_i], np.digitize(p_va, cortes_cuantiles(p_va, y_arr[tr_i])), weights="quadratic"))
        summary[mname] = {
            "rmse": rmse,
            "qwk_folds": [round(x, 5) for x in fq_q],
            "qwk_mean": float(np.mean(fq_q)),
            "qwk_std": float(np.std(fq_q, ddof=1)),
        }
        print(f"  [{mname:15s}] RMSE={rmse:.4f} | QWK={np.mean(fq_q):.4f} +/- {np.std(fq_q, ddof=1):.4f} | folds={[round(x, 4) for x in fq_q]}", flush=True)

    mnames = list(oof_dict.keys())
    if len(mnames) == 1:
        best_oof = oof_dict[mnames[0]]
        best_test = test_dict[mnames[0]]
        weights = {mnames[0]: 1.0}
        return summary, best_oof, best_test, weights, "single"

    # Out-of-fold Ridge/NNLS optimal weights (nested per fold so CV of blend is 100% honest!)
    M_oof = np.column_stack([oof_dict[m] for m in mnames])
    M_test = np.column_stack([test_dict[m] for m in mnames])

    # 1) Nested non-negative convex blend (raw scale)
    blend_oof = np.zeros(len(y_arr), dtype=np.float64)
    w_folds = []
    for tr_i, va_i in splits:
        def obj(w):
            w = np.maximum(w, 0)
            w = w / (w.sum() + 1e-12)
            return np.mean((M_oof[tr_i] @ w - y_arr[tr_i]) ** 2)
        res = minimize(obj, np.ones(len(mnames)) / len(mnames), method="Nelder-Mead")
        wf = np.maximum(res.x, 0)
        wf = wf / (wf.sum() + 1e-12)
        w_folds.append(wf)
        blend_oof[va_i] = M_oof[va_i] @ wf
    w_mean = np.mean(w_folds, axis=0)
    w_mean = w_mean / w_mean.sum()
    blend_test = M_test @ w_mean

    # 2) Rank-averaged blend using the same non-negative weights (1st place Wodori trick!)
    R_oof = np.column_stack([rankdata(oof_dict[m]) / len(y_arr) for m in mnames])
    R_test = np.column_stack([rankdata(test_dict[m]) / len(M_test) for m in mnames])
    rank_oof = R_oof @ w_mean
    rank_test = R_test @ w_mean

    # 3) Equal-weight blend
    eq_oof = M_oof.mean(axis=1)
    eq_test = M_test.mean(axis=1)

    cands = [("blend_opt", blend_oof, blend_test), ("blend_rank", rank_oof, rank_test), ("blend_eq", eq_oof, eq_test)]
    best_name, best_qwk, best_oof, best_test = None, -1.0, None, None
    for cname, coof, ctest in cands:
        fq = [cohen_kappa_score(y_arr[va_i], np.digitize(coof[va_i], cortes_cuantiles(coof[va_i], y_arr[tr_i])), weights="quadratic") for tr_i, va_i in splits]
        rmse = float(np.sqrt(np.mean((coof - y_arr) ** 2))) if cname != "blend_rank" else float("nan")
        summary[cname] = {"rmse": rmse, "qwk_folds": [round(x, 5) for x in fq], "qwk_mean": float(np.mean(fq)), "qwk_std": float(np.std(fq, ddof=1))}
        print(f"  [{cname:15s}] RMSE={rmse:.4f} | QWK={np.mean(fq):.4f} +/- {np.std(fq, ddof=1):.4f} | folds={[round(x, 4) for x in fq]}", flush=True)
        if np.mean(fq) > best_qwk:
            best_qwk, best_name, best_oof, best_test = float(np.mean(fq)), cname, coof, ctest
    weights = {m: round(float(w), 4) for m, w in zip(mnames, w_mean)}
    print("Optimal non-negative weights:", weights, "| Best ensemble:", best_name, flush=True)
    return summary, best_oof, best_test, weights, best_name


# ------------------------------------------------------------------ Run Exact 5-Fold StratifiedGroupKFold (seed=42)
MODELS = CONFIG.get("models", ["lgbm_rmse"])
SEEDS = CONFIG.get("seeds", [42, 100, 2024])

sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits_5 = list(sgkf5.split(train_raw, y_all, groups=groups_all))

CONFIG["_allow_deberta_in_fold"] = True
oof_5 = {m: np.zeros(len(train_raw), dtype=np.float64) for m in MODELS}
test_5 = {m: np.zeros(len(test_raw), dtype=np.float64) for m in MODELS}

print(f"\n=== Running 5-fold StratifiedGroupKFold (seed=42) | models={MODELS} | seeds={SEEDS} ===", flush=True)
for fi, (tr_i, va_i) in enumerate(splits_5):
    tf = time.time()
    Xtr, Xva, Xte = build_fold_features(tr_i, va_i)
    if fi == 0:
        print(f"Fold 0 feature count: {Xtr.shape[1]}", flush=True)
    for mname in MODELS:
        pv, pt = fit_predict_model(mname, Xtr, y_all.values[tr_i], Xva, Xte, SEEDS)
        oof_5[mname][va_i] = pv
        test_5[mname] += pt / len(splits_5)
    print(f"Fold {fi} done in {time.time()-tf:.1f}s", flush=True)
    del Xtr, Xva, Xte
    gc.collect()

print("\n=== 5-Fold StratifiedGroupKFold Summary ===", flush=True)
summary_5, best_oof_5, best_test_5, weights_5, best_ens_name = eval_cv_preds(oof_5, test_5, splits_5, y_all)

# Also check OOF Nelder-Mead OptimizedRounder vs Quantile Matching on the exact 5-fold splits!
qwk_nm_folds = []
for tr_i, va_i in splits_5:
    th_nm = opt_rounder_fit(best_oof_5[tr_i], y_all.values[tr_i])
    qwk_nm_folds.append(cohen_kappa_score(y_all.values[va_i], np.digitize(best_oof_5[va_i], th_nm), weights="quadratic"))
print(f"  [opt_rounder_nm ] QWK={np.mean(qwk_nm_folds):.4f} +/- {np.std(qwk_nm_folds, ddof=1):.4f} | folds={[round(x, 4) for x in qwk_nm_folds]}", flush=True)

# Optional: Train 10-fold StratifiedGroupKFold for stronger final submission (more training data per fold + 2x bagging!)
N_SUB = int(CONFIG.get("n_folds_sub", 5))
if N_SUB > 5 and not DEBERTA_TR_LIST:
    print(f"\n=== Training {N_SUB}-fold StratifiedGroupKFold for final submission ===", flush=True)
    CONFIG["_allow_deberta_in_fold"] = False
    sgkf_sub = StratifiedGroupKFold(n_splits=N_SUB, shuffle=True, random_state=2026)
    splits_sub = list(sgkf_sub.split(train_raw, y_all, groups=groups_all))
    oof_sub = {m: np.zeros(len(train_raw), dtype=np.float64) for m in MODELS}
    test_sub = {m: np.zeros(len(test_raw), dtype=np.float64) for m in MODELS}
    for fi, (tr_i, va_i) in enumerate(splits_sub):
        tf = time.time()
        Xtr, Xva, Xte = build_fold_features(tr_i, va_i)
        for mname in MODELS:
            pv, pt = fit_predict_model(mname, Xtr, y_all.values[tr_i], Xva, Xte, SEEDS)
            oof_sub[mname][va_i] = pv
            test_sub[mname] += pt / N_SUB
        print(f"Sub-Fold {fi}/{N_SUB} done in {time.time()-tf:.1f}s", flush=True)
        del Xtr, Xva, Xte
        gc.collect()
    print(f"\n=== {N_SUB}-Fold StratifiedGroupKFold Summary ===", flush=True)
    summary_sub, best_oof_sub, best_test_sub, weights_sub, best_ens_sub = eval_cv_preds(oof_sub, test_sub, splits_sub, y_all)
    # Blend 5-fold (seed 42) + 10-fold (seed 2026) test predictions in rank space for ultimate variance reduction!
    final_cont = 0.4 * (rankdata(best_test_5) / len(test_raw)) + 0.6 * (rankdata(best_test_sub) / len(test_raw))
else:
    summary_sub = None
    final_cont = best_test_5

# Final discrete submission via train-histogram quantile matching (proven most robust on Private LB by 5th & 13th place + EXP-017..035)
final_preds = pd.Series(
    np.digitize(final_cont, cortes_cuantiles(final_cont, y_all)),
    index=test_raw.index,
    name="AdoptionSpeed",
).astype(int)
final_preds.to_csv(f"{OUT}/submission.csv")

# Save OOF & test continuous predictions for downstream stacking!
oof_save = pd.DataFrame({m: oof_5[m] for m in MODELS}, index=train_raw.index, dtype="float32")
test_save = pd.DataFrame({m: test_5[m] for m in MODELS}, index=test_raw.index, dtype="float32")
oof_save["best_ens"] = best_oof_5.astype("float32")
test_save["best_ens"] = best_test_5.astype("float32")
oof_save.to_parquet(f"{OUT}/oof_preds.parquet")
test_save.to_parquet(f"{OUT}/test_preds.parquet")

res = {
    "exp": CONFIG["exp"],
    "config": {k: v for k, v in CONFIG.items() if not k.startswith("_")},
    "summary_5fold": summary_5,
    "summary_sub": summary_sub,
    "best_ens_name": best_ens_name,
    "weights_5fold": weights_5,
    "qwk_mean_exp": summary_5[best_ens_name]["qwk_mean"] if best_ens_name in summary_5 else summary_5[MODELS[0]]["qwk_mean"],
    "qwk_std_exp": summary_5[best_ens_name]["qwk_std"] if best_ens_name in summary_5 else summary_5[MODELS[0]]["qwk_std"],
    "qwk_folds_exp": summary_5[best_ens_name]["qwk_folds"] if best_ens_name in summary_5 else summary_5[MODELS[0]]["qwk_folds"],
    "qwk_nm_mean": float(np.mean(qwk_nm_folds)),
    "elapsed_sec": round(time.time() - t0, 1),
    "class_dist": final_preds.value_counts().sort_index().to_dict(),
}
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)

lines = [
    f"{CONFIG['exp']} | best={best_ens_name} | QWK 5-fold SGKF: {res['qwk_mean_exp']:.4f} +/- {res['qwk_std_exp']:.4f}",
    f"folds: {res['qwk_folds_exp']}",
    f"opt_rounder_nm: {res['qwk_nm_mean']:.4f}",
    f"weights: {weights_5}",
    f"class_dist: {res['class_dist']}",
    f"elapsed: {res['elapsed_sec']}s",
]
for m, st in summary_5.items():
    lines.append(f"  {m:15s}: RMSE={st['rmse']:.4f} QWK={st['qwk_mean']:.4f} +/- {st['qwk_std']:.4f}")
txt = "\n".join(lines) + "\n"
open(f"{OUT}/resultado.txt", "w").write(txt)
print("\n== resultado ==\n" + txt, flush=True)
