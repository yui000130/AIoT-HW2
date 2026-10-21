"""資料前處理：欄位型態判斷、日期展開、過濾、衍生欄位、log 轉換、虛擬變數、缺失值補值。

處理順序（設定檔 prep 區塊的每個鍵都對應一步，全部可省略）：
  1. 數字字串轉數值（去掉 $ , % 與空白）
  2. date_features   日期欄位 → 年 / 月
  3. filters         以 pandas eval 運算式移除異常列，例如 "bedrooms <= 11"
  4. derive          衍生欄位，例如 {"house_age": "sale_year - yr_built"}
  5. log_features    取 log，欄位改名為 log_<col>
  6. drop            刪除欄位
  7. features        指定特徵清單（省略則自動挑選：剔除 ID / 常數 / 長文字，類別欄位做虛擬變數）
  8. log_target      目標取 log（"auto"：全為正值且偏態 > 1 時才取）
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

ID_NAMES = {"id", "index", "idx", "no", "no.", "編號", "序號", "rownames", "row_id"}
TARGET_HINTS = ("price", "target", "label", "profit", "medv", "charges", "y", "salary", "sales", "revenue", "score")
_STRIP_RE = re.compile(r"[,$%\s]")
_COMPACT_DATE_RE = re.compile(r"^\d{8}(T\d{6})?$")

DEFAULT_PREP = {
    "log_target": "auto",
    "date_features": "auto",
    "filters": [],
    "derive": {},
    "log_features": [],
    "drop": [],
    "features": None,
    "categorical": [],
    "max_categories": 30,
    "max_features": 40,
}


@dataclass
class Prepared:
    X: pd.DataFrame
    y: np.ndarray
    y_raw: np.ndarray
    target: str
    model_target: str
    log_target: bool
    features: list
    transforms: dict
    raw: pd.DataFrame
    notes: list = field(default_factory=list)
    n_raw: int = 0


def _is_text(s):
    return not (ptypes.is_numeric_dtype(s) or ptypes.is_bool_dtype(s) or ptypes.is_datetime64_any_dtype(s))


def coerce_numeric(s: pd.Series):
    """若 95% 以上的非空值能轉成數字就回傳數值欄位，否則回傳 None"""
    if not _is_text(s):
        return None
    nn = s.dropna().astype(str).str.strip()
    nn = nn[nn != ""]
    if nn.empty:
        return None
    pct = nn.str.endswith("%")
    conv = pd.to_numeric(nn.str.replace(_STRIP_RE, "", regex=True), errors="coerce")
    if conv.notna().mean() < 0.95:
        return None
    if pct.mean() > 0.5:
        conv = conv / 100
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[conv.index] = conv.astype(float)
    return out


def parse_dates(s: pd.Series):
    """若 90% 以上能解析成日期就回傳 datetime 欄位，否則回傳 None"""
    if ptypes.is_datetime64_any_dtype(s):
        return s
    if not _is_text(s):
        return None
    nn = s.dropna().astype(str).str.strip()
    if nn.empty or nn.str.len().max() > 40:
        return None
    if nn.str.match(_COMPACT_DATE_RE).mean() > 0.9:
        out = pd.to_datetime(s.astype(str).str[:8], format="%Y%m%d", errors="coerce")
    else:
        if not nn.str.contains(r"\d{1,4}[-/.]\d{1,2}", regex=True).mean() > 0.9:
            return None
        try:
            out = pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            out = pd.to_datetime(s, errors="coerce")
    return out if out.notna().mean() >= 0.9 else None


def _is_id_like(name, s, n):
    low = str(name).strip().lower()
    if low in ID_NAMES or low.endswith("_id") or low.startswith("id_") or low.startswith("unnamed"):
        return True
    if ptypes.is_numeric_dtype(s) and s.nunique() == n and n > 20:
        v = s.dropna()
        if np.allclose(v, np.round(v)) and (v.is_monotonic_increasing or v.is_monotonic_decreasing):
            return True
    return False


def profile_columns(df: pd.DataFrame, max_categories=30):
    """回傳每個欄位的型態判斷：numeric / categorical / date / id / constant / text"""
    n = len(df)
    out = []
    for c in df.columns:
        s = df[c]
        kind, reason = None, ""
        num = s if ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s) else coerce_numeric(s)
        nunique = int(s.nunique(dropna=True))
        if nunique <= 1:
            kind, reason = "constant", "只有一種值"
        elif num is not None and _is_id_like(c, num, n):
            kind, reason = "id", "ID / 序號欄位"
        elif num is not None:
            kind = "numeric"
        elif parse_dates(s) is not None:
            kind, reason = "date", "日期 → 年、月"
        elif _is_id_like(c, s, n):
            kind, reason = "id", "ID / 序號欄位"
        elif nunique <= max_categories:
            kind, reason = "categorical", f"{nunique} 個類別 → 虛擬變數"
        else:
            kind, reason = "text", f"{nunique} 種不同文字，略過"
        sample = s.dropna().astype(str).head(3).tolist()
        out.append({"name": str(c), "kind": kind, "reason": reason, "n_unique": nunique,
                    "missing": int(s.isna().sum()), "sample": sample})
    return out


def suggest_target(profile):
    nums = [p["name"] for p in profile if p["kind"] == "numeric"]
    if not nums:
        return None
    for hint in TARGET_HINTS:
        for c in nums:
            if c.lower() == hint:
                return c
    for hint in TARGET_HINTS[:-6]:
        for c in nums:
            if hint in c.lower():
                return c
    return nums[-1]


def _safe_name(s):
    return re.sub(r"\s+", "_", str(s).strip())


def prepare(df: pd.DataFrame, target: str, prep: dict = None) -> Prepared:
    cfg = {**DEFAULT_PREP, **(prep or {})}
    notes = []
    n_raw = len(df)
    if target not in df.columns:
        raise ValueError(f"找不到目標欄位「{target}」")
    d = df.copy()

    # 1. 數字字串轉數值
    for c in d.columns:
        conv = coerce_numeric(d[c])
        if conv is not None:
            d[c] = conv
    if not ptypes.is_numeric_dtype(d[target]):
        raise ValueError(f"目標欄位「{target}」不是數值，無法做線性迴歸")
    d[target] = d[target].astype(float)
    miss_y = int(d[target].isna().sum())
    if miss_y:
        d = d[d[target].notna()]
        notes.append(f"移除目標欄位缺失的 {miss_y} 列")
    raw = df.loc[d.index]

    # 2. 日期欄位
    dcfg = cfg["date_features"]
    if dcfg == "auto":
        dcfg = {}
        for c in d.columns:
            if c != target and _is_text(d[c]) and parse_dates(d[c]) is not None:
                dcfg[c] = {"year": f"{_safe_name(c)}_year", "month": f"{_safe_name(c)}_month"}
    for c, parts in (dcfg or {}).items():
        dt = parse_dates(d[c]) if not ptypes.is_numeric_dtype(d[c]) else parse_dates(d[c].astype("Int64").astype(str))
        if dt is None:
            notes.append(f"日期欄位「{c}」無法解析，已略過")
            continue
        for part, name in parts.items():
            d[name] = getattr(dt.dt, part).astype(float)
        d = d.drop(columns=c)
        notes.append(f"日期「{c}」展開為 " + "、".join(parts.values()))

    # 3. 過濾異常列
    for expr in cfg["filters"]:
        n0 = len(d)
        d = d[d.eval(expr).astype(bool)]
        if n0 - len(d):
            notes.append(f"過濾條件 `{expr}` 移除 {n0 - len(d)} 列")
    raw = raw.loc[d.index]

    # 4. 衍生欄位
    for name, expr in (cfg["derive"] or {}).items():
        d[name] = d.eval(expr).astype(float)
    if cfg["derive"]:
        notes.append("衍生欄位：" + "、".join(f"{k} = {v}" for k, v in cfg["derive"].items()))

    # 5. log 轉換
    transforms = {}
    for c in cfg["log_features"]:
        v = d[c].astype(float)
        if (v <= 0).any():
            d["log_" + c] = np.log1p(v.clip(lower=0))
            transforms["log_" + c] = {"op": "log1p", "source": c}
        else:
            d["log_" + c] = np.log(v)
            transforms["log_" + c] = {"op": "log", "source": c}
        d = d.drop(columns=c)
    if cfg["log_features"]:
        notes.append("取 log 的特徵：" + "、".join(cfg["log_features"]))

    # 6. 刪除欄位
    drop = [c for c in cfg["drop"] if c in d.columns and c != target]
    d = d.drop(columns=drop)

    # 7. 特徵挑選與編碼
    y_raw = d[target].to_numpy(float)
    cand = d.drop(columns=target)
    if cfg["features"]:
        missing = [f for f in cfg["features"] if f not in cand.columns]
        if missing:
            raise ValueError(f"設定檔指定的特徵不存在：{missing}")
        cand = cand[cfg["features"]]
    forced_cat = set(cfg["categorical"])
    cols, skipped = [], []
    prof = {p["name"]: p for p in profile_columns(cand, cfg["max_categories"])}
    for c in cand.columns:
        kind = "categorical" if c in forced_cat else prof[str(c)]["kind"]
        if cfg["features"] and kind in ("id", "text", "date"):
            kind = "numeric" if ptypes.is_numeric_dtype(cand[c]) else "categorical"
        if kind == "numeric":
            v = cand[c].astype(float)
            if v.isna().any():
                notes.append(f"「{c}」缺失 {int(v.isna().sum())} 筆，補中位數")
                v = v.fillna(v.median())
            cols.append(v.rename(c))
            transforms.setdefault(c, {"op": "identity", "source": c})
        elif kind == "categorical":
            v = cand[c].astype(str).where(cand[c].notna(), None)
            if v.isna().any():
                v = v.fillna(v.mode().iloc[0])
            counts = v.value_counts()
            levels = sorted(counts.index, key=lambda L: (-counts[L], L))
            base = levels[0]
            for L in levels[1:]:
                name = f"{_safe_name(c)}_{_safe_name(L)}"
                cols.append((v == L).astype(float).rename(name))
                transforms[name] = {"op": "dummy", "source": c, "level": L, "baseline": base}
            notes.append(f"類別「{c}」以「{base}」為基準組，建立 {len(levels) - 1} 個虛擬變數")
        else:
            skipped.append(f"{c}（{prof[str(c)]['reason']}）")
    if skipped:
        notes.append("自動略過的欄位：" + "、".join(skipped))
    if not cols:
        raise ValueError("沒有可用的數值或類別特徵")
    X = pd.concat(cols, axis=1)
    X = X.loc[:, X.std() > 0]

    if X.shape[1] > cfg["max_features"]:
        corr = X.apply(lambda s: abs(np.corrcoef(s, y_raw)[0, 1])).fillna(0)
        keep = corr.sort_values(ascending=False).index[:cfg["max_features"]]
        notes.append(f"特徵數 {X.shape[1]} 超過上限 {cfg['max_features']}，保留與目標相關最高者")
        X = X[[c for c in X.columns if c in keep]]
    transforms = {k: v for k, v in transforms.items() if k in X.columns}

    # 8. 目標轉換
    lt = cfg["log_target"]
    if lt == "auto":
        lt = bool((y_raw > 0).all() and pd.Series(y_raw).skew() > 1)
        if lt:
            notes.append(f"目標「{target}」右偏（偏態 {pd.Series(y_raw).skew():.2f}），自動取 log")
    elif lt and (y_raw <= 0).any():
        lt = False
        notes.append("目標含 0 或負值，不取 log")
    y = np.log(y_raw) if lt else y_raw.copy()

    return Prepared(X=X, y=y, y_raw=y_raw, target=target,
                    model_target=f"log_{target}" if lt else target, log_target=bool(lt),
                    features=list(X.columns), transforms=transforms, raw=raw, notes=notes, n_raw=n_raw)
