"""完整分析流程（CRISP-DM 的 Data Preparation → Modeling → Evaluation → Deployment），
輸出一份可直接 JSON 序列化的報表，給網頁或其他程式使用。

    from mlr_engine import run_analysis, read_csv_any
    report = run_analysis(read_csv_any("data/kc_house_data.csv"), target="price")
"""
import math
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from .ols import OLS, vif_table
from .prep import prepare, profile_columns
from .selection import METHODS, rank_table, run_selection

DEFAULT_MODEL = {
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42,
    "methods": list(METHODS),
    "k_rule": "one_se",          # "one_se"：CV RMSE ≤ 最小值 + 1 個標準誤；"tolerance"：≤ 最小值 + k_tolerance
    "k_tolerance": 0.002,
    "max_exhaustive": 18,
    "simple_feature": None,      # 單純線性回歸用的特徵；None = 與目標相關最高者
    "reference_model": True,     # 加入 HistGradientBoosting 作為非線性參考
    "alpha": 0.05,
    "max_points": 1200,          # 網頁散佈圖最多畫幾個點
}


def _rng_sample(n, size, seed):
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size, replace=False))


def _hist(v, bins=40):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    cnt, edges = np.histogram(v, bins=bins)
    return {"edges": edges.tolist(), "counts": cnt.tolist()}


def clean(obj, digits=7):
    """numpy → 原生型別，NaN / inf → None，浮點數保留有效位數"""
    if isinstance(obj, dict):
        return {str(k): clean(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist(), digits)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            return None
        return float(f"{x:.{digits}g}")
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def run_analysis(df: pd.DataFrame, target: str, prep: dict = None, model: dict = None,
                 meta: dict = None, progress=None):
    t0 = time.time()
    cfg = {**DEFAULT_MODEL, **(model or {})}
    say = progress or (lambda m: None)
    alpha = cfg["alpha"]

    # ---------- Data Preparation ----------
    say("資料前處理")
    P = prepare(df, target, prep)
    X, y, y_raw = P.X, P.y, P.y_raw
    features = P.features
    p_feat = len(features)
    positive = bool((y_raw > 0).all())

    X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
        X, y, y_raw, test_size=cfg["test_size"], random_state=cfg["random_state"])
    kf = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])

    def to_orig(pred, resid):
        """log → 原始尺度，使用 Duan smearing 修正 exp(E[log y]) 低估平均值的偏誤"""
        return np.exp(pred) * np.mean(np.exp(resid)) if P.log_target else pred

    cv_cache = {}

    def cv_stats(cols):
        key = tuple(sorted(cols))
        if key not in cv_cache:
            s = -cross_val_score(LinearRegression(), X_train[list(cols)], y_train, cv=kf,
                                 scoring="neg_root_mean_squared_error")
            cv_cache[key] = (float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s))))
        return cv_cache[key]

    def orig_metrics(pred_o, actual_o):
        out = {"RMSE": float(np.sqrt(mean_squared_error(actual_o, pred_o))),
               "MAE": float(mean_absolute_error(actual_o, pred_o)),
               "R2_orig": float(r2_score(actual_o, pred_o))}
        out["MAPE"] = float(np.mean(np.abs(actual_o - pred_o) / actual_o) * 100) if positive else None
        return out

    # ---------- Modeling：基底模型 ----------
    say("基底模型 (Baseline OLS)")
    vif_all = vif_table(X_train)
    ols_base = OLS().fit(X_train, y_train)

    # ---------- Modeling：特徵選擇 ----------
    subsets, sel_notes = run_selection(X_train, y_train, kf, cfg["methods"], cfg["max_exhaustive"], say)
    methods = list(subsets)

    say("比較每個方法、每個 k 的表現")
    krows = []
    n_te = len(y_test)
    for m in methods:
        for k, cols in subsets[m].items():
            mdl = OLS().fit(X_train[cols], y_train)
            pte = mdl.predict(X_test[cols])
            cv_m, cv_se = cv_stats(cols)
            r2t = r2_score(y_test, pte)
            om = orig_metrics(to_orig(pte, mdl.resid), r_test)
            krows.append({"method": m, "k": k, "features": cols, "cv_rmse": cv_m, "cv_se": cv_se,
                          "test_rmse_model": float(np.sqrt(mean_squared_error(y_test, pte))),
                          "test_rmse": om["RMSE"], "test_mae": om["MAE"], "test_r2": r2t,
                          "test_adj_r2": 1 - (1 - r2t) * (n_te - 1) / (n_te - k - 1) if n_te - k - 1 > 0 else None,
                          "bic": mdl.bic, "aic": mdl.aic})
    kdf = pd.DataFrame(krows)

    # 每個方法用同一個簡約規則選 k
    k_by_method, thresholds = {}, {}
    for m in methods:
        s = kdf[kdf.method == m].set_index("k")
        i_min = s["cv_rmse"].idxmin()
        if cfg["k_rule"] == "tolerance":
            th = s.loc[i_min, "cv_rmse"] + cfg["k_tolerance"]
        else:
            th = s.loc[i_min, "cv_rmse"] + s.loc[i_min, "cv_se"]
        thresholds[m] = float(th)
        k_by_method[m] = int(s[s["cv_rmse"] <= th].index.min())

    # ---------- Evaluation：模型比較 ----------
    say("模型比較")
    corr_y = X_train.apply(lambda c: np.corrcoef(c, y_train)[0, 1]).fillna(0)
    simple_f = cfg["simple_feature"] if cfg["simple_feature"] in features else corr_y.abs().idxmax()
    model_sets = {"Simple LR": [simple_f], "Baseline OLS": features}
    for m in methods:
        model_sets[f"OLS + {m}"] = subsets[m][k_by_method[m]]

    def evaluate(name, cols):
        mdl = OLS().fit(X_train[cols], y_train)
        pte, lo, hi = mdl.predict(X_test[cols], alpha=alpha, interval="prediction")
        r2 = r2_score(y_test, pte)
        k = len(cols)
        cv_m, cv_se = cv_stats(cols)
        om = orig_metrics(to_orig(pte, mdl.resid), r_test)
        return {"model": name, "k": k, "features": cols, "train_r2": mdl.r2, "train_adj_r2": mdl.adj_r2,
                "f_stat": mdl.f_stat, "f_pvalue": mdl.f_pvalue, "cv_rmse": cv_m, "cv_se": cv_se,
                "test_r2": r2, "test_adj_r2": 1 - (1 - r2) * (n_te - 1) / (n_te - k - 1) if n_te - k - 1 > 0 else None,
                "test_rmse": om["RMSE"], "test_mae": om["MAE"], "test_mape": om["MAPE"],
                "test_r2_orig": om["R2_orig"],
                "pi_coverage": float(np.mean((y_test >= lo) & (y_test <= hi))),
                "max_vif": float(vif_table(X_train[cols])["VIF"].max()) if k > 1 else None,
                "aic": mdl.aic, "bic": mdl.bic}

    results = [evaluate(n, c) for n, c in model_sets.items()]
    fs_rows = sorted([r for r in results if r["model"].startswith("OLS + ")], key=lambda r: (r["k"], r["cv_rmse"]))
    MAIN = fs_rows[0]["model"]
    SELECTED = model_sets[MAIN]
    same_as_main = [m for m in methods if set(model_sets[f"OLS + {m}"]) == set(SELECTED)]

    reference = None
    if cfg["reference_model"]:
        say("非線性參考模型 (HistGradientBoosting)")
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            hgb = HistGradientBoostingRegressor(random_state=cfg["random_state"]).fit(X_train, y_train)
            ptr, pte = hgb.predict(X_train), hgb.predict(X_test)
            om = orig_metrics(to_orig(pte, y_train - ptr), r_test)
            reference = {"model": "HistGradientBoosting (參考)", "k": p_feat, "test_r2": r2_score(y_test, pte),
                         "test_rmse": om["RMSE"], "test_mae": om["MAE"], "test_mape": om["MAPE"]}
        except Exception as e:  # pragma: no cover - 參考模型失敗不影響主流程
            reference = {"model": "HistGradientBoosting (參考)", "error": str(e)}

    # ---------- 主模型 ----------
    say("主模型統計摘要與診斷")
    ols = OLS().fit(X_train[SELECTED], y_train)
    coef = ols.summary(alpha)
    Xtr_sel = X_train[SELECTED]
    std_X = (Xtr_sel - Xtr_sel.mean()) / Xtr_sel.std(ddof=0)
    coef_std = OLS().fit(std_X, y_train).summary(alpha).drop("const")
    vif_sel = vif_table(Xtr_sel) if len(SELECTED) > 1 else pd.DataFrame({"feature": SELECTED, "VIF": [1.0]})

    yhat_te, pi_lo, pi_hi = ols.predict(X_test[SELECTED], alpha=alpha, interval="prediction")
    _, ci_lo, ci_hi = ols.predict(X_test[SELECTED], alpha=alpha, interval="confidence")
    inside = (y_test >= pi_lo) & (y_test <= pi_hi)
    smear = float(np.mean(np.exp(ols.resid))) if P.log_target else 1.0
    back = (lambda v: np.exp(v)) if P.log_target else (lambda v: v)
    pred_o = to_orig(yhat_te, ols.resid)

    idx = _rng_sample(n_te, cfg["max_points"], cfg["random_state"])
    pred_points = {"actual": r_test[idx], "predicted": back(yhat_te[idx]),
                   "pi_low": back(pi_lo[idx]), "pi_high": back(pi_hi[idx]),
                   "ci_low": back(ci_lo[idx]), "ci_high": back(ci_hi[idx]), "inside": inside[idx]}

    # 殘差診斷
    res_te = y_test - yhat_te
    std_res = res_te / np.sqrt(ols.sigma2)
    dw = float(np.sum(np.diff(ols.resid) ** 2) / np.sum(ols.resid ** 2))
    jb_stat, jb_p = stats.jarque_bera(ols.resid)
    qs = np.linspace(0.005, 0.995, 199)
    qq = {"theoretical": stats.norm.ppf(qs), "sample": np.quantile(std_res, qs)}
    resid_points = {"fitted": yhat_te[idx], "resid": res_te[idx]}

    # 離群值與影響點
    lev, stud, cooks = ols.influence(Xtr_sel)
    n_tr = ols.n
    lev_th, cook_th = 2 * ols.p / n_tr, 4 / n_tr
    keep = cooks <= cook_th
    robust = None
    if keep.sum() > ols.p + 1 and (~keep).sum() > 0:
        ols_rob = OLS().fit(Xtr_sel[keep], y_train[keep])
        pte_rob = ols_rob.predict(X_test[SELECTED])
        shift = (ols_rob.beta[1:] - ols.beta[1:]) / ols.se[1:]
        robust = {"n_removed": int((~keep).sum()),
                  "test_r2_before": float(r2_score(y_test, yhat_te)), "test_r2_after": float(r2_score(y_test, pte_rob)),
                  "test_rmse_before": orig_metrics(pred_o, r_test)["RMSE"],
                  "test_rmse_after": orig_metrics(to_orig(pte_rob, ols_rob.resid), r_test)["RMSE"],
                  "max_coef_shift_se": float(np.abs(shift).max()),
                  "max_coef_shift_feature": SELECTED[int(np.abs(shift).argmax())]}

    raw_tr = P.raw.loc[X_train.index]
    sources = [P.transforms.get(f, {}).get("source", f) for f in SELECTED]
    show_cols = [c for c in dict.fromkeys(sources) if c in P.raw.columns and c != target][:8]
    if not show_cols:
        show_cols = [c for c in P.raw.columns if c != target][:8]
    order = np.argsort(-cooks)[:10]
    top_out = []
    for rank, i in enumerate(order, 1):
        row = {"rank": rank, "row": int(X_train.index[i]), "actual": float(r_train[i]),
               "predicted": float(back(ols.fitted[i])),
               "stud_resid": float(stud[i]), "leverage": float(lev[i]), "cooks_d": float(cooks[i])}
        row["values"] = {c: (None if pd.isna(v) else (v if isinstance(v, (int, float, np.number)) else str(v)))
                         for c, v in raw_tr.iloc[i][show_cols].items()}
        top_out.append(row)
    ci = _rng_sample(n_tr, cfg["max_points"], cfg["random_state"] + 1)
    ci = np.union1d(ci, order)
    infl_points = {"leverage": lev[ci], "stud": stud[ci], "cooks": cooks[ci], "row": X_train.index.values[ci]}

    pct = None
    if positive:
        pe = (pred_o - r_test) / r_test * 100
        pct = {"median_abs": float(np.median(np.abs(pe))), "within10": float(np.mean(np.abs(pe) <= 10)),
               "within20": float(np.mean(np.abs(pe) <= 20)), "hist": _hist(np.clip(pe, -100, 150), 50)}

    # ---------- 資料概況 ----------
    corr_full = X.apply(lambda c: np.corrcoef(c, y)[0, 1]).fillna(0)
    overview = {
        "n_raw": P.n_raw, "n_used": len(X), "n_train": len(X_train), "n_test": n_te,
        "n_features": p_feat, "columns": profile_columns(df), "notes": P.notes + sel_notes,
        "target": target, "model_target": P.model_target, "log_target": P.log_target,
        "target_stats": {"mean": float(np.mean(y_raw)), "median": float(np.median(y_raw)),
                         "std": float(np.std(y_raw)), "min": float(np.min(y_raw)), "max": float(np.max(y_raw)),
                         "skew": float(pd.Series(y_raw).skew())},
        "target_hist": _hist(y_raw), "target_hist_model": _hist(y) if P.log_target else None,
        "corr_with_target": sorted(({"feature": f, "r": float(corr_full[f])} for f in features),
                                   key=lambda r: -abs(r["r"])),
        "corr_matrix": {"features": features, "matrix": np.corrcoef(X.values, rowvar=False)} if p_feat <= 30 else None,
        "preview": {"columns": [str(c) for c in df.columns],
                    "rows": df.head(50).astype(object).where(df.head(50).notna(), None).values.tolist()},
    }

    # ---------- Deployment：網頁即時預測用的參數 ----------
    tc = float(stats.t.ppf(1 - alpha / 2, ols.df_resid))
    feat_info = []
    for f in SELECTED:
        v = X_train[f]
        tr = P.transforms.get(f, {"op": "identity", "source": f})
        info = {"name": f, "min": v.min(), "max": v.max(), "median": v.median(), "mean": v.mean(),
                "is_int": bool(np.allclose(v, np.round(v))), "is_binary": bool(set(np.unique(v)) <= {0.0, 1.0}),
                **tr}
        feat_info.append(info)
    predictor = {"features": feat_info, "beta": ols.beta, "XtX_inv": ols.XtX_inv, "sigma2": ols.sigma2,
                 "df_resid": ols.df_resid, "t_crit": tc, "smearing": smear, "log_target": P.log_target,
                 "alpha": alpha}

    report = {
        "meta": {"title": target, **(meta or {}), "generated_in_sec": round(time.time() - t0, 2),
                 "engine": "mlr_engine 1.0"},
        "settings": {**cfg, "methods": methods},
        "overview": overview,
        "vif_all": vif_all.to_dict("records"),
        "baseline": {"train_r2": ols_base.r2, "train_adj_r2": ols_base.adj_r2, "f_stat": ols_base.f_stat,
                     "f_pvalue": ols_base.f_pvalue, "test_r2": r2_score(y_test, ols_base.predict(X_test))},
        "selection": {"methods": methods, "k_table": kdf.to_dict("records"), "k_by_method": k_by_method,
                      "thresholds": thresholds, "rank": rank_table(subsets, features)},
        "models": results, "reference_model": reference,
        "main": {"name": MAIN, "features": SELECTED, "k": len(SELECTED), "same_as": same_as_main,
                 "excluded": [f for f in features if f not in SELECTED],
                 "n": ols.n, "r2": ols.r2, "adj_r2": ols.adj_r2, "aic": ols.aic, "bic": ols.bic,
                 "f_stat": ols.f_stat, "f_pvalue": ols.f_pvalue, "df_model": ols.p - 1, "df_resid": ols.df_resid,
                 "coef": coef.reset_index(names="term").to_dict("records"),
                 "coef_std": coef_std.reset_index(names="term").to_dict("records"),
                 "vif": vif_sel.to_dict("records"),
                 "pi_coverage": float(inside.mean()),
                 "pi_width_model": float(np.mean(pi_hi - pi_lo)),
                 "predictions": pred_points, "residuals": resid_points,
                 "resid_hist": _hist(std_res, 50), "qq": qq,
                 "diagnostics": {"durbin_watson": dw, "jarque_bera": float(jb_stat), "jb_pvalue": float(jb_p),
                                 "skew": float(stats.skew(ols.resid)), "kurtosis": float(stats.kurtosis(ols.resid))},
                 "influence": {"lev_threshold": lev_th, "cook_threshold": cook_th,
                               "n_high_leverage": int((lev > lev_th).sum()),
                               "n_outliers": int((np.abs(stud) > 3).sum()),
                               "n_influential": int((cooks > cook_th).sum()), "max_cooks": float(cooks.max()),
                               "points": infl_points, "top": top_out, "robust": robust},
                 "pct_error": pct},
        "predictor": predictor,
    }
    say("完成")
    return clean(report)
