"""五種特徵選擇演算法，每一種都產生「k = 1…p 時要用哪 k 個特徵」，全部只用訓練集。

1. Lasso (L1)          沿 Lasso 正則化路徑，依係數「離開 0」的先後排序
2. Forward SFS         從 0 個特徵開始，每次加入讓 k-fold CV R² 提升最多的特徵
3. RFE                 從全部特徵開始，反覆刪除標準化係數絕對值最小的特徵
4. ElasticNet          同 Lasso，沿彈性網路路徑排序（l1_ratio = 0.5），共線性高時較穩定
5. Exhaustive Search   窮舉全部 2^p − 1 種子集，找出每個 k 下訓練 SSE 最小的組合
"""
from itertools import combinations

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, enet_path, lasso_path
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

METHODS = ("Lasso", "Forward SFS", "RFE", "ElasticNet", "Exhaustive")


def _path_order(Xs, yc, features, path_fn, l1_ratio=1.0):
    """依係數第一次變成非 0 的先後排序特徵（正則化路徑上的進入順序）"""
    a_max = np.max(np.abs(Xs.T @ yc)) / len(yc) / l1_ratio
    alphas = a_max * np.logspace(0, -5, 600)
    kw = {} if path_fn is lasso_path else {"l1_ratio": l1_ratio}
    _, coefs, _ = path_fn(Xs, yc, alphas=alphas, **kw)
    nz = np.abs(coefs) > 1e-10
    first = np.where(nz.any(axis=1), nz.argmax(axis=1), len(alphas))
    return [features[i] for i in np.argsort(first, kind="stable")]


def lasso_order(Xs, yc, features):
    return _path_order(Xs, yc, features, lasso_path)


def enet_order(Xs, yc, features, l1_ratio=0.5):
    return _path_order(Xs, yc, features, enet_path, l1_ratio)


def forward_order(X_train, y_train, kf):
    features = list(X_train.columns)

    def cv_r2(cols):
        return cross_val_score(LinearRegression(), X_train[cols], y_train, cv=kf, scoring="r2").mean()

    order, remaining = [], features.copy()
    while remaining:
        scores = {f: cv_r2(order + [f]) for f in remaining}
        best = max(scores, key=scores.get)
        order.append(best)
        remaining.remove(best)
    return order


def rfe_order(Xs, y_train, features):
    """n_features_to_select=1 可得到完整的刪除順序（ranking 1 = 最後留下）"""
    rfe = RFE(LinearRegression(), n_features_to_select=1).fit(Xs, y_train)
    return [features[i] for i in np.argsort(rfe.ranking_, kind="stable")]


def exhaustive_best(X_train, y_train, progress=None):
    """用中心化後的 Gram 矩陣快速計算每個子集的 SSE"""
    features = list(X_train.columns)
    p = len(features)
    Xc = X_train.values - X_train.values.mean(axis=0)
    yc = y_train - y_train.mean()
    G, b, yy = Xc.T @ Xc, Xc.T @ yc, float(yc @ yc)
    best = {}
    for k in range(1, p + 1):
        if progress:
            progress(f"Exhaustive：k = {k} / {p}")
        best_sse, best_S = np.inf, None
        for S in combinations(range(p), k):
            S = list(S)
            bs = b[S]
            try:
                sse = yy - bs @ np.linalg.solve(G[np.ix_(S, S)], bs)
            except np.linalg.LinAlgError:
                continue
            if sse < best_sse:
                best_sse, best_S = sse, S
        if best_S is not None:
            best[k] = [features[i] for i in best_S]
    return best


def run_selection(X_train, y_train, kf, methods=METHODS, max_exhaustive=18, progress=None):
    """回傳 (subsets, notes)；subsets[method][k] = 特徵清單"""
    say = progress or (lambda m: None)
    features = list(X_train.columns)
    p = len(features)
    Xs = StandardScaler().fit_transform(X_train)
    yc = y_train - y_train.mean()
    subsets, notes = {}, []
    for m in methods:
        if m == "Lasso":
            say("Lasso 正則化路徑")
            order = lasso_order(Xs, yc, features)
        elif m == "ElasticNet":
            say("ElasticNet 正則化路徑")
            order = enet_order(Xs, yc, features)
        elif m == "Forward SFS":
            say("Forward SFS（交叉驗證）")
            order = forward_order(X_train, y_train, kf)
        elif m == "RFE":
            say("RFE 遞迴特徵消除")
            order = rfe_order(Xs, y_train, features)
        elif m == "Exhaustive":
            if p > max_exhaustive:
                notes.append(f"特徵數 {p} > {max_exhaustive}，略過 Exhaustive（需評估 2^{p} 種組合）")
                continue
            say(f"Exhaustive：評估 {2 ** p - 1:,} 種子集")
            subsets[m] = exhaustive_best(X_train, y_train, say)
            continue
        else:
            raise ValueError(f"未知的特徵選擇方法：{m}")
        subsets[m] = {k: order[:k] for k in range(1, p + 1)}
    return subsets, notes


def rank_table(subsets, features):
    """每個特徵在各方法中第幾個被選入（Exhaustive 為該特徵最早出現在最佳子集的 k）"""
    out = {}
    for f in features:
        out[f] = {m: min((k for k, s in sub.items() if f in s), default=None) for m, sub in subsets.items()}
    return out
