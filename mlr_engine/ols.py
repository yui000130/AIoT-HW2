"""自製 OLS：係數、標準誤、t / p-value、信賴區間、預測區間、F 檢定、AIC / BIC。

- 係數：beta = (X'X)^-1 X'y，sigma^2 = SSE / (n - p)
- 平均反應的信賴區間 (CI)：yhat ± t * sigma * sqrt(x0' (X'X)^-1 x0)
- 個別觀測的預測區間 (PI)：yhat ± t * sigma * sqrt(1 + x0' (X'X)^-1 x0)
"""
import numpy as np
import pandas as pd
from scipy import stats


class OLS:
    def fit(self, X, y):
        self.cols = list(X.columns) if hasattr(X, "columns") else None
        X1 = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        y = np.asarray(y, float)
        self.n, self.p = X1.shape
        self.XtX_inv = np.linalg.pinv(X1.T @ X1)
        self.beta = self.XtX_inv @ X1.T @ y
        self.fitted = X1 @ self.beta
        self.resid = y - self.fitted
        self.df_resid = self.n - self.p
        sse = float(self.resid @ self.resid)
        self.sigma2 = sse / self.df_resid
        self.se = np.sqrt(np.diag(self.sigma2 * self.XtX_inv))
        self.t = self.beta / self.se
        self.pvalues = 2 * stats.t.sf(np.abs(self.t), self.df_resid)
        sst = float(((y - y.mean()) ** 2).sum())
        self.r2 = 1 - sse / sst
        self.adj_r2 = 1 - (1 - self.r2) * (self.n - 1) / self.df_resid
        k = self.p - 1                                    # 整體 F 檢定：H0 全部係數 = 0
        self.f_stat = (self.r2 / k) / ((1 - self.r2) / self.df_resid) if k > 0 else np.nan
        self.f_pvalue = stats.f.sf(self.f_stat, k, self.df_resid) if k > 0 else np.nan
        self.mae = float(np.mean(np.abs(self.resid)))
        ll = -self.n / 2 * (np.log(2 * np.pi * sse / self.n) + 1)
        self.aic = -2 * ll + 2 * self.p
        self.bic = -2 * ll + np.log(self.n) * self.p
        return self

    def summary(self, alpha=0.05):
        tc = stats.t.ppf(1 - alpha / 2, self.df_resid)
        names = ["const"] + (self.cols or [f"x{i}" for i in range(1, self.p)])
        return pd.DataFrame({"coef": self.beta, "std_err": self.se, "t": self.t,
                             "p_value": self.pvalues,
                             "ci_low": self.beta - tc * self.se,
                             "ci_high": self.beta + tc * self.se}, index=names)

    def predict(self, X, alpha=0.05, interval=None):
        X1 = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        yhat = X1 @ self.beta
        if interval is None:
            return yhat
        h = np.einsum("ij,jk,ik->i", X1, self.XtX_inv, X1)   # x0' (X'X)^-1 x0
        tc = stats.t.ppf(1 - alpha / 2, self.df_resid)
        se = np.sqrt(self.sigma2 * (h + (1 if interval == "prediction" else 0)))
        return yhat, yhat - tc * se, yhat + tc * se

    def influence(self, X):
        """訓練集的槓桿值 h_ii、學生化殘差 r_i 與 Cook's distance D_i"""
        X1 = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        lev = np.einsum("ij,jk,ik->i", X1, self.XtX_inv, X1)
        lev = np.clip(lev, 0, 1 - 1e-12)
        stud = self.resid / np.sqrt(self.sigma2 * (1 - lev))
        cooks = stud ** 2 / self.p * lev / (1 - lev) ** 2
        return lev, stud, cooks


def vif_table(Xdf):
    """VIF_j = 1 / (1 - R_j^2)；> 10 嚴重共線性，5–10 中度"""
    Xv = np.asarray(Xdf, float)
    out = []
    for i, col in enumerate(Xdf.columns):
        others = np.delete(Xv, i, axis=1)
        if others.shape[1] == 0:
            out.append((col, 1.0))
            continue
        o1 = np.column_stack([np.ones(len(others)), others])
        coef, *_ = np.linalg.lstsq(o1, Xv[:, i], rcond=None)
        res = Xv[:, i] - o1 @ coef
        sst = ((Xv[:, i] - Xv[:, i].mean()) ** 2).sum()
        r2 = 1 - (res @ res) / sst if sst > 0 else 1.0
        out.append((col, 1 / (1 - r2) if r2 < 1 - 1e-12 else np.inf))
    return pd.DataFrame(out, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)
