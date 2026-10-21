"""mlr_engine：可重複使用的多元線性迴歸 + 特徵選擇分析引擎。

同一份程式碼可以在本機 CLI 執行，也可以在瀏覽器中由 Pyodide 執行（見 assets/js/worker.js）。
"""
from .io import read_csv_any
from .ols import OLS, vif_table
from .pipeline import DEFAULT_MODEL, run_analysis
from .prep import DEFAULT_PREP, prepare, profile_columns, suggest_target
from .selection import METHODS, run_selection

__all__ = ["read_csv_any", "OLS", "vif_table", "run_analysis", "prepare", "profile_columns",
           "suggest_target", "run_selection", "run_config", "METHODS", "DEFAULT_MODEL", "DEFAULT_PREP"]
__version__ = "1.3.0"


def run_config(config: dict, base_dir=".", progress=None):
    """依資料集設定檔（configs/*.json）讀資料並跑完整分析"""
    import os
    path = config["data"] if os.path.isabs(config["data"]) else os.path.join(base_dir, config["data"])
    df = read_csv_any(path)
    meta = {k: config[k] for k in ("id", "title", "description", "source", "data") if k in config}
    return run_analysis(df, config["target"], config.get("prep"), config.get("model"), meta, progress)
