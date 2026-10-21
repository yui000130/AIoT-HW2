"""給瀏覽器 Web Worker（Pyodide）呼叫的介面：輸入輸出都是 JSON 字串。"""
import json

from .io import read_csv_any
from .pipeline import clean, run_analysis
from .prep import profile_columns, suggest_target

_STATE = {}


def inspect(data, filename="uploaded.csv"):
    """解析 CSV，回傳欄位型態與建議目標欄位；DataFrame 暫存起來給 analyze 用"""
    if hasattr(data, "to_py"):
        data = data.to_py()
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    df = read_csv_any(data)
    _STATE["df"], _STATE["name"] = df, filename
    prof = profile_columns(df)
    numeric = [p["name"] for p in prof if p["kind"] == "numeric"]
    return json.dumps(clean({"filename": filename, "rows": len(df), "cols": df.shape[1], "columns": prof,
                             "numeric": numeric, "suggested_target": suggest_target(prof),
                             "preview": {"columns": [str(c) for c in df.columns],
                                         "rows": df.head(8).astype(object).where(df.head(8).notna(), None)
                                         .values.tolist()}}), ensure_ascii=False)


def analyze(options_json, progress=None):
    opts = json.loads(options_json)
    df = _STATE.get("df")
    if df is None:
        raise RuntimeError("請先上傳 CSV")
    prep = {"log_target": opts.get("log_target", "auto"), "drop": opts.get("drop", [])}
    model = {k: opts[k] for k in ("test_size", "cv_folds", "k_rule", "k_tolerance", "max_exhaustive",
                                  "reference_model", "methods") if k in opts}
    meta = {"id": "custom", "title": _STATE.get("name", "自訂資料集"),
            "description": "使用者上傳的資料集（瀏覽器端 Pyodide 即時分析）"}
    report = run_analysis(df, opts["target"], prep, model, meta, progress)
    return json.dumps(report, ensure_ascii=False, separators=(",", ":"))
