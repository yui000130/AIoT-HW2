"""命令列介面

  # 用設定檔分析（內建資料集）
  python -m mlr_engine --config configs/kc_house.json --out reports/kc_house.json

  # 直接分析任意 CSV
  python -m mlr_engine data/my.csv --target price --out reports/my.json

  # 重新產生 configs/ 下所有資料集的報表與 reports/index.json
  python -m mlr_engine --build-all
"""
import argparse
import glob
import json
import os
import sys
import time

from . import read_csv_any, run_analysis, run_config, suggest_target, profile_columns


def _write(report, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  → {out}（{os.path.getsize(out) / 1024:.0f} KB）")


def _summary(r):
    m = next(x for x in r["models"] if x["model"] == r["main"]["name"])
    print(f"  主模型 {m['model']}（k = {m['k']}）：Test R² = {m['test_r2']:.4f}，Test RMSE = {m['test_rmse']:,.2f}，"
          f"95% PI 涵蓋率 = {m['pi_coverage']:.1%}")
    print(f"  選入：{', '.join(r['main']['features'])}")


def build_all(config_dir="configs", report_dir="reports"):
    index = []
    for path in sorted(glob.glob(os.path.join(config_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[{cfg['id']}] {cfg.get('title', '')}")
        t0 = time.time()
        r = run_config(cfg, progress=lambda m: None)
        _write(r, os.path.join(report_dir, f"{cfg['id']}.json"))
        _summary(r)
        print(f"  耗時 {time.time() - t0:.1f} 秒")
        main = next(x for x in r["models"] if x["model"] == r["main"]["name"])
        index.append({"id": cfg["id"], "title": cfg.get("title", cfg["id"]), "description": cfg.get("description", ""),
                      "data": cfg["data"], "target": cfg["target"], "rows": r["overview"]["n_used"],
                      "features": r["overview"]["n_features"], "k": r["main"]["k"], "test_r2": main["test_r2"],
                      "order": cfg.get("order", 99)})
    index.sort(key=lambda d: (d["order"], d["id"]))
    with open(os.path.join(report_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"index.json：{len(index)} 個資料集")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mlr_engine", description="多元線性迴歸 + 特徵選擇分析")
    ap.add_argument("csv", nargs="?", help="CSV 檔案路徑")
    ap.add_argument("--target", help="目標欄位（省略則自動猜測）")
    ap.add_argument("--config", help="資料集設定檔 JSON")
    ap.add_argument("--out", help="輸出報表 JSON 路徑")
    ap.add_argument("--k-rule", choices=["one_se", "tolerance"])
    ap.add_argument("--build-all", action="store_true", help="重新產生 configs/ 下所有報表")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if a.build_all:
        return build_all()
    say = lambda m: print("  ·", m)
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f)
        if a.k_rule:
            cfg.setdefault("model", {})["k_rule"] = a.k_rule
        r = run_config(cfg, progress=say)
        out = a.out or os.path.join("reports", f"{cfg['id']}.json")
    elif a.csv:
        df = read_csv_any(a.csv)
        target = a.target or suggest_target(profile_columns(df))
        print(f"目標欄位：{target}")
        model = {"k_rule": a.k_rule} if a.k_rule else None
        r = run_analysis(df, target, model=model, meta={"title": os.path.basename(a.csv)}, progress=say)
        out = a.out or os.path.join("reports", os.path.splitext(os.path.basename(a.csv))[0] + ".json")
    else:
        ap.print_help()
        return 1
    _write(r, out)
    _summary(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
