"""讀取任意 CSV：自動判斷編碼、分隔符號與是否有標題列。"""
import csv
import io
import os
import re

import pandas as pd

ENCODINGS = ("utf-8-sig", "cp950", "big5", "latin-1")
_NUM_RE = re.compile(r"^\s*[-+]?[$]?\s*[\d,]*\.?\d+([eE][-+]?\d+)?\s*%?\s*$")


def decode_bytes(data: bytes) -> str:
    for enc in ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:30])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        counts = {d: first.count(d) for d in ",;\t|"}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else r"\s+"


def _looks_numeric(cell: str) -> bool:
    return bool(_NUM_RE.match(cell.strip().strip('"').strip("'")))


def read_csv_any(source, max_rows=None) -> pd.DataFrame:
    """source 可以是檔案路徑、bytes 或 str（CSV 內容）。"""
    if isinstance(source, bytes):
        text = decode_bytes(source)
    elif isinstance(source, str) and "\n" not in source and os.path.exists(source):
        with open(source, "rb") as f:
            text = decode_bytes(f.read())
    else:
        text = str(source)
    text = text.lstrip("﻿")
    sep = sniff_delimiter(text)
    reader_kw = {"sep": sep, "engine": "python"} if sep == r"\s+" else {"sep": sep, "low_memory": False}

    first_line = text.splitlines()[0] if text else ""
    first_cells = re.split(sep, first_line.strip()) if sep == r"\s+" else next(csv.reader([first_line], delimiter=sep))
    headerless = len(first_cells) > 1 and all(_looks_numeric(c) for c in first_cells if c.strip())

    if headerless:
        df = pd.read_csv(io.StringIO(text), header=None, nrows=max_rows, **reader_kw)
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    else:
        df = pd.read_csv(io.StringIO(text), nrows=max_rows, **reader_kw)
        df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed:") or df[c].notna().any()]]
    return df
