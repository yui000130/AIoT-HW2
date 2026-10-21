/* 背景執行緒：載入 Pyodide 與 mlr_engine（與本機 CLI 同一份 Python 程式碼） */
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const MODULES = ["__init__.py", "io.py", "ols.py", "prep.py", "selection.py", "pipeline.py", "web_api.py"];
importScripts(PYODIDE_URL + "pyodide.js");

let ready = null;

function status(text) { self.postMessage({ type: "status", text }); }

async function init() {
  status("下載 Python 執行環境（Pyodide，約 10 MB）…");
  const py = await loadPyodide({ indexURL: PYODIDE_URL });
  status("載入 numpy / pandas / scipy / scikit-learn…");
  await py.loadPackage(["numpy", "pandas", "scipy", "scikit-learn"]);
  status("載入 mlr_engine…");
  const base = new URL("../../mlr_engine/", self.location.href);
  py.FS.mkdirTree("/home/pyodide/mlr_engine");
  for (const name of MODULES) {
    const res = await fetch(new URL(name, base));
    if (!res.ok) throw new Error(`無法載入 mlr_engine/${name}（HTTP ${res.status}）`);
    py.FS.writeFile(`/home/pyodide/mlr_engine/${name}`, await res.text());
  }
  py.runPython("import sys\nsys.path.insert(0, '/home/pyodide')\nimport mlr_engine.web_api as api");
  status("Python 引擎就緒");
  return py;
}

self.onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    ready = ready || init().catch((err) => { ready = null; throw err; });
    const py = await ready;
    const api = py.globals.get("api");
    let out;
    if (type === "init") {
      out = "ok";
    } else if (type === "inspect") {
      out = api.inspect(new Uint8Array(e.data.bytes), e.data.name);
    } else if (type === "analyze") {
      const progress = (m) => self.postMessage({ type: "progress", text: m });
      out = api.analyze(JSON.stringify(e.data.options), progress);
    }
    api.destroy();
    self.postMessage({ type: "result", id, data: out });
  } catch (err) {
    const msg = String(err && err.message || err);
    const last = msg.trim().split("\n").filter(Boolean).pop();
    self.postMessage({ type: "error", id, message: last || msg });
  }
};
