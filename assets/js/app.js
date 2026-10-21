/* 多元迴歸分析平台：讀取 reports/*.json（或 Pyodide 即時產生的報表）並繪製儀表板 */
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const METHOD_ORDER = ["Lasso", "Forward SFS", "RFE", "ElasticNet", "Exhaustive"];
  const POINT_STYLES = ["circle", "rect", "triangle", "rectRot", "circle"];
  const DASHES = [[], [6, 4], [2, 3], [8, 3, 2, 3], [1, 0]];
  const logTick = (v) => { // 對數軸只標 1、2、5 開頭的刻度，避免重疊
    const m = v / Math.pow(10, Math.floor(Math.log10(v)));
    return [1, 2, 5].some((d) => Math.abs(m - d) < 1e-6) ? val(v) : "";
  };
  const URL_EXAMPLES = [
    ["50 Startups", "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"],
    ["Boston（無標題）", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"],
    ["Insurance", "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"],
  ];

  const state = { index: [], report: null, reportId: null, charts: {}, worker: null, jobs: {}, jobId: 0,
                  custom: null, kMetric: "cv_rmse", histMode: "raw", method: "all" };

  /* ---------------- 格式化 ---------------- */
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function num(v, d = 4) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-4 || a >= 1e9)) return v.toExponential(2);
    return v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: 0 });
  }
  function val(v) { // 目標尺度的數值
    if (v === null || v === undefined) return "—";
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toLocaleString("en-US", { maximumFractionDigits: 2 }) + "M";
    if (a >= 1000) return Math.round(v).toLocaleString("en-US");
    if (a >= 10) return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
    return v.toLocaleString("en-US", { maximumSignificantDigits: 4 });
  }
  const pct = (v, d = 1) => (v === null || v === undefined ? "—" : (v * 100).toFixed(d) + "%");
  const pval = (p) => (p === null || p === undefined ? "—" : p < 0.001 ? "< 0.001" : p.toFixed(3));
  const mainRow = (r) => r.models.find((m) => m.model === r.main.name);
  const methodColor = (m) => css(`--series-${METHOD_ORDER.indexOf(m) + 1}`) || css("--series-1");

  /* ---------------- Chart.js 共用設定 ---------------- */
  const refLines = {
    id: "refLines",
    afterDatasetsDraw(chart, _a, opts) {
      const lines = (opts && opts.lines) || [];
      const { ctx, chartArea: a, scales } = chart;
      lines.forEach((l) => {
        ctx.save();
        ctx.strokeStyle = l.color || css("--text-muted");
        ctx.lineWidth = l.width || 1.5;
        ctx.setLineDash(l.dash || [5, 4]);
        ctx.beginPath();
        let tx, ty;
        if (l.x !== undefined) {
          const px = scales.x.getPixelForValue(l.x);
          if (px < a.left - 1 || px > a.right + 1) { ctx.restore(); return; }
          ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); tx = px + 4; ty = a.top + 12;
        } else if (l.y !== undefined) {
          const py = scales.y.getPixelForValue(l.y);
          if (py < a.top - 1 || py > a.bottom + 1) { ctx.restore(); return; }
          ctx.moveTo(a.left, py); ctx.lineTo(a.right, py); tx = a.left + 4; ty = py - 5;
        }
        ctx.stroke();
        if (l.label) {
          ctx.setLineDash([]);
          ctx.fillStyle = css("--text-secondary");
          ctx.font = "12px " + Chart.defaults.font.family;
          ctx.fillText(l.label, tx, ty);
        }
        ctx.restore();
      });
    },
  };
  const dots = { // 在橫向條狀圖上畫點（係數點估計）
    id: "dots",
    afterDatasetsDraw(chart, _a, opts) {
      if (!opts || !opts.values) return;
      const { ctx, scales } = chart;
      ctx.save();
      opts.values.forEach((v, i) => {
        const x = scales.x.getPixelForValue(v), y = scales.y.getPixelForValue(i);
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = opts.color; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = css("--surface-1"); ctx.stroke();
      });
      ctx.restore();
    },
  };

  function chartDefaults() {
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12;
    Chart.defaults.color = css("--text-secondary");
    Chart.defaults.borderColor = css("--grid");
    Chart.defaults.animation = false;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
    Chart.defaults.plugins.tooltip.backgroundColor = css("--surface-1");
    Chart.defaults.plugins.tooltip.titleColor = css("--text-primary");
    Chart.defaults.plugins.tooltip.bodyColor = css("--text-secondary");
    Chart.defaults.plugins.tooltip.borderColor = css("--border");
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.elements.point.hitRadius = 8;
    Chart.defaults.elements.line.borderWidth = 2;
  }

  function mk(id, config) {
    if (state.charts[id]) state.charts[id].destroy();
    const el = document.getElementById(id);
    if (!el) return null;
    config.plugins = [...(config.plugins || []), refLines, dots];
    state.charts[id] = new Chart(el, config);
    return state.charts[id];
  }

  const axisTitle = (text) => ({ display: true, text, color: css("--text-secondary") });

  /* ---------------- 資料集清單 ---------------- */
  async function loadIndex() {
    try {
      const res = await fetch("reports/index.json", { cache: "no-cache" });
      if (!res.ok) throw new Error(res.status);
      state.index = await res.json();
    } catch (e) {
      $("#loading").innerHTML = `<span class="error">無法讀取 reports/index.json。請用 <code>python -m http.server</code> 開啟本頁，不要直接雙擊 HTML 檔。</span>`;
      state.index = [];
    }
    renderCards();
    const want = new URLSearchParams(location.search).get("dataset") || (state.index[0] && state.index[0].id);
    if (want === "custom") showUpload();
    else if (want) selectDataset(want);
  }

  function renderCards() {
    const sel = $("#datasetSelect");
    const opt = (id, text) => `<option value="${esc(id)}">${esc(text)}</option>`;
    sel.innerHTML = `<optgroup label="內建資料集">${state.index.map((d) => opt(d.id, d.title)).join("")}</optgroup>` +
      `<optgroup label="自訂">${state.custom ? opt("custom-report", `自訂：${state.custom.meta.title}`) : ""}` +
      opt("custom", "＋ 其他：上傳自訂 CSV") + "</optgroup>";
    sel.onchange = () => {
      const id = sel.value;
      if (id === "custom") showUpload();
      else if (id === "custom-report") { $("#uploadPanel").hidden = false; showReport(state.custom, id); }
      else selectDataset(id);
    };
    markCard();
  }

  function markCard() {
    const active = !$("#uploadPanel").hidden && state.reportId !== "custom-report" ? "custom" : state.reportId;
    const sel = $("#datasetSelect");
    if (active && [...sel.options].some((o) => o.value === active)) sel.value = active;
    const meta = $("#datasetMeta");
    if (active === "custom") {
      meta.textContent = "在瀏覽器中用 Python 即時分析任意 CSV，資料不會上傳到伺服器。";
      return;
    }
    let d = state.index.find((x) => x.id === active);
    if (active === "custom-report" && state.custom) {
      const r = state.custom;
      d = { rows: r.overview.n_used, features: r.overview.n_features, target: r.overview.target, k: r.main.k, test_r2: mainRow(r).test_r2 };
    }
    meta.innerHTML = d ? `<span><b>${d.rows.toLocaleString()}</b> 筆</span><span><b>${d.features}</b> 個候選特徵</span>` +
      `<span>目標 y = <b>${esc(d.target)}</b></span><span>主模型 k = <b>${d.k}</b></span><span>Test R² <b>${num(d.test_r2, 3)}</b></span>` : "";
  }

  async function selectDataset(id) {
    $("#uploadPanel").hidden = true;
    $("#loading").hidden = false;
    $("#loading").textContent = "載入報表中…";
    try {
      const res = await fetch(`reports/${id}.json`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      showReport(await res.json(), id);
      history.replaceState(null, "", `?dataset=${id}`);
    } catch (e) {
      $("#loading").innerHTML = `<span class="error">讀取報表失敗：${esc(e.message)}</span>`;
    }
  }

  function showReport(r, id) {
    state.report = r;
    state.reportId = id;
    $("#loading").hidden = true;
    $("#report").hidden = false;
    markCard();
    render(r);
  }

  /* ---------------- 報表 ---------------- */
  function render(r) {
    chartDefaults();
    renderSummary(r);
    renderData(r);
    renderSelection(r);
    renderModels(r);
    renderCoef(r);
    renderEval(r);
    renderOutliers(r);
    renderPredictor(r);
  }

  function kpi(label, value, sub = "") {
    return `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  }

  function renderSummary(r) {
    const o = r.overview, m = mainRow(r);
    $("#rTitle").textContent = r.meta.title || o.target;
    $("#rDesc").innerHTML = esc(r.meta.description || "") +
      (r.meta.source ? ` <a href="${esc(r.meta.source)}" target="_blank" rel="noopener">資料來源</a>` : "");
    $("#kpis").innerHTML = [
      kpi("使用筆數", o.n_used.toLocaleString(), `訓練 ${o.n_train.toLocaleString()} / 測試 ${o.n_test.toLocaleString()}`),
      kpi("候選特徵", o.n_features, `目標：${esc(o.model_target)}`),
      kpi("主模型特徵數 k", r.main.k, esc(r.main.name)),
      kpi("Test R²", num(m.test_r2, 4), o.log_target ? "log 尺度" : ""),
      kpi("Test RMSE", val(m.test_rmse), `MAE ${val(m.test_mae)}${m.test_mape != null ? ` · MAPE ${num(m.test_mape, 1)}%` : ""}`),
      kpi("95% 預測區間涵蓋率", pct(m.pi_coverage), "理想值 ≈ 95%"),
    ].join("");
    const same = r.main.same_as.filter((x) => `OLS + ${x}` !== r.main.name);
    $("#mainSentence").innerHTML = `主模型 <b>${esc(r.main.name)}</b> 選入 ${r.main.k} 個特徵：` +
      r.main.features.map((f) => `<code>${esc(f)}</code>`).join(" ") +
      (same.length ? `（${same.map(esc).join("、")} 選出相同特徵）` : "") +
      (r.main.excluded.length ? `；排除 ${r.main.excluded.map((f) => `<code>${esc(f)}</code>`).join(" ")}` : "") + "。";
    $("#notes").innerHTML = o.notes.map((n) => `<li>${esc(n)}</li>`).join("") || "<li>無</li>";
  }

  function histChart(h, label) {
    const centers = h.edges.slice(0, -1).map((e, i) => (e + h.edges[i + 1]) / 2);
    mk("cHist", {
      type: "bar",
      data: { labels: centers.map((c) => val(c)), datasets: [{ label: "筆數", data: h.counts, backgroundColor: css("--series-1"),
        borderColor: css("--surface-1"), borderWidth: 1, barPercentage: 1, categoryPercentage: 1, borderRadius: 2 }] },
      options: { plugins: { legend: { display: false }, tooltip: { callbacks: {
        title: (it) => `${label} ≈ ${it[0].label}` } } },
        scales: { x: { title: axisTitle(label), ticks: { maxTicksLimit: 8 }, grid: { display: false } }, y: { title: axisTitle("筆數") } } },
    });
  }

  function renderData(r) {
    const o = r.overview;
    const seg = $("#histSeg");
    if (o.target_hist_model) {
      seg.innerHTML = `<button type="button" data-v="raw">${esc(o.target)}</button><button type="button" data-v="model">${esc(o.model_target)}</button>`;
      seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.histMode = b.dataset.v; drawHist(); }));
    } else { seg.innerHTML = ""; state.histMode = "raw"; }
    const drawHist = () => {
      seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === state.histMode)));
      const isModel = state.histMode === "model" && o.target_hist_model;
      histChart(isModel ? o.target_hist_model : o.target_hist, isModel ? o.model_target : o.target);
    };
    drawHist();
    $("#histCap").textContent = `目標 ${o.target} 的分布（偏態 ${num(o.target_stats.skew, 2)}${o.log_target ? "，已取 log" : ""}）`;

    const corr = o.corr_with_target;
    $("#corrWrap").style.height = Math.max(240, corr.length * 22 + 60) + "px";
    mk("cCorr", {
      type: "bar",
      data: { labels: corr.map((c) => c.feature), datasets: [{ label: "r", data: corr.map((c) => c.r),
        backgroundColor: corr.map((c) => (c.r >= 0 ? css("--pos") : css("--neg"))), borderRadius: 4, barPercentage: 0.75 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (it) => `r = ${num(it.raw, 3)}` } } },
        scales: { x: { min: -1, max: 1, title: axisTitle(`與 ${o.model_target} 的相關係數 r`) }, y: { grid: { display: false } } } },
    });

    const cm = o.corr_matrix;
    $("#heatFig").hidden = !cm;
    if (cm) {
      const pos = css("--pos"), neg = css("--neg"), mid = css("--neutral");
      const cell = (v) => {
        const a = Math.min(1, Math.abs(v));
        const bg = `color-mix(in srgb, ${v >= 0 ? pos : neg} ${Math.round(a * 100)}%, ${mid})`;
        const fg = a > 0.55 ? "#fff" : css("--text-primary");
        return `<td style="background:${bg};color:${fg}" title="r = ${num(v, 3)}">${v.toFixed(2)}</td>`;
      };
      $("#heat").innerHTML = `<thead><tr><th></th>${cm.features.map((f) => `<th>${esc(f)}</th>`).join("")}</tr></thead><tbody>` +
        cm.features.map((f, i) => `<tr><th>${esc(f)}</th>${cm.matrix[i].map(cell).join("")}</tr>`).join("") + "</tbody>";
    }

    $("#colTable").innerHTML = `<thead><tr><th>欄位</th><th class="l">判斷</th><th class="l">說明</th><th>不同值</th><th>缺失</th><th class="l">範例</th></tr></thead><tbody>` +
      o.columns.map((c) => `<tr><td>${esc(c.name)}</td><td class="l"><span class="chip ${c.kind}">${c.kind}</span></td><td class="l">${esc(c.reason)}</td>
        <td>${c.n_unique.toLocaleString()}</td><td>${c.missing}</td><td class="l">${c.sample.map(esc).join(", ")}</td></tr>`).join("") + "</tbody>";

    const pv = o.preview;
    $("#previewTable").innerHTML = `<thead><tr>${pv.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
      pv.rows.map((row) => `<tr>${row.map((v) => `<td>${esc(v ?? "")}</td>`).join("")}</tr>`).join("") + "</tbody>";
  }

  const K_METRICS = {
    cv_rmse: { label: (r) => `${r.settings.cv_folds}-fold CV RMSE（${r.overview.model_target}，訓練集）`, fmt: (v) => num(v, 4) },
    test_rmse: { label: (r) => `Test RMSE（${r.overview.target}）`, fmt: val },
    test_r2: { label: (r) => `Test R²${r.overview.log_target ? "（log 尺度）" : ""}`, fmt: (v) => num(v, 4) },
  };

  const METHOD_INFO = {
    all: { name: "全部比較", how: "五種方法都只用訓練集，各自產生「k = 1、2、3… 時要選哪 k 個特徵」的清單；再用同一個簡約規則替每個方法挑 k，特徵數最少者就是主模型。RMSE 曲線越低越好、R² 越高越好。" },
    "Lasso": { name: "Lasso (L1)", how: "在迴歸加上懲罰項 λ·Σ|β|，把 λ 由大到小慢慢放寬：一開始所有係數都被壓成 0，越早「掙脫 0」的特徵越重要，依這個先後順序排名。" },
    "Forward SFS": { name: "Forward SFS 前向選擇", how: "從 0 個特徵開始，每一輪把剩下的特徵逐一試加進模型，用交叉驗證看哪一個讓 R² 提升最多就正式加入，一直加到全部特徵為止。" },
    "RFE": { name: "RFE 遞迴特徵消除", how: "先用全部特徵配適（標準化後的）迴歸，每一輪刪掉係數絕對值最小、最不重要的特徵再重新配適；越晚被刪掉的特徵排名越前面。" },
    "ElasticNet": { name: "ElasticNet 彈性網路", how: "同時加上 L1 與 L2 懲罰（各占一半），和 Lasso 一樣看特徵進入模型的先後；遇到彼此高度相關的特徵時不會只留一個，結果比 Lasso 穩定。" },
    "Exhaustive": { name: "Exhaustive 窮舉搜尋", how: "對每個 k，把所有可能的 k 個特徵組合全部試過（共 2ᵖ − 1 種），挑訓練誤差最小的組合；保證是每個 k 的最佳解，但特徵一多就非常耗時。" },
  };

  function renderSelection(r) {
    const s = r.selection, mainM = r.main.name.replace("OLS + ", "");
    if (state.method !== "all" && !s.methods.includes(state.method)) state.method = "all";
    const btns = $("#methodBtns");
    btns.innerHTML = ["all", ...s.methods].map((m) => {
      const sw = m === "all" ? "" : ` style="--sw:${methodColor(m)}"`;
      const k = m === "all" ? "" : `<span class="k">k = ${s.k_by_method[m]}</span>`;
      return `<button type="button" class="mbtn" data-m="${esc(m)}"${sw}><span class="sw${m === "all" ? " all" : ""}"></span>${esc(m === "all" ? "全部比較" : m)}${k}</button>`;
    }).join("");
    btns.querySelectorAll(".mbtn").forEach((b) => b.addEventListener("click", () => { state.method = b.dataset.m; focusMethod(); }));
    const focusMethod = () => {
      const fm = state.method, info = METHOD_INFO[fm] || { name: fm, how: "" };
      btns.querySelectorAll(".mbtn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.m === fm)));
      const desc = $("#methodDesc");
      desc.style.setProperty("--sw", fm === "all" ? css("--series-1") : methodColor(fm));
      if (fm === "all") {
        desc.innerHTML = `<p class="how"><b>${info.name}</b>：${info.how}</p><p class="res">主模型：<b>${esc(r.main.name)}</b>（k = ${r.main.k}）。</p>`;
      } else {
        const mr = r.models.find((x) => x.model === `OLS + ${fm}`) || {};
        const tag = `OLS + ${fm}` === r.main.name ? "（← 就是主模型）" : r.main.same_as.includes(fm) ? "（和主模型選出相同特徵）" : "";
        desc.innerHTML = `<p class="how"><b>${info.name}</b>：${info.how}</p>` +
          `<p class="res">這次選出 <b>k = ${s.k_by_method[fm]}</b> 個特徵：${(mr.features || []).map((f) => `<code>${esc(f)}</code>`).join(" ")}；` +
          `Test R² = <b>${num(mr.test_r2, 4)}</b>，Test RMSE = <b>${val(mr.test_rmse)}</b>${tag}。</p>`;
      }
      document.querySelectorAll("#rankTable .focus").forEach((el) => el.classList.remove("focus"));
      if (fm !== "all") {
        const col = s.methods.indexOf(fm) + 2;
        document.querySelectorAll(`#rankTable tr > :nth-child(${col})`).forEach((el) => { el.classList.add("focus"); el.style.setProperty("--sw", methodColor(fm)); });
      }
      drawK();
    };
    const rule = r.settings.k_rule === "tolerance"
      ? `CV RMSE 落在最小值 + ${r.settings.k_tolerance} 以內的最小 k`
      : "1-SE 法則：CV RMSE ≤ 最小值 + 1 個標準誤的最小 k";
    $("#ruleText").innerHTML = `每個方法用同一個簡約規則選 k（${rule}），再取特徵數最少者為主模型。各方法選出的 k：` +
      s.methods.map((m) => `${esc(m)} <b>${s.k_by_method[m]}</b>`).join("、") + "。";

    const seg = $("#kSeg");
    seg.innerHTML = Object.keys(K_METRICS).map((k) => `<button type="button" data-v="${k}">${{ cv_rmse: "CV RMSE", test_rmse: "Test RMSE", test_r2: "Test R²" }[k]}</button>`).join("");
    seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.kMetric = b.dataset.v; drawK(); }));
    const drawK = () => {
      const metric = state.kMetric, M = K_METRICS[metric];
      seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === metric)));
      const fm = state.method;
      const datasets = s.methods.map((m) => {
        const idx = METHOD_ORDER.indexOf(m);
        const dim = fm !== "all" && m !== fm;
        const c = dim ? css("--grid") : methodColor(m);
        return { label: m, data: s.k_table.filter((row) => row.method === m).map((row) => ({ x: row.k, y: row[metric] })),
          borderColor: c, backgroundColor: m === "Exhaustive" ? css("--surface-1") : c, pointStyle: POINT_STYLES[idx] || "circle",
          pointRadius: dim ? 2 : 4, pointHoverRadius: 6, borderDash: DASHES[idx] || [], borderWidth: m === fm ? 3 : 2,
          pointBorderColor: m === "Exhaustive" ? c : css("--surface-1"), pointBorderWidth: m === "Exhaustive" ? 2 : 1, order: m === fm ? 0 : 1 };
      });
      const lines = fm === "all"
        ? [{ x: r.main.k, label: `主模型 k = ${r.main.k}` }]
        : [{ x: s.k_by_method[fm], label: `${fm} 選 k = ${s.k_by_method[fm]}`, color: methodColor(fm) }];
      if (metric === "cv_rmse") lines.push({ y: s.thresholds[fm === "all" ? mainM : fm], label: "簡約門檻", dash: [2, 3] });
      mk("cK", {
        type: "line", data: { datasets },
        options: { parsing: true, interaction: { mode: "nearest", axis: "x", intersect: false },
          plugins: { refLines: { lines }, legend: { labels: { sort: (a, b) => a.datasetIndex - b.datasetIndex } }, tooltip: { callbacks: {
            title: (it) => `k = ${it[0].parsed.x}`, label: (it) => `${it.dataset.label}：${M.fmt(it.parsed.y)}` } } },
          scales: { x: { type: "linear", min: 1, max: r.overview.n_features, ticks: { stepSize: 1 }, title: axisTitle("特徵數 k") },
            y: { title: axisTitle(M.label(r)) } } },
      });
      const ks = [...new Set(s.k_table.map((row) => row.k))].sort((a, b) => a - b);
      const higherBetter = metric === "test_r2";
      $("#kTable").innerHTML = `<thead><tr><th>k</th>${s.methods.map((m) => `<th>${esc(m)}</th>`).join("")}</tr></thead><tbody>` +
        ks.map((k) => {
          const rows = s.methods.map((m) => s.k_table.find((x) => x.method === m && x.k === k));
          const vals = rows.filter(Boolean).map((x) => x[metric]);
          const best = higherBetter ? Math.max(...vals) : Math.min(...vals);
          return `<tr${k === r.main.k ? ' class="main"' : ""}><td>${k}</td>${rows.map((row, i) => {
            if (!row) return "<td>—</td>";
            const txt = M.fmt(row[metric]);
            const chosen = s.k_by_method[s.methods[i]] === k ? " ◆" : "";
            return `<td>${row[metric] === best && vals.length > 1 ? `<b>${txt}</b>` : txt}${chosen}</td>`;
          }).join("")}</tr>`;
        }).join("") + "</tbody>";
      const o = r.overview;
      const what = {
        cv_rmse: `<b>CV RMSE</b>：只用訓練集做 ${r.settings.cv_folds}-fold 交叉驗證算出的平均預測誤差（${esc(o.model_target)} 尺度），<b>越小越好</b>。` +
          `選 k 就是看這個數字：找出落在「簡約門檻」以內的最小 k，代表再多加特徵也改善有限。`,
        test_rmse: `<b>Test RMSE</b>：在從未參與訓練的 ${Math.round(r.settings.test_size * 100)}% 測試集上的預測誤差，已換回 ${esc(o.target)} 的原始單位` +
          `（例如 ${val(mainRow(r).test_rmse)} 代表預測值平均約差這麼多），<b>越小越好</b>。它反映模型面對新資料的真實表現，但不拿來選 k，避免「偷看答案」。`,
        test_r2: `<b>Test R²</b>：測試集上模型能解釋 ${esc(o.model_target)} 變異的比例，介於 0～1，<b>越接近 1 越好</b>（例如 0.76 代表解釋了約 76% 的差異）。`,
      }[metric];
      $("#kExplain").innerHTML = `<p>每一列是特徵數 <b>k</b>、每一欄是一種方法；格子裡的數字是「用該方法排名前 k 的特徵配適 OLS」後的表現。` +
        `上方可切換指標。<b>粗體</b> = 該列表現最好的方法，◆ = 該方法最後選定的 k，底色列 = 主模型的 k。</p>` +
        `<p>${what}</p>` +
        `<p>可以用來：① 比較同一個 k 下哪種方法挑的特徵比較好；② 看 k 增加到哪裡之後數字幾乎不再改善（邊際效益遞減），決定要用幾個特徵；` +
        `③ 對照 CV（訓練集）與 Test（測試集）的趨勢是否一致，確認沒有過擬合。</p>`;
    };

    const feats = Object.keys(s.rank).sort((a, b) => {
      const ra = s.rank[a][s.methods[s.methods.length - 1]] ?? 99, rb = s.rank[b][s.methods[s.methods.length - 1]] ?? 99;
      return ra - rb;
    });
    $("#rankTable").innerHTML = `<thead><tr><th>特徵</th>${s.methods.map((m) => `<th>${esc(m)}<br><span class="muted small">k = ${s.k_by_method[m]}</span></th>`).join("")}</tr></thead><tbody>` +
      feats.map((f) => `<tr><td>${esc(f)}${r.main.features.includes(f) ? " ✓" : ""}</td>${s.methods.map((m) => {
        const k = s.rank[f][m];
        return `<td class="${k !== null && k <= s.k_by_method[m] ? "on" : ""}">${k ?? "—"}</td>`;
      }).join("")}</tr>`).join("") + "</tbody>";
    focusMethod();
  }

  function renderModels(r) {
    const cols = [
      ["模型", (m) => esc(m.model), "l"], ["k", (m) => m.k], ["Train R²", (m) => num(m.train_r2, 4)],
      ["Train Adj R²", (m) => num(m.train_adj_r2, 4)], ["CV RMSE", (m) => num(m.cv_rmse, 4)],
      ["Test R²", (m) => num(m.test_r2, 4)], ["Test RMSE", (m) => val(m.test_rmse)], ["Test MAE", (m) => val(m.test_mae)],
      ["MAPE", (m) => (m.test_mape != null ? num(m.test_mape, 2) + "%" : "—")], ["95% PI 涵蓋", (m) => pct(m.pi_coverage)],
      ["最大 VIF", (m) => num(m.max_vif, 2)],
    ];
    let html = `<thead><tr>${cols.map((c) => `<th class="${c[2] || ""}">${c[0]}</th>`).join("")}</tr></thead><tbody>` +
      r.models.map((m) => `<tr${m.model === r.main.name ? ' class="main"' : ""}>${cols.map((c) => `<td class="${c[2] || ""}">${c[1](m)}</td>`).join("")}</tr>`).join("");
    const ref = r.reference_model;
    if (ref && !ref.error) {
      html += `<tr><td>${esc(ref.model)}</td><td>${ref.k}</td><td>—</td><td>—</td><td>—</td><td>${num(ref.test_r2, 4)}</td>
        <td>${val(ref.test_rmse)}</td><td>${val(ref.test_mae)}</td><td>${ref.test_mape != null ? num(ref.test_mape, 2) + "%" : "—"}</td><td>—</td><td>—</td></tr>`;
    }
    $("#modelTable").innerHTML = html + "</tbody>";
    const b = r.baseline;
    $("#modelNote").innerHTML = `CV RMSE 為 ${esc(r.overview.model_target)} 尺度；Test RMSE / MAE 為 ${esc(r.overview.target)} 原始尺度` +
      (r.overview.log_target ? "（log 轉回時用 Duan smearing 修正）" : "") +
      `。Baseline OLS 整體 F = ${num(b.f_stat, 1)}（p ${pval(b.f_pvalue)}）。` +
      (ref && !ref.error ? "梯度提升樹為非線性參考上限，不是本分析的主模型。" : "");
  }

  function coefNote(r, row) {
    if (row.term === "const") return "截距";
    const tr = (r.predictor.features.find((f) => f.name === row.term) || {});
    const c = row.coef;
    if (r.overview.log_target) {
      if (tr.op === "log") return `${tr.source} +1% → ${r.overview.target} 約 ${c >= 0 ? "+" : ""}${num(c, 3)}%`;
      const e = (Math.exp(c) - 1) * 100;
      if (tr.op === "dummy") return `${tr.level} 相對 ${tr.baseline}：${e >= 0 ? "+" : ""}${num(e, 2)}%`;
      return `+1 單位 → ${r.overview.target} ${e >= 0 ? "+" : ""}${num(e, 2)}%`;
    }
    if (tr.op === "dummy") return `${tr.level} 相對 ${tr.baseline}：${c >= 0 ? "+" : ""}${val(c)}`;
    if (tr.op === "log") return `${tr.source} +1% → ${c >= 0 ? "+" : ""}${val(c / 100)}`;
    return `+1 單位 → ${c >= 0 ? "+" : ""}${val(c)}`;
  }

  function renderFormula(r) {
    const M = r.main, o = r.overview, P = r.predictor;
    const sig = (v) => Math.abs(v).toLocaleString("en-US", { maximumSignificantDigits: 4 });
    const yName = o.log_target ? `ln(${esc(o.target)})` : esc(o.target);
    const c0 = M.coef.find((c) => c.term === "const");
    const terms = M.coef.filter((c) => c.term !== "const").map((c) =>
      `<span class="t">${c.coef < 0 ? "−" : "+"} ${sig(c.coef)}·<span class="v">${esc(c.term)}</span></span>`);
    $("#formula").innerHTML = `預測 <span class="y">${yName}</span> =&nbsp;<span class="t">${c0.coef < 0 ? "−" : ""}${sig(c0.coef)}</span> ` + terms.join(" ");
    const notes = [];
    if (o.log_target) {
      notes.push(`預測 ${esc(o.target)} = e<sup>(上式)</sup>（中位數）；平均數再乘上 smearing 修正係數 ${num(P.smearing, 4)}。`);
    }
    const logs = P.features.filter((f) => f.op === "log" || f.op === "log1p").map((f) => `${esc(f.name)} = ${f.op === "log" ? "ln" : "ln(1+"}(${esc(f.source)})${f.op === "log1p" ? ")" : ""}`);
    if (logs.length) notes.push(`其中 ${logs.join("、")}。`);
    const dums = [...new Map(P.features.filter((f) => f.op === "dummy").map((f) => [f.source, f])).values()];
    if (dums.length) notes.push(`類別特徵為 0 / 1 虛擬變數，基準組：${dums.map((f) => `${esc(f.source)} = ${esc(f.baseline)}`).join("、")}。`);
    notes.push(`係數以訓練集 ${M.n.toLocaleString()} 筆資料用最小平方法估計，完整數值見下表。`);
    $("#formulaNote").innerHTML = notes.join(" ");
  }

  function renderCoef(r) {
    const M = r.main;
    renderFormula(r);
    $("#coefText").innerHTML = `n = ${M.n.toLocaleString()}，k = ${M.k}，R² = ${num(M.r2, 4)}，Adj R² = ${num(M.adj_r2, 4)}，` +
      `AIC = ${num(M.aic, 1)}，BIC = ${num(M.bic, 1)}；整體 F 檢定 F = ${num(M.f_stat, 1)}（df = ${M.df_model}, ${M.df_resid}），p ${pval(M.f_pvalue)}。`;
    $("#coefTable").innerHTML = `<thead><tr><th>項目</th><th>係數</th><th>標準誤</th><th>t</th><th>p-value</th><th>95% CI 下限</th><th>95% CI 上限</th><th class="l">解讀</th></tr></thead><tbody>` +
      M.coef.map((c) => `<tr><td>${esc(c.term)}</td><td>${num(c.coef, 5)}</td><td>${num(c.std_err, 5)}</td><td>${num(c.t, 2)}</td>
        <td>${pval(c.p_value)}</td><td>${num(c.ci_low, 5)}</td><td>${num(c.ci_high, 5)}</td><td class="l">${esc(coefNote(r, c))}</td></tr>`).join("") + "</tbody>";

    const cs = [...M.coef_std].sort((a, b) => b.coef - a.coef);
    $("#stdWrap").style.height = Math.max(220, cs.length * 30 + 70) + "px";
    mk("cStd", {
      type: "bar",
      data: { labels: cs.map((c) => c.term), datasets: [{ label: "95% CI", data: cs.map((c) => [c.ci_low, c.ci_high]),
        backgroundColor: css("--text-muted"), barThickness: 3, borderSkipped: false }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, dots: { values: cs.map((c) => c.coef), color: css("--series-1") },
        refLines: { lines: [{ x: 0, dash: [], color: css("--neg"), width: 1 }] },
        tooltip: { callbacks: { label: (it) => { const c = cs[it.dataIndex]; return `係數 ${num(c.coef, 4)}（CI ${num(c.ci_low, 4)} ~ ${num(c.ci_high, 4)}）`; } } } },
        scales: { x: { title: axisTitle(`每增加 1 個標準差，${r.overview.model_target} 的變化`) }, y: { grid: { display: false } } } },
    });

    const vif = M.vif;
    $("#vifWrap").style.height = Math.max(220, vif.length * 30 + 70) + "px";
    const vmax = Math.max(11, ...vif.map((v) => v.VIF || 0));
    mk("cVif", {
      type: "bar",
      data: { labels: vif.map((v) => v.feature), datasets: [{ label: "VIF", data: vif.map((v) => v.VIF),
        backgroundColor: vif.map((v) => (v.VIF > 10 ? css("--neg") : v.VIF > 5 ? css("--series-4") : css("--series-1"))), borderRadius: 4, barPercentage: 0.7 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false },
        refLines: { lines: [{ x: 5, label: "5" }, { x: 10, label: "10" }] },
        tooltip: { callbacks: { label: (it) => `VIF = ${num(it.raw, 2)}` } } },
        scales: { x: { min: 0, max: Math.ceil(vmax), title: axisTitle("VIF") }, y: { grid: { display: false } } } },
    });
  }

  function renderEval(r) {
    const M = r.main, P = M.predictions, D = M.diagnostics, logT = r.overview.log_target;
    $("#diagKpis").innerHTML = [
      kpi("95% PI 涵蓋率", pct(M.pi_coverage), "測試集"),
      kpi("Durbin-Watson", num(D.durbin_watson, 3), "≈ 2 表示誤差獨立"),
      kpi("Jarque-Bera", num(D.jarque_bera, 1), `p ${pval(D.jb_pvalue)}`),
      kpi("殘差偏態 / 峰度", `${num(D.skew, 2)} / ${num(D.kurtosis, 2)}`, "常態為 0 / 0"),
    ].join("");

    const pts = P.actual.map((a, i) => ({ x: a, y: P.predicted[i], inside: P.inside[i] }));
    const all = pts.flatMap((p) => [p.x, p.y]).filter((v) => !logT || v > 0);
    const lo = Math.min(...all), hi = Math.max(...all);
    const scaleType = logT ? "logarithmic" : "linear";
    mk("cAvp", {
      type: "scatter",
      data: { datasets: [
        { label: "落在 95% PI 內", data: pts.filter((p) => p.inside), backgroundColor: css("--series-1") + "88", pointRadius: 2.5, pointStyle: "circle" },
        { label: "落在 95% PI 外", data: pts.filter((p) => !p.inside), backgroundColor: css("--series-2"), pointRadius: 3.5, pointStyle: "triangle" },
        { label: "y = x", type: "line", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: css("--text-muted"), borderDash: [5, 4], pointRadius: 0, borderWidth: 1.5 },
      ] },
      options: { plugins: { tooltip: { callbacks: { label: (it) => `實際 ${val(it.parsed.x)}，預測 ${val(it.parsed.y)}` } } },
        scales: { x: { type: scaleType, title: axisTitle(`實際 ${r.overview.target}`), ticks: { callback: logT ? logTick : (v) => val(v), maxTicksLimit: logT ? undefined : 7, autoSkip: !logT } },
          y: { type: scaleType, title: axisTitle(`預測 ${r.overview.target}`), ticks: { callback: logT ? logTick : (v) => val(v), maxTicksLimit: logT ? undefined : 7, autoSkip: !logT } } } },
    });

    const n = Math.min(150, P.actual.length);
    const order = P.predicted.map((v, i) => i).slice(0, n).sort((a, b) => P.predicted[a] - P.predicted[b]);
    const xs = order.map((_, j) => j + 1);
    const band = css("--series-1");
    mk("cPi", {
      type: "line",
      data: { labels: xs, datasets: [
        { label: "95% 預測區間", data: order.map((i) => P.pi_high[i]), borderWidth: 0, pointRadius: 0, fill: "+1", backgroundColor: band + "26", stepped: "middle" },
        { label: "_pi_low", data: order.map((i) => P.pi_low[i]), borderWidth: 0, pointRadius: 0, fill: false, stepped: "middle" },
        { label: "95% 信賴區間", data: order.map((i) => P.ci_high[i]), borderWidth: 0, pointRadius: 0, fill: "+1", backgroundColor: band + "88", stepped: "middle" },
        { label: "_ci_low", data: order.map((i) => P.ci_low[i]), borderWidth: 0, pointRadius: 0, fill: false, stepped: "middle" },
        { label: "預測值", data: order.map((i) => P.predicted[i]), borderColor: band, pointRadius: 0 },
        { label: "實際值", data: order.map((i) => P.actual[i]), showLine: false, pointRadius: 3,
          pointBackgroundColor: order.map((i) => (P.inside[i] ? css("--text-primary") : css("--series-2"))),
          pointStyle: order.map((i) => (P.inside[i] ? "circle" : "triangle")), borderColor: "transparent", backgroundColor: css("--text-primary") },
      ] },
      options: { interaction: { mode: "index", intersect: false },
        plugins: { legend: { labels: { filter: (l) => !l.text.startsWith("_") } },
          tooltip: { filter: (it) => !it.dataset.label.startsWith("_"), callbacks: { label: (it) => `${it.dataset.label}：${val(it.parsed.y)}` } } },
        scales: { x: { title: axisTitle(`${n} 筆測試資料（依預測值排序）`), ticks: { maxTicksLimit: 8 }, grid: { display: false } },
          y: { type: scaleType, title: axisTitle(r.overview.target), ticks: { callback: logT ? logTick : (v) => val(v), maxTicksLimit: logT ? undefined : 7, autoSkip: !logT } } } },
    });
    $("#piCap").textContent = `測試集預測值與 95% 預測區間（隨機 ${n} 筆，依預測值排序；三角形 = 落在區間外）`;

    const R = M.residuals;
    mk("cRes", {
      type: "scatter",
      data: { datasets: [{ label: "殘差", data: R.fitted.map((f, i) => ({ x: f, y: R.resid[i] })), backgroundColor: css("--series-1") + "88", pointRadius: 2.5 }] },
      options: { plugins: { legend: { display: false }, refLines: { lines: [{ y: 0, dash: [], color: css("--neg"), width: 1 }] },
        tooltip: { callbacks: { label: (it) => `預測 ${num(it.parsed.x, 3)}，殘差 ${num(it.parsed.y, 3)}` } } },
        scales: { x: { title: axisTitle(`預測值（${r.overview.model_target}）`) }, y: { title: axisTitle("殘差") } } },
    });

    const Q = M.qq, qlo = Math.min(Q.theoretical[0], Q.sample[0]), qhi = Math.max(Q.theoretical.at(-1), Q.sample.at(-1));
    mk("cQq", {
      type: "scatter",
      data: { datasets: [
        { label: "標準化殘差分位數", data: Q.theoretical.map((t, i) => ({ x: t, y: Q.sample[i] })), backgroundColor: css("--series-1"), pointRadius: 2.5 },
        { label: "常態參考線", type: "line", data: [{ x: qlo, y: qlo }, { x: qhi, y: qhi }], borderColor: css("--neg"), pointRadius: 0, borderWidth: 1.5 },
      ] },
      options: { plugins: { tooltip: { callbacks: { label: (it) => `理論 ${num(it.parsed.x, 2)}，樣本 ${num(it.parsed.y, 2)}` } } },
        scales: { x: { title: axisTitle("常態理論分位數") }, y: { title: axisTitle("標準化殘差") } } },
    });
  }

  function renderOutliers(r) {
    const I = r.main.influence, pe = r.main.pct_error, rb = I.robust;
    $("#inflText").innerHTML = `高槓桿點（h > ${num(I.lev_threshold, 4)}）${I.n_high_leverage.toLocaleString()} 筆；` +
      `離群值（|學生化殘差| > 3）${I.n_outliers.toLocaleString()} 筆；高影響點（Cook's D > 4/n = ${num(I.cook_threshold, 5)}）${I.n_influential.toLocaleString()} 筆；` +
      `最大 Cook's D = ${num(I.max_cooks, 4)}${I.max_cooks < 1 ? "（< 1，沒有單一樣本主導模型）" : "（≥ 1，有樣本強烈影響係數）"}。` +
      (rb ? ` 移除 ${rb.n_removed.toLocaleString()} 筆高影響點後重新配適：Test R² ${num(rb.test_r2_before, 4)} → ${num(rb.test_r2_after, 4)}，` +
        `係數最大變動 ${num(rb.max_coef_shift_se, 1)} 個標準誤（${esc(rb.max_coef_shift_feature)}）。` : "");

    const p = I.points, big = p.cooks.map((c) => c > I.cook_threshold);
    const mkPts = (flag) => p.leverage.map((l, i) => ({ x: l, y: p.stud[i], c: p.cooks[i], row: p.row[i] })).filter((_, i) => big[i] === flag);
    mk("cInfl", {
      type: "scatter",
      data: { datasets: [
        { label: "一般樣本", data: mkPts(false), backgroundColor: css("--series-1") + "77", pointRadius: 2.5 },
        { label: "Cook's D > 4/n", data: mkPts(true), backgroundColor: css("--series-2"), pointRadius: 3.5, pointStyle: "triangle" },
      ] },
      options: { plugins: { refLines: { lines: [{ y: 3, label: "+3" }, { y: -3, label: "−3" }, { x: I.lev_threshold, label: "2p/n", dash: [2, 3] }] },
        tooltip: { callbacks: { label: (it) => `第 ${it.raw.row} 列：h = ${num(it.raw.x, 4)}，r = ${num(it.raw.y, 2)}，D = ${num(it.raw.c, 4)}` } } },
        scales: { x: { title: axisTitle("槓桿值 hᵢᵢ") }, y: { title: axisTitle("學生化殘差") } } },
    });

    $("#pctFig").hidden = !pe;
    if (pe) {
      $("#pctCap").textContent = `測試集誤差百分比：中位數 ${num(pe.median_abs, 1)}%，±10% 內 ${pct(pe.within10, 0)}，±20% 內 ${pct(pe.within20, 0)}`;
      const c = pe.hist.edges.slice(0, -1).map((e, i) => (e + pe.hist.edges[i + 1]) / 2);
      mk("cPct", {
        type: "bar",
        data: { labels: c.map((v) => v.toFixed(0)), datasets: [{ label: "筆數", data: pe.hist.counts,
          backgroundColor: c.map((v) => (Math.abs(v) <= 20 ? css("--series-1") : css("--text-muted"))),
          borderColor: css("--surface-1"), borderWidth: 1, barPercentage: 1, categoryPercentage: 1, borderRadius: 2 }] },
        options: { plugins: { legend: { display: false }, tooltip: { callbacks: { title: (it) => `誤差 ≈ ${it[0].label}%` } } },
          scales: { x: { title: axisTitle("(預測 − 實際) / 實際 (%)，藍色 = ±20% 內"), ticks: { maxTicksLimit: 10 }, grid: { display: false } },
            y: { title: axisTitle("筆數") } } },
      });
    }

    const top = I.top, vcols = top.length ? Object.keys(top[0].values) : [];
    $("#outTable").innerHTML = `<thead><tr><th>#</th><th>資料列</th><th>實際</th><th>預測</th><th>學生化殘差</th><th>槓桿值</th><th>Cook's D</th>${vcols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
      top.map((t) => `<tr><td>${t.rank}</td><td>${t.row}</td><td>${val(t.actual)}</td><td>${val(t.predicted)}</td><td>${num(t.stud_resid, 2)}</td>
        <td>${num(t.leverage, 4)}</td><td>${num(t.cooks_d, 4)}</td>${vcols.map((c) => { const v = t.values[c]; return `<td>${esc(typeof v === "number" ? (Number.isInteger(v) ? String(v) : num(v, 4)) : v ?? "")}</td>`; }).join("")}</tr>`).join("") + "</tbody>";
  }

  /* ---------------- 即時預測 ---------------- */
  function renderPredictor(r) {
    const P = r.predictor, form = $("#pform");
    const groups = [];
    P.features.forEach((f, i) => {
      if (f.op === "dummy") {
        let g = groups.find((x) => x.kind === "dummy" && x.source === f.source);
        if (!g) groups.push(g = { kind: "dummy", source: f.source, baseline: f.baseline, items: [] });
        g.items.push({ i, level: f.level, median: f.median });
      } else groups.push({ kind: f.op, i, f });
    });
    form.innerHTML = groups.map((g, gi) => {
      if (g.kind === "dummy") {
        const def = g.items.find((it) => it.median >= 0.5);
        return `<label>${esc(g.source)}<select data-g="${gi}"><option value="">${esc(g.baseline)}（基準組）</option>` +
          g.items.map((it) => `<option value="${it.i}"${def && def.i === it.i ? " selected" : ""}>${esc(it.level)}</option>`).join("") + "</select></label>";
      }
      const f = g.f, isLog = g.kind === "log" || g.kind === "log1p";
      if (f.is_binary && !isLog) {
        return `<label>${esc(f.name)}<select data-g="${gi}"><option value="0"${f.median < 0.5 ? " selected" : ""}>0</option><option value="1"${f.median >= 0.5 ? " selected" : ""}>1</option></select></label>`;
      }
      // log 特徵的滑桿在 log 尺度上移動，顯示原始值
      const step = f.is_int && !isLog ? 1 : (f.max - f.min) / 200;
      return `<label><span class="lbl"><span>${esc(isLog ? f.source : f.name)}</span><output data-o="${gi}"></output></span>
        <input type="range" data-g="${gi}" min="${f.min}" max="${f.max}" step="${step}" value="${f.median}"></label>`;
    }).join("");

    const update = () => {
      const x = new Array(P.features.length).fill(0);
      groups.forEach((g, gi) => {
        const el = form.querySelector(`[data-g="${gi}"]`);
        if (g.kind === "dummy") { if (el.value !== "") x[+el.value] = 1; return; }
        const v = parseFloat(el.value);
        const out = form.querySelector(`[data-o="${gi}"]`);
        const shown = g.kind === "log" ? Math.exp(v) : g.kind === "log1p" ? Math.expm1(v) : v;
        if (out) out.textContent = g.f.is_int && Math.abs(shown) < 1e4 ? String(Math.round(shown)) : val(shown);
        x[g.i] = v;
      });
      const x1 = [1, ...x];
      const yhat = x1.reduce((s, v, i) => s + v * P.beta[i], 0);
      let q = 0;
      for (let i = 0; i < x1.length; i++) for (let j = 0; j < x1.length; j++) q += x1[i] * P.XtX_inv[i][j] * x1[j];
      const se = Math.sqrt(P.sigma2 * (1 + q)), lo = yhat - P.t_crit * se, hi = yhat + P.t_crit * se;
      const back = P.log_target ? Math.exp : (v) => v;
      $("#pValue").textContent = val(back(yhat));
      $("#pRange").innerHTML = `${P.log_target ? "預測中位數" : "點估計"}；95% 預測區間 <b>${val(back(lo))} ~ ${val(back(hi))}</b>` +
        (P.log_target ? `；平均數（smearing 修正）${val(Math.exp(yhat) * P.smearing)}` : "");

      const contrib = P.features.map((f, i) => ({ name: f.op === "dummy" ? `${f.source}=${f.level}` : f.source || f.name,
        v: P.beta[i + 1] * (x[i] - f.median) }));
      contrib.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      const toShow = (v) => (P.log_target ? (Math.exp(v) - 1) * 100 : v);
      const minSpan = P.log_target ? 10 : Math.abs(r.overview.target_stats.std) * 0.25;
      $("#contribWrap").style.height = Math.max(200, contrib.length * 28 + 70) + "px";
      const ch = state.charts.cContrib;
      const data = contrib.map((c) => toShow(c.v));
      if (ch && ch.data.labels.length === contrib.length && ch.$reportId === state.reportId) {
        ch.data.labels = contrib.map((c) => c.name);
        ch.data.datasets[0].data = data;
        ch.data.datasets[0].backgroundColor = data.map((v) => (v >= 0 ? css("--pos") : css("--neg")));
        ch.update("none");
      } else {
        const c = mk("cContrib", {
          type: "bar",
          data: { labels: contrib.map((c) => c.name), datasets: [{ label: "貢獻", data, borderRadius: 4, barPercentage: 0.7,
            backgroundColor: data.map((v) => (v >= 0 ? css("--pos") : css("--neg"))) }] },
          options: { indexAxis: "y", plugins: { legend: { display: false },
            tooltip: { callbacks: { label: (it) => (P.log_target ? `${it.raw >= 0 ? "+" : ""}${num(it.raw, 2)}%` : `${it.raw >= 0 ? "+" : ""}${val(it.raw)}`) } } },
            scales: { x: { suggestedMin: -minSpan, suggestedMax: minSpan,
              title: axisTitle(P.log_target ? `對 ${r.overview.target} 的影響（%）` : `對 ${r.overview.target} 的影響`) }, y: { grid: { display: false } } } },
        });
        c.$reportId = state.reportId;
      }
    };
    form.oninput = update;
    form.onchange = update;
    update();
  }

  /* ---------------- 自訂 CSV（Pyodide） ---------------- */
  function showUpload() {
    $("#uploadPanel").hidden = false;
    markCard();
    $("#loading").hidden = true;
    $("#uploadPanel").scrollIntoView({ behavior: "smooth", block: "start" });
    history.replaceState(null, "", "?dataset=custom");
    ensureWorker();
  }

  function ensureWorker() {
    if (state.worker) return state.worker;
    const w = new Worker("assets/js/worker.js");
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "status") $("#engineStatus").textContent = m.text;
      else if (m.type === "progress") $("#runStatus").textContent = "分析中：" + m.text;
      else if (m.type === "result" || m.type === "error") {
        const job = state.jobs[m.id];
        delete state.jobs[m.id];
        if (job) (m.type === "result" ? job.resolve(m.data) : job.reject(new Error(m.message)));
      }
    };
    w.onerror = (e) => { $("#engineStatus").innerHTML = `<span class="error">Worker 錯誤：${esc(e.message)}</span>`; };
    state.worker = w;
    call("init").catch((e) => { $("#engineStatus").innerHTML = `<span class="error">${esc(e.message)}</span>`; });
    return w;
  }

  function call(type, payload = {}, transfer = []) {
    const id = ++state.jobId;
    return new Promise((resolve, reject) => {
      state.jobs[id] = { resolve, reject };
      ensureWorker().postMessage({ id, type, ...payload }, transfer);
    });
  }

  async function inspectBytes(buf, name) {
    $("#inspectBox").hidden = true;
    $("#runStatus").textContent = "";
    $("#engineStatus").textContent = `解析 ${name}…（首次使用需下載 Python 環境）`;
    try {
      const info = JSON.parse(await call("inspect", { bytes: buf, name }, [buf]));
      $("#engineStatus").textContent = "Python 引擎就緒";
      showInspect(info);
    } catch (e) {
      $("#engineStatus").innerHTML = `<span class="error">解析失敗：${esc(e.message)}</span>`;
    }
  }

  function showInspect(info) {
    $("#inspectBox").hidden = false;
    $("#inspectTitle").textContent = `${info.filename}：${info.rows.toLocaleString()} 列 × ${info.cols} 欄`;
    const usable = (k) => ["numeric", "categorical", "date"].includes(k);
    $("#inspectTable").innerHTML = `<thead><tr><th>使用</th><th class="l">欄位</th><th class="l">判斷</th><th class="l">說明</th><th>不同值</th><th>缺失</th><th class="l">範例</th></tr></thead><tbody>` +
      info.columns.map((c) => `<tr><td><input type="checkbox" data-col="${esc(c.name)}" ${usable(c.kind) ? "checked" : "disabled"}></td>
        <td class="l">${esc(c.name)}</td><td class="l"><span class="chip ${c.kind}">${c.kind}</span></td><td class="l">${esc(c.reason)}</td>
        <td>${c.n_unique.toLocaleString()}</td><td>${c.missing}</td><td class="l">${c.sample.map(esc).join(", ")}</td></tr>`).join("") + "</tbody>";
    $("#optTarget").innerHTML = info.numeric.map((c) => `<option${c === info.suggested_target ? " selected" : ""}>${esc(c)}</option>`).join("");
    if (!info.numeric.length) $("#runStatus").innerHTML = `<span class="error">找不到數值欄位，無法做迴歸</span>`;
  }

  async function runCustom() {
    const target = $("#optTarget").value;
    if (!target) return;
    const drop = [...document.querySelectorAll("#inspectTable input[data-col]")]
      .filter((c) => !c.checked && !c.disabled && c.dataset.col !== target).map((c) => c.dataset.col);
    const lt = $("#optLog").value;
    const options = { target, drop, log_target: lt === "auto" ? "auto" : lt === "true",
      k_rule: $("#optRule").value, k_tolerance: parseFloat($("#optTol").value) || 0.002,
      test_size: parseFloat($("#optTest").value), cv_folds: parseInt($("#optFolds").value, 10),
      max_exhaustive: parseInt($("#optExh").value, 10) || 15, reference_model: $("#optRef").checked };
    const btn = $("#runBtn");
    btn.disabled = true;
    $("#runStatus").textContent = "分析中…";
    const t0 = performance.now();
    try {
      const r = JSON.parse(await call("analyze", { options }));
      state.custom = r;
      $("#runStatus").textContent = `完成（${((performance.now() - t0) / 1000).toFixed(1)} 秒）`;
      renderCards();
      showReport(r, "custom-report");
      $("#s-summary").scrollIntoView({ behavior: "smooth" });
    } catch (e) {
      $("#runStatus").innerHTML = `<span class="error">分析失敗：${esc(e.message)}</span>`;
    } finally {
      btn.disabled = false;
    }
  }

  function bindUpload() {
    const dz = $("#dropzone"), fi = $("#fileInput");
    fi.addEventListener("change", async () => { const f = fi.files[0]; if (f) inspectBytes(await f.arrayBuffer(), f.name); fi.value = ""; });
    ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
    dz.addEventListener("drop", async (e) => { const f = e.dataTransfer.files[0]; if (f) inspectBytes(await f.arrayBuffer(), f.name); });
    const loadUrl = async () => {
      const url = $("#urlInput").value.trim();
      if (!url) return;
      $("#engineStatus").textContent = "下載中…";
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        inspectBytes(await res.arrayBuffer(), decodeURIComponent(url.split("/").pop() || "data.csv"));
      } catch (e) {
        $("#engineStatus").innerHTML = `<span class="error">下載失敗（${esc(e.message)}），網址需允許跨網域存取 (CORS)</span>`;
      }
    };
    $("#urlBtn").addEventListener("click", loadUrl);
    $("#urlInput").addEventListener("keydown", (e) => { if (e.key === "Enter") loadUrl(); });
    $("#urlExamples").innerHTML = URL_EXAMPLES.map(([n, u]) => `<button type="button" data-u="${esc(u)}">${esc(n)}</button>`).join("");
    $("#urlExamples").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { $("#urlInput").value = b.dataset.u; loadUrl(); }));
    $("#runBtn").addEventListener("click", runCustom);
  }

  /* ---------------- 其他 ---------------- */
  function bindTheme() {
    const btn = $("#themeBtn"), root = document.documentElement;
    let saved = null;
    try { saved = localStorage.getItem("mlr-theme"); } catch (e) { /* 無痕模式 */ }
    if (saved) root.dataset.theme = saved;
    const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    const label = () => { btn.textContent = isDark() ? "淺色模式" : "深色模式"; };
    label();
    btn.addEventListener("click", () => {
      root.dataset.theme = isDark() ? "light" : "dark";
      try { localStorage.setItem("mlr-theme", root.dataset.theme); } catch (e) { /* ignore */ }
      label();
      if (state.report) render(state.report);
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { label(); if (state.report) render(state.report); });
  }

  $("#downloadBtn").addEventListener("click", () => {
    if (!state.report) return;
    const blob = new Blob([JSON.stringify(state.report)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${state.report.meta.id || "report"}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  bindTheme();
  bindUpload();
  loadIndex();
})();
