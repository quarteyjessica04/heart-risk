"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
let MODEL = null, METRICS = null;

// ---------- PWA ----------
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch(console.warn));
}
let deferredPrompt;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault(); deferredPrompt = e; $("#install").hidden = false;
});
$("#install").addEventListener("click", async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt(); await deferredPrompt.userChoice;
  deferredPrompt = null; $("#install").hidden = true;
});
window.addEventListener("appinstalled", () => ($("#install").hidden = true));
// iOS has no install prompt: show a hint instead.
if (/iphone|ipad|ipod/i.test(navigator.userAgent) && !navigator.standalone) {
  const b = $("#install"); b.hidden = false; b.textContent = "Add to Home Screen";
  b.onclick = () => alert("To install: tap the Share button in Safari, then “Add to Home Screen”.");
}
function net() {
  const el = $("#net"), on = navigator.onLine;
  el.textContent = on ? "Online" : "Offline"; el.className = "pill " + (on ? "ok" : "off");
}
addEventListener("online", net); addEventListener("offline", net); net();

// ---------- Theme ----------
const root = document.documentElement;
const saved = localStorage.getItem("theme");
root.dataset.theme = saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
$("#theme").onclick = () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("theme", root.dataset.theme);
};

// ---------- Tabs ----------
$$(".tab").forEach((t) => (t.onclick = () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === t.dataset.tab));
}));

// ---------- Form ----------
const LABELS = { age: "Age", sex: "Sex", blood_pressure: "Blood pressure", cholesterol: "Cholesterol", bmi: "BMI",
  smoker: "Smoker", diabetes: "Diabetes", physical_activity: "Activity", family_history: "Family history", stress_level: "Stress" };
const form = $("#form");
function syncOutputs() {
  $$("output", form).forEach((o) => (o.textContent = form.elements[o.dataset.for].value));
}
form.addEventListener("input", syncOutputs); syncOutputs();
$$(".seg").forEach((seg) => seg.addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$("button", seg).forEach((x) => x.classList.toggle("on", x === b));
}));
const segVal = (n) => Number($(`.seg[data-name="${n}"] button.on`).dataset.v);
$("#reset").onclick = () => {
  form.reset(); $$(".seg").forEach((s) => $$("button", s).forEach((b, i) => b.classList.toggle("on", i === 0)));
  syncOutputs(); render(null);
};

// ---------- Inference (logistic regression, in-browser) ----------
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
function predict(x) {
  const z = MODEL.features.map((f, i) => ((x[f] - MODEL.mean[i]) / MODEL.scale[i]) * MODEL.coef[i]);
  const logit = z.reduce((a, b) => a + b, MODEL.intercept);
  return { p: sigmoid(logit), contrib: MODEL.features.map((f, i) => ({ f, v: z[i] })) };
}
function values() {
  const v = {};
  ["age", "blood_pressure", "cholesterol", "bmi", "physical_activity", "stress_level"].forEach((n) => (v[n] = Number(form.elements[n].value)));
  ["sex", "smoker", "diabetes", "family_history"].forEach((n) => (v[n] = segVal(n)));
  return v;
}
form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (!MODEL) return;
  render(predict(values()));
});

function band(p) { return p < 0.4 ? ["Lower risk", "var(--ok)"] : p < 0.6 ? ["Moderate risk", "var(--warn)"] : ["Higher risk", "var(--bad)"]; }
function render(r) {
  const arc = $("#arc");
  if (!r) {
    arc.setAttribute("stroke-dasharray", "0 100"); $("#pct").textContent = "—"; $("#band").textContent = "Press “Calculate risk”";
    $("#summary").textContent = "Enter the patient’s details and calculate to see the estimated probability.";
    $("#factors").innerHTML = ""; return;
  }
  const [label, color] = band(r.p);
  arc.style.stroke = color; arc.setAttribute("stroke-dasharray", `${(r.p * 100).toFixed(1)} 100`);
  $("#pct").textContent = pct(r.p, 0); $("#band").textContent = label;
  $("#summary").textContent = `Estimated probability of heart disease: ${pct(r.p)}. This is a statistical estimate, not a diagnosis.`;
  const top = r.contrib.slice().sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 5);
  const max = Math.max(...top.map((t) => Math.abs(t.v)), 0.01);
  $("#factors").innerHTML = top.map((t) => {
    const w = (Math.abs(t.v) / max) * 50;
    return `<li><span>${LABELS[t.f]}</span><div class="bar"><i class="${t.v >= 0 ? "up" : "down"}" style="width:${w}%"></i></div>
      <span class="muted">${t.v >= 0 ? "+" : "−"}${Math.abs(t.v).toFixed(2)}</span></li>`;
  }).join("");
}

// ---------- Metrics ----------
function renderMetrics() {
  const m = METRICS, dep = m.models.find((x) => x.model === m.deployed_model);
  const v = $("#verdict");
  if (m.meets_80_percent) {
    v.className = "banner good";
    v.innerHTML = `<strong>Target met.</strong> Best cross-validated accuracy is ${pct(m.best_cv_accuracy)}.`;
  } else {
    v.className = "banner bad";
    v.innerHTML = `<strong>Low reliability.</strong> Best cross-validated accuracy is ${pct(m.best_cv_accuracy)}, versus a
      ${pct(m.majority_baseline_accuracy)} baseline from always guessing the most common class. On this dataset the models perform at chance level,
      so predictions should <u>not</u> be relied on. Retrain with real clinical data (<code>python ml/train.py data.csv</code>).`;
    const w = $("#reliability"); w.hidden = false;
    w.textContent = `⚠ Model accuracy is ${pct(m.best_cv_accuracy, 0)} (chance level). This estimate is not reliable.`;
  }
  const t = dep.test;
  $("#stats").innerHTML = [["Accuracy", t.accuracy], ["Precision", t.precision], ["Recall", t.recall], ["F1 score", t.f1], ["ROC-AUC", t.roc_auc]]
    .map(([k, x]) => `<div class="stat"><b>${pct(x)}</b><span>${k} (hold-out)</span></div>`).join("");
  $("#table").innerHTML = `<tr><th>Model</th><th>CV accuracy</th><th>CV AUC</th><th>Test acc.</th><th>Precision</th><th>Recall</th><th>F1</th><th>Train acc.</th></tr>` +
    m.models.map((r) => `<tr class="${r.model === m.deployed_model ? "deployed" : ""}"><td>${r.model}${r.model === m.deployed_model ? " ✓ deployed" : ""}</td>
      <td>${pct(r.cv_accuracy_mean)} ± ${pct(r.cv_accuracy_std)}</td><td>${r.cv_roc_auc_mean.toFixed(3)}</td><td>${pct(r.test.accuracy)}</td>
      <td>${pct(r.test.precision)}</td><td>${pct(r.test.recall)}</td><td>${pct(r.test.f1)}</td><td>${pct(r.train_accuracy)}</td></tr>`).join("");
  const c = t.confusion_matrix;
  $("#cm").innerHTML = `<div><b>${c.tn}</b>True neg.</div><div><b>${c.fp}</b>False pos.</div><div><b>${c.fn}</b>False neg.</div><div><b>${c.tp}</b>True pos.</div>`;
  $("#trainmeta").textContent = `${m.n_samples} patients (${m.n_train} train / ${m.n_test} test) · ${pct(m.positive_rate)} positive · trained ${m.trained_on}`;
}

Promise.all([fetch("/model.json").then((r) => r.json()), fetch("/metrics.json").then((r) => r.json())])
  .then(([a, b]) => { MODEL = a; METRICS = b; renderMetrics(); })
  .catch((e) => { $("#summary").textContent = "Could not load model files."; console.error(e); });
