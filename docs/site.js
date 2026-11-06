(() => {
"use strict";
const data = window.SEOULMMOD_RESULTS;
let scale = "district";
const horizon = document.querySelector("#horizon");
const displayName = name => ({HA:"Historical Average",LV:"Latest Value",LatestValue:"Latest Value",GWNet:"Graph WaveNet"}[name] || name);
const cell = (tag, text, className) => {
  const el = document.createElement(tag); el.textContent = text;
  if (className) el.className = className;
  if (tag === "th") el.scope = "row";
  return el;
};
function renderBenchmark() {
  const h = Number(horizon.value);
  const rows = [...data.benchmark[scale]].sort((a,b) => (a.scores[h].mae ?? Infinity) - (b.scores[h].mae ?? Infinity));
  const best = Math.min(...rows.filter(r=>r.scores[h].mae !== null).map(r=>r.scores[h].mae));
  const tbody = document.querySelector("#benchmark-table tbody"); tbody.replaceChildren();
  rows.forEach(row => {
    const tr = document.createElement("tr"), score = row.scores[h];
    tr.append(cell("th",displayName(row.model)));
    tr.append(cell("td",score.mae === null ? "—" : score.mae.toFixed(scale === "district" ? 1 : 3),score.mae === best ? "best" : ""));
    tr.append(cell("td",score.rmse === null ? "—" : score.rmse.toString()));
    const status = cell("td","");
    status.append(cell("span",({OOM:"Out of memory",TE:"Time limit"}[row.status] || "Completed"),"status"+(score.mae === null ? " failed" : "")));
    tr.append(status); tbody.append(tr);
  });
  const learned = rows.filter(r=>!["HA","LV"].includes(r.model) && r.scores[h].mae !== null).length;
  document.querySelector("#benchmark-summary").textContent = (scale === "district" ? "625" : "181,476") + " OD pairs · " + [6,12,24][h] + "-hour forecast · " + learned + " of 12 learned baselines completed";
}
document.querySelectorAll("[data-scale]").forEach(button => button.addEventListener("click",() => {
  scale = button.dataset.scale;
  document.querySelectorAll("[data-scale]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));
  renderBenchmark();
}));
horizon.addEventListener("change",renderBenchmark);
function renderSparsity() {
  const metric = document.querySelector("#sparsity-metric").value;
  const tbody = document.querySelector("#sparsity-table tbody"); tbody.replaceChildren();
  const winners = data.sparsity.buckets.map(b=>Math.min(...data.sparsity.models.map(m=>data.sparsity.scores[m][b][metric])));
  data.sparsity.models.forEach(model=>{
    const tr = document.createElement("tr"); tr.append(cell("th",displayName(model)));
    data.sparsity.buckets.forEach((bucket,index)=>{
      const value = data.sparsity.scores[model][bucket][metric];
      tr.append(cell("td",value.toFixed(metric==="positive"?2:3),value===winners[index]?"best":""));
    });
    tbody.append(tr);
  });
}
document.querySelector("#sparsity-metric").addEventListener("change",renderSparsity);
const yearBody = document.querySelector("#year-table tbody");
data.crossyear.forEach(row=>{
  const tr=document.createElement("tr");
  tr.append(cell("th",displayName(row.model)),cell("td",row.baseline.toFixed(2)),cell("td",row.test.toFixed(2)),cell("td","+"+row.delta.toFixed(2)+"%"));
  yearBody.append(tr);
});
document.querySelector("#copy-citation").addEventListener("click",async()=>{
  const text = document.querySelector("#bibtex").textContent;
  const status = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(text); status.textContent = "Copied.";
  } catch (_) {
    const selection = window.getSelection(), range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex")); selection.removeAllRanges(); selection.addRange(range);
    status.textContent = "Citation selected. Press Ctrl+C or ⌘C to copy.";
  }
});
renderBenchmark(); renderSparsity();
})();

