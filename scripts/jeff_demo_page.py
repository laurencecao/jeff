"""Interactive demo page for Jeff: type a claim, get probabilities.

Serves a single page where you enter a claim and its evidence, and see the
model's actual probability distribution over supported / refuted /
not_enough_info — plus Noul and Score questions on the same state.

    uv run python -m scripts.jev_clf_server     # then open http://127.0.0.1:8079
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[1]

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Jeff — a local Jev replacement</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font: 16px/1.55 -apple-system, system-ui, sans-serif; max-width: 900px;
         margin: 0 auto; padding: 2.2rem 1.4rem; }
  h1 { font-size: 2rem; margin: 0; }
  .sub { color: #888; margin: .2rem 0 1.6rem; }
  label { display:block; font-weight:600; margin:.9rem 0 .25rem; font-size:.92rem; }
  textarea { width:100%; min-height:64px; padding:.6rem .7rem; font:inherit;
             border-radius:8px; border:1px solid #8886; background:transparent; }
  button { margin-top:1rem; padding:.6rem 1.4rem; font:inherit; font-weight:600;
           border-radius:8px; border:0; cursor:pointer; background:#2a8; color:#fff; }
  button:disabled { opacity:.5; cursor:default; }
  .row { display:flex; gap:1rem; flex-wrap:wrap; }
  .row > div { flex:1 1 260px; }
  .out { margin-top:1.6rem; }
  .card { border:1px solid #8884; border-radius:10px; padding:1rem 1.2rem; margin-bottom:.9rem; }
  .verdict { font-size:1.5rem; font-weight:700; }
  .bar { height:20px; border-radius:5px; background:#8882; overflow:hidden; margin:.2rem 0 .6rem; }
  .bar > div { height:100%; }
  .lab { display:flex; justify-content:space-between; font-size:.9rem; }
  .lvl { margin-bottom:.15rem; }
  .muted { color:#888; font-size:.88rem; }
  .err { color:#c33; font-weight:600; }
  table { border-collapse:collapse; width:100%; margin-top:.6rem; }
  th,td { text-align:left; padding:.35rem .6rem; border-bottom:1px solid #8883; font-size:.92rem; }
  .num { text-align:right; font-variant-numeric:tabular-nums; }
  img { max-width:760px; width:100%; border-radius:8px; }
</style>
</head>
<body>
<h1>Jeff</h1>
<p class="sub">A local, decision-only fact-checking model — enter a claim and its evidence.</p>

<label>Claim</label>
<textarea id="claim">The Eiffel Tower is located in Barcelona.</textarea>

<label>Evidence passages (one per line)</label>
<textarea id="ev">The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.</textarea>

<button onclick="go()" id="b">Score it</button>
<span id="lat" class="muted" style="margin-left:.8rem"></span>

<div class="out" id="out"></div>

<h2>Measured on 9,730 unseen claims (human labels)</h2>
<table>
 <tr><th>model</th><th class="num">accuracy</th><th class="num">ECE</th></tr>
 <tr><td>TypeSafe Jev 1.13.0 (hosted)</td><td class="num">0.8283</td><td class="num">0.0932</td></tr>
 <tr><td><strong>Jeff</strong> (this, local 4B)</td><td class="num">0.8183</td><td class="num">0.0807</td></tr>
</table>
<p class="muted">Calibration: stated confidence vs actual correctness on human labels, with confidence defined the same way for both models (max class probability). Jev also reports its own confidence statistic, which is not max probability; scored on that it gets 0.0790, but that number is not comparable to ours, so the table uses the like-for-like 0.0932. Low-confidence behaviour is where the two differ most.</p>
<p class="muted">The accuracy table is the sealed choice-verdict benchmark; the Score and Noul cards above are a small-scale demonstration on the same adapter, not part of that metric.</p>
<img src="/static/calibration_plot.png" alt="reliability diagram">

<script>
const C = {supported:"#2a8", refuted:"#c44", not_enough_info:"#999"};
  function bars(probs){
    return Object.entries(probs||{}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>{
      const col = C[k] || "#777";
      return `<div class="lab"><span>${k}</span><span>${(v*100).toFixed(2)}%</span></div>
              <div class="bar"><div style="width:${(v*100).toFixed(1)}%;background:${col}"></div></div>`;
    }).join("");
  }
  // Levels, not probability, order a score question: the ordinal scale is the meaning.
  function scoreRows(s){
    const probs=(s&&s.probabilities)||{};
    const crit=(s&&Array.isArray(s.criteria))?s.criteria:[];
    const keys=Object.keys(probs).sort((a,b)=>(Number(a)-Number(b))||a.localeCompare(b));
    return keys.map(k=>{
      const p=probs[k]||0;
      const text=crit[Number(k)]||("level "+k);
      return `<div class="lvl"><div class="lab"><span>${k}: ${text}</span><span>${(p*100).toFixed(2)}%</span></div>
              <div class="bar"><div style="width:${(p*100).toFixed(1)}%;background:#2a8"></div></div></div>`;
    }).join("");
  }
  // Optional readout field (added by the readout layer); show it if present, nothing if not.
  function readout(q){
    const r=q&&q.readout;
    if(r===undefined||r===null) return "";
    return `<div class="muted">readout: ${typeof r==="string"?r:JSON.stringify(r)}</div>`;
  }
async function go(){
  const b=document.getElementById("b"); b.disabled=true; b.textContent="Scoring…";
  document.getElementById("lat").textContent="";
  const claim=document.getElementById("claim").value.trim();
  const evs=document.getElementById("ev").value.split("\\n").map(s=>s.trim()).filter(Boolean);
  const body={ state:{claim, evidence:evs.map((t,i)=>({title:"evidence "+(i+1), text:t}))},
    questions:{
      verdict:{type:"choice", instructions:"Read the claim and the evidence passages in `state`. Decide whether the evidence establishes the claim, contradicts it, or cannot decide it.",
        criteria:{supported:"The evidence passages together entail the claim.", refuted:"The evidence passages together contradict the claim.", not_enough_info:"No evidence, or too weak to decide."}},
      has_date:{type:"noul", instructions:"Does the evidence contain a date or a number?"},
      strength:{type:"score", instructions:"How much evidence is provided for the claim?",
        criteria:["no evidence at all","a single weak or unrelated passage","one relevant passage","several relevant passages"]}
    }};
  try{
    const r=await fetch("/v1/systemone",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    if(!r.ok) throw new Error((await r.text()).slice(0,200));
      const j=await r.json()||{};
      const v=j.verdict||{}, s=j.strength||{}, nd=j.has_date||{};
      const ny=typeof nd.noul==="number"?nd.noul:0;
      const conf=typeof v.confidence==="number"?`confidence ${(v.confidence*100).toFixed(1)}%`:"";
      const lat=typeof j.latency_ms==="number"?`latency ${j.latency_ms} ms`:"";
      const skeys=Object.keys(s.probabilities||{}).sort((a,b)=>Number(a)-Number(b));
      document.getElementById("out").innerHTML =
        `<div class="muted" style="margin-bottom:.7rem">${[j.model||"?",lat].filter(Boolean).join(" &middot; ")}</div>` +
        `<div class="card"><div class="verdict">${(v.choice||"?").toUpperCase()}</div>` + bars(v.probabilities||{}) + readout(v) +
        `<div class="muted">${[conf,lat].filter(Boolean).join(" &middot; ")}</div></div>` +
        `<div class="card"><strong>Score — how much evidence?</strong> expected ${(typeof s.score==="number"?s.score:0).toFixed(2)}` +
        (skeys.length?" / "+skeys[skeys.length-1]:"") +
        scoreRows(s) + readout(s) + `</div>` +
        `<div class="card"><strong>Noul — contains a date or number?</strong>` + readout(nd) +
        `<div class="lab"><span>yes</span><span>${(ny*100).toFixed(2)}%</span></div>` +
        `<div class="bar"><div style="width:${(ny*100).toFixed(1)}%;background:#2a8"></div></div>` +
        `<div class="lab"><span>no</span><span>${((1-ny)*100).toFixed(2)}%</span></div>` +
        `<div class="bar"><div style="width:${((1-ny)*100).toFixed(1)}%;background:#999"></div></div>` +
        `<div class="muted">P(yes) = ${ny.toFixed(4)}</div></div>`;
    document.getElementById("lat").textContent = "";
  }catch(e){
    document.getElementById("out").innerHTML = `<div class="err">error: ${e.message}</div>`;
  }finally{ b.disabled=false; b.textContent="Score it"; }
}
go();
</script>
</body>
</html>
"""


def demo_page() -> HTMLResponse:
    return HTMLResponse(PAGE)
