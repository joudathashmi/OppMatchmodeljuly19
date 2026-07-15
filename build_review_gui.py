#!/usr/bin/env python3
"""Build a self-contained HTML review page from Output/matches_v2.xlsx.

The page opens in any browser with no server and no internet: browse every
opportunity's candidates, see the graded gate verdict / confidence / agreement /
explanation, filter by tier, and record your own Agree / Disagree / Unsure
verdict plus notes. Evaluations persist in the browser (localStorage) and export
to CSV.

Usage:
  python3 build_review_gui.py                 # reads Output/matches_v2.xlsx
  python3 build_review_gui.py --xlsx PATH --out PATH
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os

import pandas as pd

BRAND = "#02714E"  # MISA green


def _clean(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return v


def load(xlsx: str):
    allp = pd.read_excel(xlsx, sheet_name="All_Pairs")
    try:
        abst = pd.read_excel(xlsx, sheet_name="Abstentions")
    except Exception:
        abst = pd.DataFrame()
    try:
        diag = pd.read_excel(xlsx, sheet_name="Diagnostics")
    except Exception:
        diag = pd.DataFrame()
    return allp, abst, diag


def build_payload(allp: pd.DataFrame, abst: pd.DataFrame, diag: pd.DataFrame) -> dict:
    # Only qualified candidates are worth reviewing; keep all of them per opp.
    q = allp[allp["qualified"]].copy()
    q = q.sort_values(["opportunity", "rank_for_opportunity"], na_position="last")

    def row(r):
        gpt = str(_clean(r.get("gpt_decision", "")))
        return {
            "id": f"{_clean(r['company'])}||{_clean(r['opportunity'])}",
            "company": _clean(r["company"]),
            "company_sector": _clean(r.get("company_sector", "")),
            "opportunity": _clean(r["opportunity"]),
            "opportunity_sector": _clean(r.get("opportunity_sector", "")),
            "label": _clean(r.get("ai_decision", "")),
            "gpt": gpt,
            "confidence": _clean(r.get("gpt_confidence", "")),
            "agreement": _clean(r.get("gpt_agreement", "")),
            "final": _clean(r.get("final_score", "")),
            "sector_label": _clean(r.get("sector_label", "")),
            "bridge": _clean(r.get("bridge", "")),
            "evidence": _clean(r.get("evidence_terms", "")),
            "explanation": _clean(r.get("gpt_explanation", "")) or _clean(r.get("sector_reason", "")),
            "gpt_backed": gpt in ("Direct", "Partial", "No", "Yes"),
        }

    opps = []
    for opp_name, grp in q.groupby("opportunity"):
        cands = [row(r) for _, r in grp.iterrows()]
        opps.append({
            "name": opp_name,
            "sector": cands[0]["opportunity_sector"] if cands else "",
            "n": len(cands),
            "direct": sum(1 for c in cands if c["gpt"] in ("Direct", "Yes")),
            "partner": sum(1 for c in cands if c["gpt"] == "Partial"),
            "candidates": cands,
        })
    opps.sort(key=lambda o: (-(o["direct"] * 10 + o["partner"]), o["name"]))

    abstentions = []
    if len(abst) and "opportunity" in abst.columns:
        for _, r in abst.iterrows():
            if str(_clean(r.get("opportunity", ""))) in ("", "-"):
                continue
            abstentions.append({
                "opportunity": _clean(r.get("opportunity", "")),
                "status": _clean(r.get("status", "")),
                "best": _clean(r.get("best_candidate", "")),
                "detail": _clean(r.get("detail", "")),
            })

    diag_rows = [{"metric": _clean(r["metric"]), "value": _clean(r["value"])}
                 for _, r in diag.iterrows()] if len(diag) else []
    return {"opportunities": opps, "abstentions": abstentions, "diagnostics": diag_rows}


def render(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    # NOTE: the page is a plain string template; the only interpolation is the
    # JSON payload and the brand colour, both controlled here.
    return TEMPLATE.replace("__BRAND__", BRAND).replace(
        "/*__DATA__*/", "window.__DATA__ = " + data_json + ";"
    )


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Match Review</title>
<style>
  :root{
    --brand:__BRAND__; --bg:#f6f7f8; --card:#fff; --ink:#1a2129; --muted:#5b6772;
    --line:#e4e8eb; --direct:#02714E; --partner:#0d8f9e; --review:#c07a12; --low:#8a94a0;
  }
  @media (prefers-color-scheme: dark){
    :root{--bg:#12171c;--card:#1b2229;--ink:#e8edf1;--muted:#9aa6b1;--line:#2a333c;}
  }
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       background:var(--bg);color:var(--ink)}
  header{background:var(--brand);color:#fff;padding:18px 22px}
  header h1{margin:0;font-size:19px;font-weight:650;letter-spacing:.2px}
  header .sub{opacity:.85;font-size:13px;margin-top:3px}
  .wrap{display:flex;min-height:calc(100vh - 62px)}
  .side{width:320px;flex:0 0 320px;border-right:1px solid var(--line);background:var(--card);
        overflow:auto;max-height:calc(100vh - 62px);position:sticky;top:0}
  .main{flex:1;padding:20px 24px;overflow:auto;max-height:calc(100vh - 62px)}
  .tiles{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
  .tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:96px}
  .tile b{display:block;font-size:22px;font-weight:680}
  .tile span{font-size:12px;color:var(--muted)}
  .opp{padding:11px 16px;border-bottom:1px solid var(--line);cursor:pointer}
  .opp:hover{background:rgba(2,113,78,.06)}
  .opp.active{background:rgba(2,113,78,.12);border-left:3px solid var(--brand)}
  .opp .t{font-weight:600;font-size:13.5px}
  .opp .m{font-size:11.5px;color:var(--muted);margin-top:3px;display:flex;gap:6px;flex-wrap:wrap}
  .pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;font-weight:600;color:#fff}
  .pill.direct{background:var(--direct)} .pill.partner{background:var(--partner)}
  .pill.review{background:var(--review)} .pill.low{background:var(--low)} .pill.abst{background:#b23b3b}
  .filters{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:6px;flex-wrap:wrap}
  .filters button{font:inherit;font-size:12px;padding:4px 10px;border:1px solid var(--line);
       background:var(--card);color:var(--ink);border-radius:20px;cursor:pointer}
  .filters button.on{background:var(--brand);color:#fff;border-color:var(--brand)}
  h2.opp-title{margin:0 0 2px;font-size:20px}
  .opp-sub{color:var(--muted);font-size:13px;margin-bottom:16px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px 17px;margin-bottom:13px}
  .card .top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
  .card .co{font-weight:650;font-size:16px}
  .card .cs{color:var(--muted);font-size:12.5px;margin-top:1px}
  .badges{display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
  .meta{display:flex;gap:14px;flex-wrap:wrap;margin:11px 0 8px;font-size:12.5px;color:var(--muted)}
  .meta b{color:var(--ink);font-weight:600}
  .bar{height:6px;border-radius:4px;background:var(--line);overflow:hidden;width:120px;display:inline-block;vertical-align:middle}
  .bar>i{display:block;height:100%;background:var(--brand)}
  .evi{margin:8px 0}
  .chip{display:inline-block;background:rgba(2,113,78,.10);color:var(--brand);border-radius:6px;
        padding:2px 8px;font-size:11.5px;margin:2px 4px 2px 0}
  @media (prefers-color-scheme: dark){.chip{background:rgba(13,143,158,.18);color:#5fd0dc}}
  .expl{font-size:13.5px;line-height:1.55;margin-top:6px}
  .evalbar{display:flex;gap:8px;align-items:center;margin-top:12px;padding-top:12px;border-top:1px dashed var(--line);flex-wrap:wrap}
  .evalbar button{font:inherit;font-size:12.5px;padding:5px 12px;border-radius:8px;border:1px solid var(--line);
        background:var(--card);color:var(--ink);cursor:pointer}
  .evalbar button.agree.on{background:#1f8a4c;color:#fff;border-color:#1f8a4c}
  .evalbar button.disagree.on{background:#b23b3b;color:#fff;border-color:#b23b3b}
  .evalbar button.unsure.on{background:#c07a12;color:#fff;border-color:#c07a12}
  .evalbar input{flex:1;min-width:160px;font:inherit;font-size:12.5px;padding:5px 9px;border:1px solid var(--line);
        border-radius:8px;background:var(--bg);color:var(--ink)}
  .toolbar{display:flex;gap:8px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
  .toolbar button{font:inherit;font-size:13px;padding:6px 13px;border-radius:8px;border:1px solid var(--brand);
        background:var(--brand);color:#fff;cursor:pointer}
  .toolbar .ghost{background:transparent;color:var(--brand)}
  .toolbar .count{color:var(--muted);font-size:12.5px}
  .abst{background:rgba(178,59,59,.07);border:1px solid rgba(178,59,59,.25);border-radius:10px;padding:12px 15px}
  .abst h3{margin:0 0 8px;font-size:14px}
  .abst .row{font-size:13px;margin:4px 0}
  .muted{color:var(--muted)} .hidden{display:none}
</style>
</head>
<body>
<header>
  <h1>Company &times; Opportunity &mdash; Match Review</h1>
  <div class="sub">Graded gate verdicts with self-consistency voting. Your evaluations save in this browser and export to CSV.</div>
</header>
<div class="wrap">
  <div class="side">
    <div class="filters" id="filters">
      <button data-f="all" class="on">All</button>
      <button data-f="direct">Direct</button>
      <button data-f="partner">Partner</button>
      <button data-f="rest">Review/Low</button>
    </div>
    <div id="opplist"></div>
  </div>
  <div class="main" id="main"></div>
</div>
<script>
/*__DATA__*/
(function(){
  var D = window.__DATA__ || {opportunities:[],abstentions:[],diagnostics:[]};
  var LS = "match_eval_v1";
  var evals = {};
  try { evals = JSON.parse(localStorage.getItem(LS) || "{}"); } catch(e){ evals = {}; }
  function save(){ try{ localStorage.setItem(LS, JSON.stringify(evals)); }catch(e){} }
  var esc = function(s){ var d=document.createElement("div"); d.textContent=(s==null?"":String(s)); return d.innerHTML; };
  var filter = "all", current = 0;

  function tierClass(g,label){
    if(g==="Direct"||g==="Yes") return "direct";
    if(g==="Partial") return "partner";
    if(label==="Review Needed") return "review";
    return "low";
  }
  function tierText(g,label){
    if(g==="Direct"||g==="Yes") return "Direct";
    if(g==="Partial") return "Partner";
    if(g==="No") return "Rejected";
    return label||"";
  }

  function diagVal(m){ var r=(D.diagnostics||[]).find(function(x){return x.metric===m}); return r?r.value:"-"; }

  function renderTiles(){
    var t=[
      ["Opportunities", D.opportunities.length],
      ["Direct fits", diagVal("pairs_direct_fit")],
      ["Partner fits", diagVal("pairs_partner_fit")],
      ["Abstentions", (D.abstentions||[]).length],
      ["Embeddings", diagVal("embedding_mode")]
    ];
    return '<div class="tiles">'+t.map(function(x){
      return '<div class="tile"><b>'+esc(x[1])+'</b><span>'+esc(x[0])+'</span></div>';
    }).join("")+'</div>';
  }

  function oppMatches(o){
    if(filter==="all") return true;
    if(filter==="direct") return o.direct>0;
    if(filter==="partner") return o.partner>0;
    if(filter==="rest") return o.direct===0 && o.partner===0;
    return true;
  }

  function renderList(){
    var el=document.getElementById("opplist"); el.innerHTML="";
    D.opportunities.forEach(function(o,i){
      if(!oppMatches(o)) return;
      var badges="";
      if(o.direct>0) badges+='<span class="pill direct">'+o.direct+' Direct</span>';
      if(o.partner>0) badges+='<span class="pill partner">'+o.partner+' Partner</span>';
      if(o.direct===0&&o.partner===0) badges+='<span class="pill abst">No validated fit</span>';
      var div=document.createElement("div");
      div.className="opp"+(i===current?" active":"");
      div.innerHTML='<div class="t">'+esc(o.name)+'</div><div class="m"><span>'+esc(o.sector)+
        '</span><span>&middot; '+o.n+' candidates</span></div><div class="m">'+badges+'</div>';
      div.onclick=function(){ current=i; render(); };
      el.appendChild(div);
    });
  }

  function evalButtons(id){
    var e=evals[id]||{};
    function b(k,txt){ return '<button class="'+k+(e.verdict===k?" on":"")+'" data-id="'+esc(id)+'" data-v="'+k+'">'+txt+'</button>'; }
    return '<div class="evalbar">'+b("agree","Agree")+b("disagree","Disagree")+b("unsure","Unsure")+
      '<input type="text" placeholder="notes" data-note="'+esc(id)+'" value="'+esc(e.note||"")+'"></div>';
  }

  function card(c){
    var cls=tierClass(c.gpt,c.label), tt=tierText(c.gpt,c.label);
    var conf=(c.confidence!==""&&c.confidence!=null)?" &middot; conf "+esc(c.confidence):"";
    var agr=c.agreement?(" &middot; agree "+esc(c.agreement)):"";
    var pct=Math.round((parseFloat(c.final)||0)*100);
    var evi=(c.evidence?String(c.evidence).split(",").map(function(t){t=t.trim();return t?'<span class="chip">'+esc(t)+'</span>':"";}).join(""):"");
    var bridge=c.bridge?'<span> &middot; bridge: <b>'+esc(c.bridge)+'</b></span>':"";
    return '<div class="card">'+
      '<div class="top"><div><div class="co">'+esc(c.company)+'</div><div class="cs">'+esc(c.company_sector)+'</div></div>'+
      '<div class="badges"><span class="pill '+cls+'">'+esc(tt)+'</span></div></div>'+
      '<div class="meta"><span>Label: <b>'+esc(c.label)+'</b></span>'+
        '<span>Gate: <b>'+esc(c.gpt||"not graded")+'</b>'+conf+agr+'</span>'+
        '<span>Score: <span class="bar"><i style="width:'+pct+'%"></i></span> <b>'+esc(c.final)+'</b></span>'+
        '<span>Sector: <b>'+esc(c.sector_label)+'</b>'+bridge+'</span></div>'+
      (evi?'<div class="evi">'+evi+'</div>':"")+
      '<div class="expl">'+esc(c.explanation)+'</div>'+
      evalButtons(c.id)+
      '</div>';
  }

  function cardVisible(c){
    if(filter==="all") return true;
    if(filter==="direct") return c.gpt==="Direct"||c.gpt==="Yes";
    if(filter==="partner") return c.gpt==="Partial";
    if(filter==="rest") return !(c.gpt==="Direct"||c.gpt==="Yes"||c.gpt==="Partial");
    return true;
  }

  function evalCount(){ return Object.keys(evals).filter(function(k){return evals[k]&&evals[k].verdict}).length; }

  function exportCsv(){
    var rows=[["opportunity","company","gate","label","confidence","agreement","final_score","your_verdict","notes"]];
    D.opportunities.forEach(function(o){ o.candidates.forEach(function(c){
      var e=evals[c.id]||{};
      rows.push([o.name,c.company,c.gpt,c.label,c.confidence,c.agreement,c.final,e.verdict||"",(e.note||"").replace(/\n/g," ")]);
    });});
    var csv=rows.map(function(r){return r.map(function(v){v=(v==null?"":String(v));return '"'+v.replace(/"/g,'""')+'"';}).join(",");}).join("\n");
    var blob=new Blob([csv],{type:"text/csv"});
    var a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="match_evaluations.csv"; a.click();
  }

  function renderMain(){
    var o=D.opportunities[current]; var m=document.getElementById("main");
    if(!o){ m.innerHTML=renderTiles()+"<p class='muted'>No opportunity selected.</p>"; return; }
    var cands=o.candidates.filter(cardVisible);
    var abstHtml="";
    var ab=(D.abstentions||[]).filter(function(a){return a.opportunity===o.name;});
    if(ab.length){ abstHtml='<div class="abst"><h3>Abstention</h3>'+ab.map(function(a){
      return '<div class="row"><b>'+esc(a.status)+'</b> &middot; best (rejected): '+esc(a.best)+
        (a.detail?'<div class="muted">'+esc(a.detail)+'</div>':"")+'</div>';}).join("")+'</div>'; }
    m.innerHTML=renderTiles()+
      '<div class="toolbar"><button onclick="__export()">Export evaluations (CSV)</button>'+
      '<button class="ghost" onclick="__reset()">Clear my evaluations</button>'+
      '<span class="count">'+evalCount()+' evaluated</span></div>'+
      '<h2 class="opp-title">'+esc(o.name)+'</h2>'+
      '<div class="opp-sub">'+esc(o.sector)+' &middot; '+o.n+' qualified candidates &middot; '+
        o.direct+' Direct, '+o.partner+' Partner</div>'+
      abstHtml+
      (cands.length?cands.map(card).join(""):"<p class='muted'>No candidates match this filter.</p>");
    // wire eval controls
    m.querySelectorAll(".evalbar button").forEach(function(btn){
      btn.onclick=function(){
        var id=btn.getAttribute("data-id"), v=btn.getAttribute("data-v");
        evals[id]=evals[id]||{}; evals[id].verdict=(evals[id].verdict===v?"":v); save(); renderMain();
      };
    });
    m.querySelectorAll("input[data-note]").forEach(function(inp){
      inp.oninput=function(){ var id=inp.getAttribute("data-note"); evals[id]=evals[id]||{}; evals[id].note=inp.value; save(); };
    });
  }

  window.__export=exportCsv;
  window.__reset=function(){ if(confirm("Clear all your evaluations on this page?")){ evals={}; save(); renderMain(); } };

  function render(){ renderList(); renderMain(); }

  document.getElementById("filters").addEventListener("click",function(e){
    var b=e.target.closest("button"); if(!b) return;
    filter=b.getAttribute("data-f");
    document.querySelectorAll("#filters button").forEach(function(x){x.classList.toggle("on",x===b);});
    current=0; render();
  });

  render();
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", default="Output/matches_v2.xlsx")
    ap.add_argument("--out", default="Output/matches_review.html")
    args = ap.parse_args()

    if not os.path.exists(args.xlsx):
        raise SystemExit(f"Not found: {args.xlsx} (run matching_v2.py first).")
    allp, abst, diag = load(args.xlsx)
    payload = build_payload(allp, abst, diag)
    page = render(payload)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(page)
    n_cand = sum(o["n"] for o in payload["opportunities"])
    print(f"Wrote {args.out}  ({len(payload['opportunities'])} opportunities, "
          f"{n_cand} candidates, {len(payload['abstentions'])} abstentions).")


if __name__ == "__main__":
    main()
