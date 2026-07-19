#!/usr/bin/env python3
"""Build a simple, self-contained HTML page of the matches.

One page, top to bottom, plain English: every opportunity, the companies that
match it, why, and (when the brief is rich enough) what the opportunity needs
and who covers each need. Opportunities with no match say so. Opens in any
browser, no server, no internet.

Human-in-the-loop: every match card has Agree / Not a fit buttons. Verdicts
save in the browser; "Download my reviews" exports human_reviews.csv - save it
as Data/human_reviews.csv and re-run matching_v2.py to teach the model.

Only real matches are shown (the gate's Direct and Partner verdicts) - the
hundreds of scored-but-rejected pairs stay in the Excel workbook.

Usage:
  python3 build_review_gui.py                  # reads Output/matches_v2.xlsx
  python3 build_review_gui.py --xlsx PATH --out PATH
"""
from __future__ import annotations

import argparse
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


def build_payload(xlsx: str) -> dict:
    allp = pd.read_excel(xlsx, sheet_name="All_Pairs")

    # A "match" is only what the gate actually endorsed.
    matched = allp[allp["gpt_decision"].isin(["Direct", "Partial", "Yes"])].copy()
    matched = matched.sort_values(["opportunity", "final_score"], ascending=[True, False])

    # Consortium view (optional sheet; readiness-gated upstream).
    consortium = {}
    try:
        cons = pd.read_excel(xlsx, sheet_name="Consortium_View")
        for opp_name, grp in cons.groupby("opportunity"):
            status = str(_clean(grp.iloc[0]["status"]))
            items = []
            if status == "Ready":
                for _, r in grp.iterrows():
                    items.append({
                        "need": _clean(r["need"]),
                        "quote": _clean(r["source_quote"]),
                        "covered_by": _clean(r["covered_by"]),
                        "gap": _clean(r["gap"]) == "GAP",
                    })
            consortium[opp_name] = {"status": status, "items": items}
    except Exception:
        pass

    opps = []
    for opp_name in sorted(allp["opportunity"].unique()):
        rows = matched[matched["opportunity"] == opp_name]
        sector = _clean(allp[allp["opportunity"] == opp_name].iloc[0].get("opportunity_sector", ""))
        companies = []
        for _, r in rows.iterrows():
            tier = str(_clean(r["gpt_decision"]))
            companies.append({
                "name": _clean(r["company"]),
                "sector": _clean(r.get("company_sector", "")),
                "kind": "Can build it" if tier in ("Direct", "Yes") else "Good supplier / partner",
                "why": _clean(r.get("gpt_explanation", "")),
            })
        opps.append({"name": opp_name, "sector": sector, "companies": companies,
                     "consortium": consortium.get(opp_name)})

    # Matches first, then the ones with nothing.
    opps.sort(key=lambda o: (len(o["companies"]) == 0, o["name"]))
    total = sum(len(o["companies"]) for o in opps)
    with_matches = sum(1 for o in opps if o["companies"])
    return {"opportunities": opps, "total": total, "with_matches": with_matches,
            "no_match": len(opps) - with_matches}


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Matches</title>
<style>
  :root{ --brand:__BRAND__; --bg:#f7f8f8; --card:#fff; --ink:#1b2129; --muted:#616c77; --line:#e5e9ec; }
  @media (prefers-color-scheme: dark){
    :root{--bg:#12171c;--card:#1b2229;--ink:#e9eef2;--muted:#98a4af;--line:#2b343d;}
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{background:var(--brand);color:#fff;padding:24px 20px}
  .inner{max-width:820px;margin:0 auto}
  header h1{margin:0;font-size:24px;font-weight:650}
  header p{margin:6px 0 0;opacity:.9;font-size:15px}
  header .bar{margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  header button{font:inherit;font-size:13.5px;padding:6px 14px;border-radius:8px;
       border:1px solid rgba(255,255,255,.55);background:rgba(255,255,255,.12);color:#fff;cursor:pointer}
  header .cnt{font-size:13px;opacity:.85}
  main{max-width:820px;margin:0 auto;padding:26px 20px 60px}
  .opp{margin-bottom:34px}
  .opp h2{font-size:19px;margin:0 0 2px;font-weight:650}
  .opp .sec{color:var(--muted);font-size:14px;margin-bottom:12px}
  .co{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--brand);
      border-radius:8px;padding:14px 16px;margin-bottom:10px}
  .co .n{font-weight:650;font-size:16.5px}
  .co .k{display:inline-block;font-size:13px;color:var(--brand);font-weight:600;margin-left:8px}
  .co .s{color:var(--muted);font-size:13.5px;margin-top:1px}
  .co .w{margin-top:8px;font-size:15px}
  .rev{margin-top:10px;display:flex;gap:8px}
  .rev button{font:inherit;font-size:13px;padding:4px 12px;border-radius:8px;
      border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}
  .rev button.agree.on{background:#1f8a4c;border-color:#1f8a4c;color:#fff}
  .rev button.disagree.on{background:#b23b3b;border-color:#b23b3b;color:#fff}
  .none{background:var(--card);border:1px dashed var(--line);border-radius:8px;padding:14px 16px;
        color:var(--muted);font-size:15px}
  .needs{background:var(--card);border:1px solid var(--line);border-radius:8px;
        padding:13px 16px;margin:4px 0 10px}
  .needs h3{margin:0 0 8px;font-size:14.5px;color:var(--brand)}
  .needs .item{font-size:14.5px;margin:5px 0}
  .needs .gap{color:#b23b3b;font-weight:600}
  .needs .cov{color:var(--muted)}
  .thin{color:var(--muted);font-size:13.5px;font-style:italic;margin:2px 0 10px}
  .divider{margin:38px 0 22px;border-top:1px solid var(--line);padding-top:18px}
  .divider h3{margin:0 0 4px;font-size:16px}
  .divider p{margin:0;color:var(--muted);font-size:14.5px}
</style>
</head>
<body>
<header><div class="inner">
  <h1>Company matches</h1>
  <p id="sum"></p>
  <div class="bar">
    <button onclick="__export()">Download my reviews</button>
    <span class="cnt" id="cnt"></span>
    <span class="cnt">Save the file as Data/human_reviews.csv and re-run the model to teach it.</span>
  </div>
</div></header>
<main id="main"></main>
<script>
/*__DATA__*/
(function(){
  var D = window.__DATA__;
  var LS = "match_reviews_v1";
  var reviews = {};
  try { reviews = JSON.parse(localStorage.getItem(LS) || "{}"); } catch(e){ reviews = {}; }
  function save(){ try{ localStorage.setItem(LS, JSON.stringify(reviews)); }catch(e){} updateCnt(); }
  function updateCnt(){
    var n = Object.keys(reviews).filter(function(k){return reviews[k];}).length;
    document.getElementById("cnt").textContent = n + " reviewed";
  }
  var esc = function(s){ var d=document.createElement("div"); d.textContent=(s==null?"":String(s)); return d.innerHTML; };

  document.getElementById("sum").textContent =
    D.total + " matches found across " + D.with_matches + " of " +
    D.opportunities.length + " opportunities. " + D.no_match + " have no suitable company.";

  window.__export = function(){
    var rows = [["company","opportunity","verdict"]];
    Object.keys(reviews).forEach(function(k){
      if(!reviews[k]) return;
      var p = k.split("||");
      rows.push([p[0], p[1], reviews[k]]);
    });
    var csv = rows.map(function(r){ return r.map(function(v){
      return '"' + String(v==null?"":v).replace(/"/g,'""') + '"'; }).join(","); }).join("\n");
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
    a.download = "human_reviews.csv"; a.click();
  };

  var html = "", shownDivider = false;
  D.opportunities.forEach(function(o){
    if(!o.companies.length && !shownDivider){
      shownDivider = true;
      html += '<div class="divider"><h3>No suitable company found</h3>'+
              '<p>Nothing in the current company list can build or supply these.</p></div>';
    }
    html += '<div class="opp"><h2>'+esc(o.name)+'</h2><div class="sec">'+esc(o.sector)+'</div>';
    if(o.consortium){
      if(o.consortium.status === "Ready" && o.consortium.items.length){
        html += '<div class="needs"><h3>What this opportunity needs</h3>';
        o.consortium.items.forEach(function(it){
          html += '<div class="item">'+esc(it.need)+' — '+
            (it.gap ? '<span class="gap">GAP: no validated company covers this</span>'
                    : '<span class="cov">covered by '+esc(it.covered_by)+'</span>')+'</div>';
        });
        html += '</div>';
      } else if(o.consortium.status && o.consortium.status !== "Ready"){
        html += '<div class="thin">Needs breakdown unavailable: the opportunity brief lacks explicit detail.</div>';
      }
    }
    if(o.companies.length){
      o.companies.forEach(function(c){
        var id = c.name + "||" + o.name;
        html += '<div class="co"><div><span class="n">'+esc(c.name)+'</span>'+
                '<span class="k">'+esc(c.kind)+'</span></div>'+
                '<div class="s">'+esc(c.sector)+'</div>'+
                '<div class="w">'+esc(c.why)+'</div>'+
                '<div class="rev" data-id="'+esc(id)+'">'+
                '<button class="agree">Agree</button>'+
                '<button class="disagree">Not a fit</button></div></div>';
      });
    } else {
      html += '<div class="none">No company in the list is a credible fit.</div>';
    }
    html += '</div>';
  });
  document.getElementById("main").innerHTML = html;

  document.querySelectorAll(".rev").forEach(function(bar){
    var id = bar.getAttribute("data-id");
    var a = bar.querySelector(".agree"), d = bar.querySelector(".disagree");
    function paint(){
      a.classList.toggle("on", reviews[id] === "agree");
      d.classList.toggle("on", reviews[id] === "disagree");
    }
    a.onclick = function(){ reviews[id] = (reviews[id]==="agree" ? "" : "agree"); save(); paint(); };
    d.onclick = function(){ reviews[id] = (reviews[id]==="disagree" ? "" : "disagree"); save(); paint(); };
    paint();
  });
  updateCnt();
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
    payload = build_payload(args.xlsx)
    page = (TEMPLATE.replace("__BRAND__", BRAND)
            .replace("/*__DATA__*/", "window.__DATA__ = " + json.dumps(payload, ensure_ascii=False) + ";"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"Wrote {args.out}: {payload['total']} matches across "
          f"{payload['with_matches']} opportunities, {payload['no_match']} with none.")


if __name__ == "__main__":
    main()
