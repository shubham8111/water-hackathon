"""Build two insight-first, chart-led reports: outputs/KRS_insights.html and outputs/Hemavathy_insights.html.

Reads outputs/dam_results.json (run dam_analysis.py first), outputs/event_results.json, outputs/inflow_results.json
and outputs/data_fixes.json. Self-contained: d3 and all data are inlined, so the files open offline.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
NONE, F30 = "No rain forecast", "+ next 30 days"
ISSUES = [(6, 1), (6, 15), (7, 1), (7, 15), (8, 1), (8, 15), (9, 1), (9, 15)]
GOOD = 20  # "reasonably informed": 4 years in 5 within +/-20% of the actual monsoon total
DAMS = {
    "krs": {"name": "Krishnaraja Sagar (KRS)", "short": "KRS", "file": "KRS_insights.html", "cap": 49.45},
    "hemavathy": {"name": "Hemavathy", "short": "Hemavathy", "file": "Hemavathy_insights.html", "cap": 37.10},
}


def issue_label(s):
    d, m = s.split()
    return f"{int(d)} {m}"


TMC = 86400 / 1e9


def eda(dam, df, R):
    """One row per year: Jun-Sep catchment rain, inflow, outflow; storage on 1 Jun and 30 Sep."""
    w = pd.Series({c["cell_id"]: c["frac"] for c in R["cells"][dam]})
    rain = (df[w.index] * w).sum(axis=1) / w.sum()
    cap = DAMS[dam]["cap"]
    rows = []
    for y in range(1991, 2026):
        a, b = pd.Timestamp(y, 6, 1), pd.Timestamp(y, 9, 30)
        s0, s1 = df.at[a, f"{dam}_storage"], df.at[b, f"{dam}_storage"]
        rows.append({"year": y, "rain": round(float(rain[a:b].sum())),
                     "inflow": round(float(df.loc[a:b, f"{dam}_inflow"].sum() * TMC), 1),
                     "outflow": round(float(df.loc[a:b, f"{dam}_outflow"].sum() * TMC), 1),
                     "s_jun1": None if pd.isna(s0) else round(float(s0), 1),
                     "s_sep30": None if pd.isna(s1) else round(float(s1), 1),
                     "pct_jun1": None if pd.isna(s0) else round(float(s0 / cap * 100), 1),
                     "pct_sep30": None if pd.isna(s1) else round(float(s1 / cap * 100), 1)})
    t = pd.DataFrame(rows)
    dry = t.inflow < t.inflow.quantile(1 / 3)
    t["dry"] = dry
    st = {}
    for k in ["rain", "inflow", "outflow"]:
        st[f"dry_{k}_pct"] = round(float((t.loc[dry, k].mean() / t.loc[~dry, k].mean() - 1) * 100))
    st["dry_full_pct"], st["wet_full_pct"] = round(float(t.loc[dry, "pct_sep30"].mean())), round(float(t.loc[~dry, "pct_sep30"].mean()))
    b1, b0 = np.polyfit(t.rain, t.inflow, 1)
    st["rain_slope_per_100mm"] = round(float(b1 * 100), 1)
    st["r_rain_inflow"] = round(float(np.corrcoef(t.rain, t.inflow)[0, 1]), 2)
    st["r_inflow_storage"] = round(float(t[["inflow", "pct_sep30"]].dropna().corr().iloc[0, 1]), 2)
    st["r_rain_storage"] = round(float(t[["rain", "pct_sep30"]].dropna().corr().iloc[0, 1]), 2)
    st["release_share"] = round(float((t.outflow / t.inflow).median() * 100))
    st["fit"] = [round(float(b0), 2), round(float(b1), 4)]
    # inflow above which (nearly) every year ends September >= 90% full
    full = t.dropna(subset=["pct_sep30"]).sort_values("inflow")
    st["fill_threshold"], st["fill_above"], st["fill_below"] = None, None, None
    for v in full.inflow:
        above = full[full.inflow >= v]
        if (above.pct_sep30 >= 90).mean() >= 0.9 and len(above) >= 5:
            below = full[full.inflow < v]
            st["fill_threshold"] = round(float(v))
            st["fill_above"] = f"{int((above.pct_sep30 >= 90).sum())}/{len(above)}"
            st["fill_below"] = f"{int((below.pct_sep30 >= 90).sum())}/{len(below)}"
            break
    st["mean"] = {k: round(float(t[k].mean()), 1) for k in ["rain", "inflow", "outflow", "pct_sep30"]}
    st["range"] = {k: [float(t[k].min()), float(t[k].max())] for k in ["rain", "inflow", "outflow", "pct_sep30"]}
    return t.to_dict("records"), st


def ladder(dam, R, cc):
    """Every daily-inflow approach we tried, scored the same way on 2020-2025."""
    daily, rp, bp, pk = R["daily"], R["river_experiments"], R["baseline_peaks"], R["peaks"]
    fromd = lambda k: (daily[k]["all"]["NSE"], (bp.get(k) or pk.get(k)))
    fromr = lambda k: (rp[k]["all"]["NSE"], rp[k]["peaks"])
    fromc = lambda k: (cc[k]["NSE"], cc[k])
    if dam == "krs":
        items = [("Simple baselines", "Average for the day of year", fromd("krs|Climatology")),
                 ("Simple baselines", "Same as yesterday", fromd("krs|Persistence (yesterday)")),
                 ("Rain only", "Rain grouped in 3 distance bands", fromr("krs (rain only)|Straight-line bands")),
                 ("Rain only", "Rain with per-cell response + soil moisture", fromd("krs|Rain only")),
                 ("Rain + upstream releases", "Rain in 3 distance bands", fromc("Current: distance bands")),
                 ("Rain + upstream releases", "Bands by distance along the river", fromr("krs|River-distance bands")),
                 ("Rain + upstream releases", "Raw rain of every cell", fromc("+ raw rain of every cell (lags 0-5)")),
                 ("Rain + upstream releases", "Per-cell response", fromc("+ per-cell routed rain")),
                 ("Rain + upstream releases", "Per-cell response + soil moisture (used in this report)", fromd("krs|Rain + upstream releases")),
                 ("Adding last week's inflow", "Final model + last week's inflow", fromd("krs|Rain + upstream + past inflow"))]
    else:
        items = [("Simple baselines", "Average for the day of year", fromd("hemavathy|Climatology")),
                 ("Simple baselines", "Same as yesterday", fromd("hemavathy|Persistence (yesterday)")),
                 ("Rain", "Rain in 3 distance bands", fromc("Current: distance bands")),
                 ("Rain", "Bands by distance along the river", fromr("hemavathy|River-distance bands")),
                 ("Rain", "Raw rain of every cell", fromc("+ raw rain of every cell (lags 0-5)")),
                 ("Rain", "Per-cell response", fromc("+ per-cell routed rain")),
                 ("Rain", "Per-cell response + soil moisture (used in this report)", fromd("hemavathy|Rain only")),
                 ("Adding last week's inflow", "Final model + last week's inflow", fromd("hemavathy|Rain + past inflow"))]
    return [{"group": g, "label": lab, "nse": round(float(n), 3), "peak": p["peak_median_err_pct"],
             "within25": round(p["peak_within_25pct"] * 100), "rise": round(p["rise_capture_median"] * 100),
             "final": "used in this report" in lab} for g, lab, (n, p) in items]


def dam_data(dam, R, inflow, fixes, CR, df, SOIL, EV):
    mi = R["monsoon_inflow"][dam]
    fc = []
    for r in mi["by_issue"]:
        fc.append({"issue": issue_label(r["issue_date"]), "seen": r["already_observed_pct"],
                   "clim_med": r["climatology_median_abs_pct_err"], "clim_p80": r["climatology_p80_abs_pct_err"],
                   "none_med": r[NONE]["median_abs_pct_err"], "none_p80": r[NONE]["p80_abs_pct_err"],
                   "f30_med": r[F30]["median_abs_pct_err"], "f30_p80": r[F30]["p80_abs_pct_err"],
                   "none_dry": r[NONE]["dry_years_caught"], "f30_dry": r[F30]["dry_years_caught"]})
    first = lambda key: next((f["issue"] for f in fc if f[key] <= GOOD), None)
    per_year = {issue_label(r["issue_date"]): [{"year": y["year"], "actual": y["actual"], "none": y[NONE], "f30": y[F30]}
                                               for y in r["years"]] for r in mi["by_issue"]}
    after = mi["after_aug1_pct"]
    years = [{"year": int(y), "total": v, "after": after.get(y, after.get(str(y)))} for y, v in mi["by_year"].items()]
    srt = sorted(after.values())
    d = {"dam": dam, **DAMS[dam], "mean_total": mi["mean_total_tmc"], "min_total": mi["min_tmc"],
         "max_total": mi["max_tmc"], "forecast": fc, "per_year": per_year, "years": years,
         "after_aug1_median": srt[len(srt) // 2],
         "reliable_from": first("none_p80"), "reliable_from_rain": first("f30_p80"),
         "area_km2": R["inputs"][f"{dam}_watershed_km2"],
         "hydro": R["hydrograph"][dam], "lead": R["lead_time"][dam],
         "fixes": [f for f in fixes if f["dataset"] == dam]}
    d["at"] = {f["issue"]: f for f in fc}
    d["ladder"] = ladder(dam, R, CR[dam]["daily_comparison"])
    so = SOIL[dam]
    d["soil"] = {"without": so["daily"]["Without soil data"], "with": so["daily"]["With soil data"],
                 "sub_watersheds": so["sub_watersheds"], "months": so["months"],
                 "corr": so["checks"]["soil_vs_wetness_index_corr"], "corr_anom": so["checks"]["soil_vs_wetness_index_corr_anomaly"],
                 "monsoon": [{"issue": issue_label(a["issue"]), "without": a["p80"], "with": b["p80"]}
                             for a, b in zip(so["monsoon_total"]["Without soil data"], so["monsoon_total"]["With soil data"])]}
    se = EV[dam]["season"]
    reg_dos = {(pd.Timestamp(2001, m, dd) - pd.Timestamp(2001, 6, 1)).days: f"{dd} {pd.Timestamp(2001, m, 1):%b}" for m, dd in ISSUES}
    traj = {}
    for y, t in se["trajectory"].items():
        pts = [{"dos": k, "date": lab, "f": t["forecast"][k], "heavy": False} for k, lab in reg_dos.items()]
        pts += [{"dos": e["dos"], "date": e["date"], "f": e["forecast"], "heavy": True, "rain": e["rain_mm"],
                 "prev": e["prev"], "prev_date": e["prev_date"]} for e in t["events"]]
        traj[y] = {"actual": t["actual"], "rain": t["rain"], "pts": sorted(pts, key=lambda r: r["dos"])}
    d["events"] = {"summary": se["summary"], "by_month": se["by_month"], "traj": traj,
                   "default": max(traj, key=lambda y: traj[y]["actual"])}
    if dam == "hemavathy":
        d["paper"] = EV["paper"]
    n15 = EV[dam]["next15"]
    d["next15"] = {"models": n15["models"], "series": n15["series"], "n_heavy": n15["n_heavy_test_days"],
                   "mean_volume": n15["mean_volume_tmc"]}
    other = "hemavathy" if dam == "krs" else "krs"
    d["soil_other"] = {"name": DAMS[other]["short"], "without": SOIL[other]["daily"]["Without soil data"]["NSE"],
                       "with": SOIL[other]["daily"]["With soil data"]["NSE"]}
    order = [("Assume an average year", None), ("Model, no rain forecast", NONE), ("+ next 7 days of rain", "+ next 7 days of rain"),
             ("+ next 15 days of rain", "+ next 15 days"), ("+ next 30 days of rain", F30), ("+ all rain up to 30 Sep", "+ rest of season (to 30 Sep)")]
    d["matrix"] = {"issues": [issue_label(r["issue_date"]) for r in mi["by_issue"]],
                   "rows": [{"label": lab,
                             "p80": [r["climatology_p80_abs_pct_err"] if k is None else r[k]["p80_abs_pct_err"] for r in mi["by_issue"]],
                             "med": [r["climatology_median_abs_pct_err"] if k is None else r[k]["median_abs_pct_err"] for r in mi["by_issue"]]}
                            for lab, k in order]}
    d["eda"], d["eda_stats"] = eda(dam, df, R)
    cr = CR[dam]
    d["cells"], d["geo"], d["held"] = cr["cells"], cr["geo"], cr.get("held_by_hemavathy", [])
    d["runoff_share"] = cr["catchment_runoff_share_pct"]
    cmp_ = cr["daily_comparison"]
    d["cell_cmp"] = {"before": cmp_["Current: distance bands"], "after": cmp_["+ per-cell routed rain"]}
    act = [c for c in cr["cells"] if c["gain_cusec_days_per_mm"] > 1]
    d["cells_active"], d["cells_total"] = len(act), len(cr["cells"])
    top = sorted(cr["cells"], key=lambda c: -c["gain_cusec_days_per_mm"])
    tot = sum(c["gain_cusec_days_per_mm"] for c in top)
    gmax = max(c["gain_cusec_days_per_mm"] for c in cr["cells"])
    d["cells_quiet"] = sum(c["gain_cusec_days_per_mm"] < 0.05 * gmax for c in cr["cells"])
    d["top3_share"] = round(sum(c["gain_cusec_days_per_mm"] for c in top[:3]) / tot * 100)
    dl = sorted(c["delay_days"] for c in act)
    d["delay_range"] = [dl[0], dl[-1]]
    if dam == "krs":
        c = inflow["lag_response"]["coefficients"]
        d["lag_rain"] = {b: [a + w for a, w in zip(c[f"rain_{b}"], c[f"rainwet_{b}"])] for b in ["near", "mid", "far"]}
        d["lag_up"] = {"Harangi": c["harangi"], "Hemavathy": c["hemavathy"]}
        d["peaks"] = R["peaks"]["krs|Rain + upstream releases"]
        d["daily_nse"] = R["daily"]["krs|Rain + upstream releases"]["all"]["NSE"]
        d["importance"] = inflow["importance"]
        ex = R["river_experiments"]
        d["river"] = [{"label": "Straight-line distance (current)", "v": ex["krs|Straight-line bands (current)"]},
                      {"label": "Distance along the river", "v": ex["krs|River-distance bands"]},
                      {"label": "+ drop rain held by Hemavathy dam", "v": ex["krs|River-distance bands, free-draining catchment only"]},
                      {"label": "+ that rain as a separate input", "v": ex["krs|... + Hemavathy-catchment rain as its own input"]}]
        d["river"] = [{"label": r["label"], "peak_err": r["v"]["peaks"]["peak_median_err_pct"],
                       "rise": round(r["v"]["peaks"]["rise_capture_median"] * 100), "nse": r["v"]["all"]["NSE"]}
                      for r in d["river"]]
    else:
        lag = R["lag"]["hemavathy"]
        d["lag_rain"] = {b: lag[b]["curve"] for b in ["near", "mid", "far"]}
        d["peaks"] = R["peaks"]["hemavathy|Rain only"]
        d["daily_nse"] = R["daily"]["hemavathy|Rain only"]["all"]["NSE"]
        other = {issue_label(r["issue_date"]): r[NONE]["p80_abs_pct_err"] for r in R["monsoon_inflow"]["krs"]["by_issue"]}
        d["vs_krs"] = [{"issue": f["issue"], "hem": f["none_p80"], "krs": other[f["issue"]]} for f in fc]
        d["vs_krs_daily"] = {"hem_rain": R["daily"]["hemavathy|Rain only"]["all"]["NSE"],
                             "krs_rain": R["daily"]["krs|Rain only"]["all"]["NSE"],
                             "krs_up": R["daily"]["krs|Rain + upstream releases"]["all"]["NSE"]}
    lr = d["lag_rain"]
    tot = [sum(v[k] for v in lr.values()) for k in range(len(lr["far"]))]
    d["rain_peak_day"] = max(range(len(tot)), key=lambda k: tot[k])
    d["rain_week_share"] = round(sum(tot[:8]) / max(sum(tot), 1e-9) * 100)
    return d


CSS = """
:root { color-scheme: light; --bg:#f6f7f9; --panel:#fff; --ink:#15202b; --ink2:#4f5b67; --ink3:#7d8893;
  --line:#e3e7ec; --grid:#eef1f4; --axis:#c4ccd4; --blue:#2a78d6; --orange:#eb6834; --green:#1baf7a;
  --blue-soft:rgba(42,120,214,.12); --good:#1d7a43; --radius:12px; }
* { box-sizing: border-box; }
html, body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1120px; margin:0 auto; padding:40px 16px 72px; }
header h1 { font-size:2rem; line-height:1.2; margin:0 0 6px; letter-spacing:-.01em; }
header p { margin:0; color:var(--ink2); max-width:70ch; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:28px 0 8px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:18px 20px; }
.tile .k { font-size:.8rem; color:var(--ink2); text-transform:uppercase; letter-spacing:.04em; }
.tile .v { font-size:2rem; font-weight:650; margin:4px 0 2px; font-variant-numeric:tabular-nums; }
.tile .s { font-size:.85rem; color:var(--ink2); }
.card { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:22px 24px 18px; margin-top:20px; }
.card h2 { font-size:1.2rem; margin:0 0 2px; line-height:1.3; }
.card .sub { color:var(--ink2); margin:0 0 12px; font-size:.9rem; }
.row2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
.row3 { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; }
.row3 > div h3 { font-size:.9rem; margin:0 0 6px; font-weight:600; min-height:2.7em; }
@media (max-width:900px) { .row3 { grid-template-columns:1fr; } .row3 > div h3 { min-height:0; } }
.flow { margin-top:8px; }
.stage { display:grid; grid-template-columns:170px 1fr; gap:16px; align-items:start; }
.stage .sh { font-weight:650; font-size:.95rem; padding-top:10px; }
.stage .sh span { display:inline-grid; place-items:center; width:24px; height:24px; border-radius:50%; background:var(--ink); color:#fff; font-size:.8rem; margin-right:8px; }
.stage .sh small { display:block; font-weight:400; color:var(--ink2); font-size:.8rem; margin:4px 0 0 32px; }
.nodes { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px; }
.nodes.two { grid-template-columns:1fr 1fr; }
.nodes.four { grid-template-columns:repeat(4,1fr); }
.nodes.three { grid-template-columns:repeat(3,1fr); }
.node { border:1px solid var(--line); border-left:4px solid var(--c, var(--ink3)); border-radius:10px; padding:10px 12px; background:#fff; font-size:.85rem; color:var(--ink2); }
.node b { display:block; color:var(--ink); font-size:.9rem; margin-bottom:3px; }
.node ul { margin:4px 0 0; padding-left:16px; } .node li { margin:2px 0; }
.node .out { margin-top:6px; color:var(--ink); }
.down { height:30px; margin-left:186px; position:relative; }
.down::before { content:""; position:absolute; left:50%; top:4px; bottom:8px; border-left:2px solid var(--axis); }
.down::after { content:""; position:absolute; left:calc(50% - 6px); bottom:2px; border:6px solid transparent; border-top:8px solid var(--axis); border-bottom:0; }
.n-data { --c:#7d8893; } .n-prep { --c:#2a78d6; } .n-model { --c:#eb6834; } .n-out { --c:#1d7a43; }
@media (max-width:760px) { .stage { grid-template-columns:1fr; gap:6px; } .down { margin-left:0; } .nodes.two { grid-template-columns:1fr; } .nodes.four { grid-template-columns:1fr 1fr; } .nodes.three { grid-template-columns:1fr; } }
h2.sec { font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; color:var(--ink2); margin:40px 0 -6px; font-weight:600; }
.row2 > div h3 { font-size:.95rem; margin:0 0 4px; font-weight:600; }
@media (max-width:760px) { .row2 { grid-template-columns:1fr; } .card { padding:18px 16px; } header h1 { font-size:1.6rem; } }
.chart { width:100%; position:relative; }
.chart svg { display:block; overflow:visible; }
.legend { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:.85rem; color:var(--ink2); margin:0 0 6px; }
.legend span { display:inline-flex; align-items:center; gap:6px; }
.legend i { width:18px; height:3px; border-radius:2px; display:inline-block; }
.legend i.dot { width:10px; height:10px; border-radius:50%; }
.legend i.dash { background:repeating-linear-gradient(90deg,var(--ink3) 0 5px,transparent 5px 9px) !important; }
.legend i.area { height:10px; border-radius:3px; }
.seg { display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; margin:0 0 10px; flex-wrap:wrap; }
.seg button { font:inherit; font-size:.85rem; border:0; background:var(--panel); color:var(--ink2); padding:6px 12px; cursor:pointer; }
select.yr { font:inherit; font-size:.85rem; border:1px solid var(--line); border-radius:8px; padding:6px 10px; margin:0 0 10px; background:var(--panel); color:var(--ink); }
.seg button + button { border-left:1px solid var(--line); }
.seg button[aria-pressed=true] { background:var(--ink); color:#fff; }
.axis text { fill:var(--ink3); font-size:11.5px; }
.axis path, .axis line { stroke:var(--axis); }
.grid line { stroke:var(--grid); }
.grid path { display:none; }
.lbl { font-size:12px; fill:var(--ink); font-weight:600; }
.lbl2 { font-size:11.5px; fill:var(--ink2); }
.tt { position:absolute; pointer-events:none; background:#fff; border:1px solid var(--line); border-radius:8px;
  box-shadow:0 6px 18px rgba(21,32,43,.12); padding:8px 10px; font-size:12.5px; line-height:1.45; min-width:140px; z-index:5; }
.tt b { font-weight:600; } .tt .r { display:flex; justify-content:space-between; gap:14px; }
.tt .sw { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
[hidden] { display:none !important; }
details { margin-top:20px; background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:14px 22px; }
details summary { cursor:pointer; font-weight:600; }
details li { margin:4px 0; color:var(--ink2); }
details code, details pre { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.82rem; }
details pre { background:var(--bg); padding:10px 12px; border-radius:8px; overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.85rem; }
td, th { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--ink2); font-weight:600; } td.n { text-align:right; font-variant-numeric:tabular-nums; }
.foot { color:var(--ink3); font-size:.8rem; margin-top:28px; }
"""

JS = r"""
const D = window.DAM;
const C = { blue:'#2a78d6', orange:'#eb6834', green:'#1baf7a', ink:'#15202b', ink3:'#7d8893', grid:'#eef1f4' };
const fmt0 = d3.format(',.0f'), pct = (v) => `${Math.round(v)}%`;
const tip = () => { let t = document.querySelector('.tt'); if (!t) { t = document.createElement('div'); t.className = 'tt'; t.hidden = true; document.body.appendChild(t); } return t; };
function showTip(ev, html) { const t = tip(); t.innerHTML = html; t.hidden = false;
  const w = t.offsetWidth, x = ev.pageX + 14 + w > window.scrollX + innerWidth ? ev.pageX - w - 14 : ev.pageX + 14;
  t.style.left = x + 'px'; t.style.top = (ev.pageY - 10) + 'px'; }
const hideTip = () => { tip().hidden = true; };
const row = (c, k, v) => `<div class="r"><span><span class="sw" style="background:${c}"></span>${k}</span><b>${v}</b></div>`;
function legend(el, items) { el.innerHTML = items.map((i) => `<span><i class="${i.cls || ''}" style="background:${i.c}"></i>${i.n}</span>`).join(''); }
function frame(el, h, m) {
  el.innerHTML = ''; const W = el.clientWidth, H = h;
  const svg = d3.select(el).append('svg').attr('width', W).attr('height', H);
  return { svg, W, H, iw: W - m.l - m.r, ih: H - m.t - m.b, g: svg.append('g').attr('transform', `translate(${m.l},${m.t})`) };
}
function yAxis(f, y, fmt, ticks = 5) {
  f.g.append('g').attr('class', 'grid').call(d3.axisLeft(y).ticks(ticks).tickSize(-f.iw).tickFormat(''));
  f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(ticks).tickFormat(fmt).tickSizeOuter(0)).call((a) => a.select('.domain').remove());
}
function yTitle(f, text) { f.g.append('text').attr('class', 'lbl2').attr('x', 0).attr('y', -12).text(text); }

/* ---- 1. forecast error by date ---------------------------------------------------- */
function chartForecast() {
  const el = document.getElementById('c-forecast');
  const F = D.forecast, m = { t: 28, r: 16, b: 34, l: 44 }, f = frame(el, 340, m);
  const x = d3.scalePoint().domain(F.map((d) => d.issue)).range([0, f.iw]).padding(0.2);
  const ymax = Math.max(80, d3.max(F, (d) => Math.max(d.none_p80, d.f30_p80, d.clim_p80)));
  const y = d3.scaleLinear().domain([0, Math.ceil(ymax / 10) * 10]).range([f.ih, 0]);
  f.g.append('rect').attr('x', 0).attr('width', f.iw).attr('y', y(D.GOOD)).attr('height', f.ih - y(D.GOOD)).attr('fill', 'rgba(29,122,67,.07)');
  yAxis(f, y, (v) => v + '%');
  f.g.append('text').attr('class', 'lbl2').attr('x', 4).attr('y', y(D.GOOD) + 14).attr('fill', '#1d7a43').text(`Within ±${D.GOOD}% in 4 of 5 years: reasonably informed`);
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).tickSizeOuter(0));
  yTitle(f, 'Error not exceeded in 4 of 5 years');
  const S = [{ k: 'clim_p80', n: 'Assume an average year', c: C.ink3, dash: true },
             { k: 'none_p80', n: 'Model, no rain forecast', c: C.blue },
             { k: 'f30_p80', n: 'Model + 30-day rain forecast', c: C.orange }];
  S.forEach((s) => {
    const line = d3.line().x((d) => x(d.issue)).y((d) => y(d[s.k]));
    f.g.append('path').datum(F).attr('fill', 'none').attr('stroke', s.c).attr('stroke-width', s.dash ? 1.5 : 2.5)
      .attr('stroke-dasharray', s.dash ? '5 4' : null).attr('d', line);
    if (!s.dash) f.g.selectAll(null).data(F).join('circle').attr('cx', (d) => x(d.issue)).attr('cy', (d) => y(d[s.k]))
      .attr('r', 4).attr('fill', s.c).attr('stroke', '#fff').attr('stroke-width', 2);
    if (s.dash) f.g.append('text').attr('class', 'lbl2').attr('x', f.iw).attr('text-anchor', 'end').attr('y', y(F[F.length - 1][s.k]) - 6).text('Average-year guess');
  });
  legend(document.getElementById('l-forecast'), S.map((s) => ({ n: s.n, c: s.c, cls: s.dash ? 'dash' : '' })));
  crosshair(f, x, F, (d) => `<b>Forecast made ${d.issue}</b><div class="lbl2" style="margin:2px 0 4px">${d.seen}% of the season's inflow already measured</div>
    ${row(C.ink3, 'Average year', `typical ${pct(d.clim_med)} · 4 in 5 ≤ ${pct(d.clim_p80)}`)}
    ${row(C.blue, 'No rain forecast', `typical ${pct(d.none_med)} · 4 in 5 ≤ ${pct(d.none_p80)}`)}
    ${row(C.orange, '+ 30-day rain', `typical ${pct(d.f30_med)} · 4 in 5 ≤ ${pct(d.f30_p80)}`)}`, (d) => d.issue);
}
function crosshair(f, x, data, html, key) {
  const v = f.g.append('line').attr('y1', 0).attr('y2', f.ih).attr('stroke', C.ink3).attr('stroke-width', 1).attr('opacity', 0);
  f.g.append('rect').attr('width', f.iw).attr('height', f.ih).attr('fill', 'transparent')
    .on('mousemove', (ev) => {
      const [mx] = d3.pointer(ev);
      const d = data.reduce((a, b) => Math.abs(x(key(b)) - mx) < Math.abs(x(key(a)) - mx) ? b : a);
      v.attr('x1', x(key(d))).attr('x2', x(key(d))).attr('opacity', .5); showTip(ev, html(d));
    }).on('mouseleave', () => { v.attr('opacity', 0); hideTip(); });
}

/* ---- 2. EDA: year-by-year stack + relations -------------------------------------- */
function chartStack() {
  const E = D.eda, el = document.getElementById('c-stack'), cap = D.cap;
  const P = [{ k: 'rain', n: 'Catchment rainfall', u: 'mm', f: (v) => fmt0(v) + ' mm' },
             { k: 'inflow', n: 'Inflow', u: 'TMC', f: (v) => v.toFixed(1) + ' TMC' },
             { k: 'outflow', n: 'Outflow (releases)', u: 'TMC', f: (v) => v.toFixed(1) + ' TMC' },
             { k: 'pct_sep30', n: 'Storage on 30 Sep', u: '% full', f: (v) => v == null ? '—' : Math.round(v) + '% full' }];
  const ph = 104, gap = 34, m = { t: 22, r: 16, b: 28, l: 52 };
  const f = frame(el, m.t + m.b + P.length * ph + (P.length - 1) * gap, m);
  const x = d3.scaleBand().domain(E.map((d) => d.year)).range([0, f.iw]).padding(0.2);
  P.forEach((p, i) => {
    const g = f.g.append('g').attr('transform', `translate(0,${i * (ph + gap)})`);
    const mx = p.k === 'pct_sep30' ? 100 : d3.max(E, (d) => d[p.k]) * 1.05;
    const y = d3.scaleLinear().domain([0, mx]).nice(3).range([ph, 0]);
    g.append('g').attr('class', 'grid').call(d3.axisLeft(y).ticks(3).tickSize(-f.iw).tickFormat(''));
    g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(3).tickFormat((v) => p.k === 'pct_sep30' ? v + '%' : d3.format('~s')(v)).tickSizeOuter(0)).call((a) => a.select('.domain').remove());
    g.append('text').attr('class', 'lbl').attr('x', 0).attr('y', -8).text(`${p.n}`).append('tspan').attr('class', 'lbl2').attr('font-weight', 400).text(`  ${p.u}, Jun–Sep${p.k === 'pct_sep30' ? ' (end)' : ' total'}`);
    g.selectAll(null).data(E.filter((d) => d[p.k] != null)).join('path').attr('fill', (d) => d.dry ? C.orange : C.blue)
      .attr('d', (d) => { const x0 = x(d.year), w = x.bandwidth(), y0 = y(d[p.k]), r = Math.min(2.5, w / 2, ph - y0);
        return `M${x0},${ph}V${y0 + r}Q${x0},${y0} ${x0 + r},${y0}H${x0 + w - r}Q${x0 + w},${y0} ${x0 + w},${y0 + r}V${ph}Z`; });
    if (p.k === 'pct_sep30') g.selectAll(null).data(E.filter((d) => d.pct_jun1 != null)).join('line')
      .attr('x1', (d) => x(d.year) - 1).attr('x2', (d) => x(d.year) + x.bandwidth() + 1).attr('y1', (d) => y(d.pct_jun1)).attr('y2', (d) => y(d.pct_jun1))
      .attr('stroke', C.ink).attr('stroke-width', 2.5);
    const av = d3.mean(E, (d) => d[p.k]);
    g.append('line').attr('x1', 0).attr('x2', f.iw).attr('y1', y(av)).attr('y2', y(av)).attr('stroke', C.ink).attr('stroke-dasharray', '4 4').attr('opacity', .7);
    if (i === P.length - 1) g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ph})`)
      .call(d3.axisBottom(x).tickValues(E.map((d) => d.year).filter((v) => v % 5 === 0)).tickSizeOuter(0));
  });
  const hl = f.g.append('rect').attr('y', -20).attr('height', P.length * (ph + gap) - gap + 20).attr('fill', 'rgba(21,32,43,.07)').attr('opacity', 0).attr('rx', 3);
  f.g.append('g').selectAll(null).data(E).join('rect').attr('x', (d) => x(d.year) - x.step() * x.padding() / 2).attr('width', x.step())
    .attr('y', -20).attr('height', P.length * (ph + gap)).attr('fill', 'transparent')
    .on('mousemove', (ev, d) => { hl.attr('x', x(d.year) - x.step() * x.padding() / 2).attr('width', x.step()).attr('opacity', 1);
      showTip(ev, `<b>${d.year}${d.dry ? ' · dry year' : ''}</b>` + P.map((p) => row(d.dry ? C.orange : C.blue, p.n, p.f(d[p.k]))).join('') +
        row(C.ink, 'Storage on 1 Jun', d.pct_jun1 == null ? '—' : Math.round(d.pct_jun1) + '% full')); })
    .on('mouseleave', () => { hl.attr('opacity', 0); hideTip(); });
}
function relScatter(id, xk, yk, xl, yl, opts = {}) {
  const E = D.eda.filter((d) => d[xk] != null && d[yk] != null), el = document.getElementById(id);
  const m = { t: 22, r: 12, b: 38, l: 46 }, f = frame(el, 250, m);
  const x = d3.scaleLinear().domain([0, d3.max(E, (d) => d[xk]) * 1.05]).nice().range([0, f.iw]);
  const y = d3.scaleLinear().domain([0, opts.ymax || d3.max(E, (d) => d[yk]) * 1.05]).nice().range([f.ih, 0]);
  yAxis(f, y, opts.yfmt || d3.format('~s'), 4); yTitle(f, yl);
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(4).tickFormat(d3.format('~s')).tickSizeOuter(0));
  f.g.append('text').attr('class', 'lbl2').attr('x', f.iw).attr('y', f.ih + 32).attr('text-anchor', 'end').text(xl);
  if (opts.line) f.g.append('line').attr('x1', x(opts.line[0][0])).attr('y1', y(opts.line[0][1])).attr('x2', x(opts.line[1][0])).attr('y2', y(opts.line[1][1]))
    .attr('stroke', C.ink).attr('stroke-dasharray', '5 4').attr('opacity', .7);
  if (opts.vline != null) { f.g.append('line').attr('x1', x(opts.vline)).attr('x2', x(opts.vline)).attr('y1', 0).attr('y2', f.ih).attr('stroke', C.ink).attr('stroke-dasharray', '5 4').attr('opacity', .7);
    f.g.append('text').attr('class', 'lbl2').attr('x', x(opts.vline) + 4).attr('y', 10).text(`${opts.vline} TMC`); }
  if (opts.hline != null) f.g.append('line').attr('x1', 0).attr('x2', f.iw).attr('y1', y(opts.hline)).attr('y2', y(opts.hline)).attr('stroke', '#1d7a43').attr('stroke-dasharray', '2 3');
  f.g.selectAll(null).data(E).join('circle').attr('cx', (d) => x(d[xk])).attr('cy', (d) => y(d[yk])).attr('r', 5)
    .attr('fill', (d) => d.dry ? C.orange : C.blue).attr('fill-opacity', .85).attr('stroke', '#fff').attr('stroke-width', 1.5)
    .on('mousemove', (ev, d) => showTip(ev, `<b>${d.year}</b>${row('#fff', 'Rainfall', fmt0(d.rain) + ' mm')}${row('#fff', 'Inflow', d.inflow + ' TMC')}${row('#fff', 'Outflow', d.outflow + ' TMC')}${row('#fff', '30 Sep storage', d.pct_sep30 == null ? '—' : Math.round(d.pct_sep30) + '% full')}`))
    .on('mouseleave', hideTip);
}
function chartRelations() {
  const S = D.eda_stats, rx = d3.extent(D.eda, (d) => d.rain);
  relScatter('c-r1', 'rain', 'inflow', 'Jun–Sep rainfall, mm', 'Inflow, TMC', { line: rx.map((r) => [r, S.fit[0] + S.fit[1] * r]) });
  relScatter('c-r2', 'inflow', 'pct_sep30', 'Jun–Sep inflow, TMC', 'Storage on 30 Sep, % full', { ymax: 100, yfmt: (v) => v + '%', vline: S.fill_threshold, hline: 90 });
  const mi = d3.max(D.eda, (d) => d.inflow);
  relScatter('c-r3', 'inflow', 'outflow', 'Jun–Sep inflow, TMC', 'Outflow, TMC', { line: [[0, 0], [mi, mi * S.release_share / 100]] });
}

/* ---- 3. forecast vs actual, per year ----------------------------------------------- */
let issueSel = '1 Aug';
function chartScatter() {
  const el = document.getElementById('c-scatter'), P = D.per_year[issueSel];
  const m = { t: 24, r: 16, b: 40, l: 48 }, f = frame(el, Math.min(420, el.clientWidth * 0.9), m);
  const hi = Math.ceil(d3.max(P, (d) => Math.max(d.actual, d.none, d.f30)) / 20) * 20;
  const x = d3.scaleLinear().domain([0, hi]).range([0, f.iw]), y = d3.scaleLinear().domain([0, hi]).range([f.ih, 0]);
  f.g.append('path').attr('d', d3.area().x((v) => x(v)).y0((v) => y(Math.min(hi, v * 1.2))).y1((v) => y(v * 0.8))([0, hi / 1.2, hi])).attr('fill', 'rgba(29,122,67,.08)');
  yAxis(f, y, (v) => v);
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  f.g.append('line').attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(hi)).attr('y2', y(hi)).attr('stroke', C.ink3).attr('stroke-dasharray', '4 4');
  yTitle(f, 'Forecast, TMC');
  f.g.append('text').attr('class', 'lbl2').attr('x', f.iw).attr('y', f.ih + 34).attr('text-anchor', 'end').text('Actual Jun–Sep inflow, TMC');
  f.g.append('text').attr('class', 'lbl2').attr('x', x(hi * 0.06)).attr('y', y(hi * 0.06 * 1.2) - 18).attr('fill', '#1d7a43').text('±20%');
  const pts = P.flatMap((d) => [{ ...d, v: d.none, s: 'none' }, { ...d, v: d.f30, s: 'f30' }]);
  f.g.selectAll(null).data(pts).join('circle').attr('cx', (d) => x(d.actual)).attr('cy', (d) => y(d.v)).attr('r', 5)
    .attr('fill', (d) => d.s === 'none' ? C.blue : C.orange).attr('fill-opacity', .85).attr('stroke', '#fff').attr('stroke-width', 1.5)
    .on('mousemove', (ev, d) => showTip(ev, `<b>${d.year}</b>${row('#fff', 'Actual', d.actual + ' TMC')}
      ${row(C.blue, 'No rain forecast', `${d.none} (${d3.format('+.0f')((d.none - d.actual) / d.actual * 100)}%)`)}
      ${row(C.orange, '+ 30-day rain', `${d.f30} (${d3.format('+.0f')((d.f30 - d.actual) / d.actual * 100)}%)`)}`))
    .on('mouseleave', hideTip);
  const inside = (k) => P.filter((d) => Math.abs(d[k] - d.actual) / d.actual <= 0.2).length;
  document.getElementById('s-scatter').textContent =
    `Forecast made ${issueSel}: ${inside('none')} of ${P.length} years within ±20% with no rain forecast, ${inside('f30')} of ${P.length} with a 30-day rain forecast.`;
}
function segIssue() {
  const el = document.getElementById('seg-issue');
  ['15 Jul', '1 Aug', '15 Aug', '1 Sep'].forEach((k) => {
    const b = document.createElement('button'); b.textContent = k; b.setAttribute('aria-pressed', k === issueSel);
    b.onclick = () => { issueSel = k; el.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b)); chartScatter(); };
    el.appendChild(b);
  });
}

/* ---- 4. lag curves ---------------------------------------------------------------- */
function lagChart(id, lid, series, yt, yfmt, tfmt) {
  const el = document.getElementById(id), m = { t: 28, r: 16, b: 34, l: 48 }, f = frame(el, 260, m);
  const n = series[0].v.length, x = d3.scaleLinear().domain([0, n - 1]).range([0, f.iw]);
  const y = d3.scaleLinear().domain([0, d3.max(series, (s) => d3.max(s.v)) * 1.1]).nice().range([f.ih, 0]);
  yAxis(f, y, yfmt, 4); yTitle(f, yt);
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(n).tickFormat((d) => d).tickSizeOuter(0));
  f.g.append('text').attr('class', 'lbl2').attr('x', f.iw).attr('y', f.ih + 30).attr('text-anchor', 'end').text('days later');
  series.forEach((s) => {
    f.g.append('path').datum(s.v).attr('fill', 'none').attr('stroke', s.c).attr('stroke-width', 2.5).attr('stroke-linejoin', 'round')
      .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d)).curve(d3.curveMonotoneX));
  });
  legend(document.getElementById(lid), series.map((s) => ({ n: s.n, c: s.c })));
  crosshair(f, x, d3.range(n), (i) => `<b>${i} day${i === 1 ? '' : 's'} after</b>` + series.map((s) => row(s.c, s.n, tfmt(s.v[i]))).join(''), (i) => i);
}

/* ---- 5. hydrograph ---------------------------------------------------------------- */
let yearSel = null;
function chartHydro() {
  const H = D.hydro, keys = Object.keys(H).filter((k) => k !== 'dates' && k !== 'observed');
  const idx = H.dates.map((d, i) => [d, i]).filter(([d]) => +d.slice(0, 4) === yearSel).map(([, i]) => i);
  const rows = idx.map((i) => ({ t: new Date(H.dates[i]), o: H.observed[i], a: H[keys[0]][i], b: H[keys[1]][i] }));
  const el = document.getElementById('c-hydro'), m = { t: 28, r: 16, b: 30, l: 56 }, f = frame(el, 320, m);
  const x = d3.scaleTime().domain(d3.extent(rows, (d) => d.t)).range([0, f.iw]);
  const y = d3.scaleLinear().domain([0, d3.max(rows, (d) => Math.max(d.o, d.a, d.b)) * 1.05]).nice().range([f.ih, 0]);
  yAxis(f, y, d3.format('~s')); yTitle(f, 'Daily inflow, cusecs');
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(d3.timeMonth.every(1)).tickFormat(d3.timeFormat('%b')).tickSizeOuter(0));
  f.g.append('path').datum(rows).attr('fill', 'rgba(21,32,43,.08)').attr('d', d3.area().x((d) => x(d.t)).y0(f.ih).y1((d) => y(d.o)));
  const S = [{ k: 'o', n: 'Observed', c: C.ink, w: 1.5 }, { k: 'a', n: D.hydroNames[0], c: C.blue, w: 2 }, { k: 'b', n: D.hydroNames[1], c: C.green, w: 2 }];
  S.forEach((s) => f.g.append('path').datum(rows).attr('fill', 'none').attr('stroke', s.c).attr('stroke-width', s.w)
    .attr('d', d3.line().x((d) => x(d.t)).y((d) => y(d[s.k]))));
  legend(document.getElementById('l-hydro'), S.map((s) => ({ n: s.n, c: s.c })));
  const bis = d3.bisector((d) => d.t).center;
  const v = f.g.append('line').attr('y1', 0).attr('y2', f.ih).attr('stroke', C.ink3).attr('opacity', 0);
  f.g.append('rect').attr('width', f.iw).attr('height', f.ih).attr('fill', 'transparent')
    .on('mousemove', (ev) => { const d = rows[bis(rows, x.invert(d3.pointer(ev)[0]))];
      v.attr('x1', x(d.t)).attr('x2', x(d.t)).attr('opacity', .5);
      showTip(ev, `<b>${d3.timeFormat('%d %b %Y')(d.t)}</b>` + S.map((s) => row(s.c, s.n, fmt0(d[s.k]))).join('')); })
    .on('mouseleave', () => { v.attr('opacity', 0); hideTip(); });
}
function segYear() {
  const el = document.getElementById('seg-year'), yrs = [...new Set(D.hydro.dates.map((d) => +d.slice(0, 4)))];
  yearSel = yrs.reduce((a, b) => (d3.max(D.hydro.observed.filter((v, i) => +D.hydro.dates[i].slice(0, 4) === b)) >
    d3.max(D.hydro.observed.filter((v, i) => +D.hydro.dates[i].slice(0, 4) === a)) ? b : a));
  yrs.forEach((k) => { const b = document.createElement('button'); b.textContent = k; b.setAttribute('aria-pressed', k === yearSel);
    b.onclick = () => { yearSel = k; el.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b)); chartHydro(); };
    el.appendChild(b); });
}

/* ---- 6. lead time ----------------------------------------------------------------- */
function leadChart(id, key, yt, fmt, domain) {
  const L = D.lead, el = document.getElementById(id), m = { t: 28, r: 16, b: 34, l: 48 }, f = frame(el, 250, m);
  const leads = [...new Set(L.map((d) => d.lead_days))];
  const x = d3.scalePoint().domain(leads).range([0, f.iw]).padding(0.3);
  const y = d3.scaleLinear().domain(domain).range([f.ih, 0]);
  yAxis(f, y, fmt, 4); yTitle(f, yt);
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).tickFormat((d) => d + (d === 1 ? ' day' : ' days')).tickSizeOuter(0));
  if (domain[0] < 0) f.g.append('line').attr('x1', 0).attr('x2', f.iw).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', C.ink3);
  const S = [{ v: 'Known on the day', n: 'Today’s data only', c: C.blue }, { v: '+ rain forecast', n: '+ rain forecast', c: C.orange }];
  S.forEach((s) => { const d = L.filter((r) => r.variant === s.v);
    f.g.append('path').datum(d).attr('fill', 'none').attr('stroke', s.c).attr('stroke-width', 2.5).attr('d', d3.line().x((r) => x(r.lead_days)).y((r) => y(r[key])));
    f.g.selectAll(null).data(d).join('circle').attr('cx', (r) => x(r.lead_days)).attr('cy', (r) => y(r[key])).attr('r', 4).attr('fill', s.c).attr('stroke', '#fff').attr('stroke-width', 2); });
  crosshair(f, x, leads, (l) => `<b>${l} day${l === 1 ? '' : 's'} ahead</b>` + S.map((s) => { const r = L.find((q) => q.lead_days === l && q.variant === s.v);
    return row(s.c, s.n, `${fmt(r[key])}`) + `<div class="lbl2" style="margin:-2px 0 4px 14px">false alarms ${pct(r.false_alarm_pct)}</div>`; }).join(''), (l) => l);
  return S;
}

/* ---- 7. dam-specific -------------------------------------------------------------- */
function hbar(id, items, fmt, color, max) {
  const el = document.getElementById(id), m = { t: 4, r: 56, b: 6, l: Math.min(250, el.clientWidth * 0.45) };
  const f = frame(el, items.length * 34 + 10, m);
  const y = d3.scaleBand().domain(items.map((d) => d.label)).range([0, f.ih]).padding(0.3);
  const x = d3.scaleLinear().domain([0, max || d3.max(items, (d) => Math.abs(d.v))]).range([0, f.iw]);
  f.g.selectAll(null).data(items).join('rect').attr('y', (d) => y(d.label)).attr('height', y.bandwidth()).attr('x', 0)
    .attr('width', (d) => x(Math.abs(d.v))).attr('rx', 3).attr('fill', (d) => d.c || color)
    .on('mousemove', (ev, d) => showTip(ev, `<b>${d.label}</b>${d.tip || ''}`)).on('mouseleave', hideTip);
  f.g.selectAll(null).data(items).join('text').attr('class', 'lbl2').attr('x', -10).attr('text-anchor', 'end')
    .attr('y', (d) => y(d.label) + y.bandwidth() / 2 + 4).text((d) => d.label);
  f.g.selectAll(null).data(items).join('text').attr('class', 'lbl').attr('x', (d) => x(Math.abs(d.v)) + 8)
    .attr('y', (d) => y(d.label) + y.bandwidth() / 2 + 4).text((d) => fmt(d.v));
}
function chartVsKrs() {
  const V = D.vs_krs, el = document.getElementById('c-vskrs'), m = { t: 28, r: 16, b: 34, l: 44 }, f = frame(el, 280, m);
  const x = d3.scalePoint().domain(V.map((d) => d.issue)).range([0, f.iw]).padding(0.2);
  const y = d3.scaleLinear().domain([0, Math.ceil(d3.max(V, (d) => Math.max(d.hem, d.krs)) / 10) * 10]).range([f.ih, 0]);
  yAxis(f, y, (v) => v + '%'); yTitle(f, 'Error not exceeded in 4 of 5 years (no rain forecast)');
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).tickSizeOuter(0));
  const S = [{ k: 'hem', n: 'Hemavathy', c: C.blue }, { k: 'krs', n: 'KRS', c: C.ink3 }];
  S.forEach((s) => { f.g.append('path').datum(V).attr('fill', 'none').attr('stroke', s.c).attr('stroke-width', 2.5).attr('d', d3.line().x((d) => x(d.issue)).y((d) => y(d[s.k])));
    f.g.selectAll(null).data(V).join('circle').attr('cx', (d) => x(d.issue)).attr('cy', (d) => y(d[s.k])).attr('r', 4).attr('fill', s.c).attr('stroke', '#fff').attr('stroke-width', 2);
  });
  legend(document.getElementById('l-vskrs'), S.map((s) => ({ n: s.n, c: s.c })));
  crosshair(f, x, V, (d) => `<b>Forecast made ${d.issue}</b>` + S.map((s) => row(s.c, s.n, pct(d[s.k]))).join(''), (d) => d.issue);
}

function cellMap(id, lid, key, ramp, fmt, title) {
  const el = document.getElementById(id), G = D.geo, cells = D.cells;
  const lons = G.outline.map((p) => p[0]), lats = G.outline.map((p) => p[1]);
  const x0 = d3.min(lons) - 0.13, x1 = d3.max(lons) + 0.13, y0 = d3.min(lats) - 0.13, y1 = d3.max(lats) + 0.13;
  const k = Math.cos((y0 + y1) / 2 * Math.PI / 180), W = el.clientWidth;
  const sc = W / ((x1 - x0) * k), H = Math.min((y1 - y0) * sc, 520), s = Math.min(sc, H / (y1 - y0));
  const px = (lon) => (lon - x0) * k * s + (W - (x1 - x0) * k * s) / 2, py = (lat) => (y1 - lat) * s;
  el.innerHTML = '';
  const svg = d3.select(el).append('svg').attr('width', W).attr('height', (y1 - y0) * s);
  const pat = svg.append('defs').append('pattern').attr('id', id + 'h').attr('width', 6).attr('height', 6).attr('patternUnits', 'userSpaceOnUse').attr('patternTransform', 'rotate(45)');
  pat.append('rect').attr('width', 6).attr('height', 6).attr('fill', '#f1f3f5'); pat.append('line').attr('y2', 6).attr('stroke', '#c4ccd4').attr('stroke-width', 2);
  const clip = svg.append('clipPath').attr('id', id + 'c');
  clip.append('path').attr('d', 'M' + G.outline.map((p) => `${px(p[0])},${py(p[1])}`).join('L') + 'Z');
  const vals = cells.map((c) => c[key]).filter((v) => v != null && v > 0);
  const color = d3.scaleSequential(ramp).domain([0, d3.max(vals)]);
  const cg = svg.append('g').attr('clip-path', `url(#${id}c)`);
  const half = 0.125;
  (D.held || []).forEach((c) => cg.append('rect').attr('x', px(c.lon - half)).attr('y', py(c.lat + half)).attr('width', 0.25 * k * s).attr('height', 0.25 * s).attr('fill', `url(#${id}h)`));
  cg.selectAll(null).data(cells).join('rect').attr('x', (c) => px(c.lon - half)).attr('y', (c) => py(c.lat + half))
    .attr('width', 0.25 * k * s - 1).attr('height', 0.25 * s - 1)
    .attr('fill', (c) => c[key] == null || c.gain_cusec_days_per_mm <= 1 ? '#eef1f4' : color(c[key]))
    .attr('fill-opacity', (c) => c.area_km2 < 150 ? 0.45 : 1);
  svg.append('path').attr('d', 'M' + G.outline.map((p) => `${px(p[0])},${py(p[1])}`).join('L') + 'Z').attr('fill', 'none').attr('stroke', C.ink).attr('stroke-width', 1.5);
  G.rivers.forEach((r) => svg.append('path').attr('d', 'M' + r.c.map((p) => `${px(p[0])},${py(p[1])}`).join('L'))
    .attr('fill', 'none').attr('stroke', '#3d5a80').attr('stroke-opacity', .5).attr('stroke-width', r.o >= 4 ? 2.6 : r.o === 3 ? 1.8 : 1));
  svg.append('circle').attr('cx', px(G.dam[0])).attr('cy', py(G.dam[1])).attr('r', 6).attr('fill', C.ink).attr('stroke', '#fff').attr('stroke-width', 2);
  svg.append('text').attr('class', 'lbl').attr('x', px(G.dam[0]) + 9).attr('y', py(G.dam[1]) + 4).text(D.short + ' dam');
  svg.append('g').selectAll(null).data(cells).join('rect').attr('x', (c) => px(c.lon - half)).attr('y', (c) => py(c.lat + half))
    .attr('width', 0.25 * k * s).attr('height', 0.25 * s).attr('fill', 'transparent')
    .on('mousemove', (ev, c) => showTip(ev, `<b>Cell ${c.lat.toFixed(2)}°N ${c.lon.toFixed(2)}°E</b>
      ${row('#fff', 'Multiplier', c.gain_cusec_days_per_mm > 1 ? fmt0(c.gain_cusec_days_per_mm) + ' cusec-days per mm' : '≈ 0')}
      ${row('#fff', 'Delay', c.delay_days != null && c.gain_cusec_days_per_mm > 1 ? c.delay_days + ' days (peak day ' + c.peak_day + ')' : '—')}
      ${row('#fff', 'Area in catchment', c.area_km2 + ' km²')}
      ${c.area_km2 < 150 ? '<div class="lbl2">Small edge cell: low confidence</div>' : ''}`))
    .on('mouseleave', hideTip);
  const lg = document.getElementById(lid), mx = d3.max(vals);
  lg.innerHTML = `<span style="gap:8px">low<i style="width:90px;height:10px;background:linear-gradient(90deg,${color(mx * 0.05)},${color(mx)})"></i>high · ${fmt(0)}–${fmt(mx)}</span>
    <span><i class="area" style="background:#eef1f4;border:1px solid #dde3e8"></i>≈ no effect</span>` +
    (D.held && D.held.length ? `<span><i class="area" style="background:repeating-linear-gradient(45deg,#c4ccd4 0 2px,#f1f3f5 2px 5px)"></i>held by Hemavathy dam, arrives via its releases</span>` : '') +
    `<span><i class="area" style="background:${color(mx)};opacity:.45"></i>faded = small edge cell</span>`;
}

function chartLadder() {
  const L = D.ladder, el = document.getElementById('c-ladder');
  const W = el.clientWidth, narrow = W < 700, rowH = 30, gH = 26;
  let yy = 0; const pos = [];
  L.forEach((r, i) => { if (i === 0 || r.group !== L[i - 1].group) yy += gH; pos.push(yy); yy += rowH; });
  const lw = narrow ? 0 : Math.min(300, W * 0.34), pw = narrow ? (W - 24) / 2 : (W - lw - 40) / 2;
  const top = 26, H = top + yy + 10 + (narrow ? 0 : 0);
  el.innerHTML = '';
  const svg = d3.select(el).append('svg').attr('width', W).attr('height', H + (narrow ? L.length * 16 : 0));
  const xN = d3.scaleLinear().domain([0, 1]).range([0, pw - 50]);
  const xP = d3.scaleLinear().domain([0, 100]).range([0, pw - 50]);
  const x0 = lw + (narrow ? 0 : 12), x1 = x0 + pw + 16;
  svg.append('text').attr('class', 'lbl').attr('x', x0).attr('y', 14).text('Daily accuracy (NSE)');
  svg.append('text').attr('class', 'lbl2').attr('x', x0 + 130).attr('y', 14).text('higher is better');
  svg.append('text').attr('class', 'lbl').attr('x', x1).attr('y', 14).text('Flood peaks: size error');
  svg.append('text').attr('class', 'lbl2').attr('x', x1 + 150).attr('y', 14).text('shorter is better');
  const off = (i) => narrow ? i * 16 : 0;
  L.forEach((r, i) => {
    const y = top + pos[i] + off(i);
    if (i === 0 || r.group !== L[i - 1].group)
      svg.append('text').attr('x', 0).attr('y', y - 8).attr('class', 'lbl2').attr('font-weight', 600).attr('letter-spacing', '.04em').text(r.group.toUpperCase());
    const col = r.final ? C.blue : r.group === 'Simple baselines' ? '#d3d9df' : '#9aa6b2';
    const by = narrow ? y + 14 : y + 3;
    svg.append('text').attr('x', narrow ? 0 : lw).attr('y', narrow ? y + 6 : y + 15).attr('text-anchor', narrow ? 'start' : 'end')
      .attr('class', r.final ? 'lbl' : 'lbl2').attr('fill', C.ink).text(r.label);
    const n = Math.max(0, r.nse);
    svg.append('rect').attr('x', x0).attr('y', by).attr('height', 16).attr('width', xN(n)).attr('rx', 3).attr('fill', col);
    svg.append('text').attr('class', r.final ? 'lbl' : 'lbl2').attr('x', x0 + xN(n) + 6).attr('y', by + 12).text(r.nse.toFixed(2));
    const pe = Math.min(100, Math.abs(r.peak));
    svg.append('rect').attr('x', x1).attr('y', by).attr('height', 16).attr('width', xP(pe)).attr('rx', 3).attr('fill', col);
    svg.append('text').attr('class', r.final ? 'lbl' : 'lbl2').attr('x', x1 + xP(pe) + 6).attr('y', by + 12).text(`${Math.round(Math.abs(r.peak))}% ${r.peak < 0 ? 'low' : 'high'}`);
    svg.append('rect').attr('x', 0).attr('y', y).attr('width', W).attr('height', narrow ? rowH + 16 : rowH).attr('fill', 'transparent')
      .on('mousemove', (ev) => showTip(ev, `<b>${r.label}</b><div class="lbl2" style="margin-bottom:4px">${r.group}</div>${row(col, 'Daily NSE (2020–2023)', r.nse.toFixed(2))}${row(col, 'Flood-peak size error (median)', Math.round(r.peak) + '%')}${row(col, 'Peaks within ±25%', r.within25 + '%')}${row(col, 'Share of sharp rises caught', r.rise + '%')}`))
      .on('mouseleave', hideTip);
  });
}
function chartMatrix() {
  const M = D.matrix, el = document.getElementById('c-matrix'), W = el.clientWidth;
  const lw = Math.min(190, W * 0.36), cw = (W - lw) / M.issues.length, ch = 38, top = 26;
  el.innerHTML = '';
  const svg = d3.select(el).append('svg').attr('width', W).attr('height', top + M.rows.length * ch + 4);
  const color = d3.scaleSequential(d3.interpolate('#fff4ec', '#c4441a')).domain([0, 70]).clamp(true);
  M.issues.forEach((iss, j) => svg.append('text').attr('class', 'lbl2').attr('x', lw + j * cw + cw / 2).attr('y', 16).attr('text-anchor', 'middle').text(iss));
  M.rows.forEach((r, i) => {
    const y = top + i * ch;
    svg.append('text').attr('class', i === 1 || i === 4 ? 'lbl' : 'lbl2').attr('fill', C.ink).attr('x', lw - 10).attr('y', y + ch / 2 + 4).attr('text-anchor', 'end').text(r.label);
    r.p80.forEach((v, j) => {
      const ok = v <= D.GOOD;
      svg.append('rect').attr('x', lw + j * cw + 1).attr('y', y + 1).attr('width', cw - 2).attr('height', ch - 2).attr('rx', 4).attr('fill', color(v))
        .on('mousemove', (ev) => showTip(ev, `<b>${r.label}</b><div class="lbl2" style="margin-bottom:4px">forecast made ${M.issues[j]}</div>${row('#fff', 'Typical error', pct(r.med[j]))}${row('#fff', '4 years in 5 within', pct(v))}`))
        .on('mouseleave', hideTip);
      svg.append('text').attr('x', lw + j * cw + cw / 2).attr('y', y + ch / 2 + 4).attr('text-anchor', 'middle').attr('pointer-events', 'none')
        .attr('font-size', cw < 62 ? 11 : 12).attr('font-weight', ok ? 700 : 400).attr('fill', v > 45 ? '#fff' : C.ink)
        .text((ok && cw >= 62 ? '✓ ' : '') + Math.round(v) + '%');
    });
  });
}

function chartSoil() {
  const S = D.soil, el = document.getElementById('c-soil');
  const yrs = Object.keys(S.without.nse_by_year), m = { t: 28, r: 16, b: 30, l: 44 }, f = frame(el, 260, m);
  const x0 = d3.scaleBand().domain(yrs.concat(['All years'])).range([0, f.iw]).padding(0.25);
  const x1 = d3.scaleBand().domain(['without', 'with']).range([0, x0.bandwidth()]).padding(0.12);
  const val = (k, y) => y === 'All years' ? S[k].NSE : S[k].nse_by_year[y];
  const y = d3.scaleLinear().domain([0, 1]).range([f.ih, 0]);
  yAxis(f, y, (v) => v.toFixed(1), 5); yTitle(f, 'Daily NSE (1 = perfect)');
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x0).tickSizeOuter(0));
  const col = { without: '#9aa6b2', with: C.blue };
  x0.domain().forEach((yr) => ['without', 'with'].forEach((k) => {
    const v = val(k, yr), bx = x0(yr) + x1(k), h = f.ih - y(Math.max(0, v));
    f.g.append('rect').attr('x', bx).attr('width', x1.bandwidth()).attr('y', y(Math.max(0, v))).attr('height', h).attr('rx', 3).attr('fill', col[k])
      .on('mousemove', (ev) => showTip(ev, `<b>${yr}</b>${row(col.without, 'Without soil data', val('without', yr).toFixed(2))}${row(col.with, 'With soil data', val('with', yr).toFixed(2))}`))
      .on('mouseleave', hideTip);
    f.g.append('text').attr('class', k === 'with' ? 'lbl' : 'lbl2').attr('x', bx + x1.bandwidth() / 2).attr('y', y(Math.max(0, v)) - 5).attr('text-anchor', 'middle').text(v.toFixed(2));
  }));
  legend(document.getElementById('l-soil'), [{ n: 'Without soil data', c: col.without, cls: 'area' }, { n: 'With soil data', c: col.with, cls: 'area' }]);
}

/* ---- heavy-rain updates and next 15 days ------------------------------------------ */
let evYear = null, n15Year = null;
function chartEvents() {
  const T = D.events.traj[evYear], el = document.getElementById('c-events'), m = { t: 28, r: 16, b: 30, l: 48 }, f = frame(el, 360, m);
  const rainH = 56, gap = 22, top = f.ih - rainH - gap;
  const x = d3.scaleLinear().domain([0, 121]).range([0, f.iw]);
  const ymax = d3.max([T.actual * 1.25, ...T.pts.map((p) => p.f)]) * 1.05;
  const y = d3.scaleLinear().domain([0, ymax]).nice().range([top, 0]);
  f.g.append('g').attr('class', 'grid').call(d3.axisLeft(y).ticks(4).tickSize(-f.iw).tickFormat(''));
  f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickSizeOuter(0)).call((a) => a.select('.domain').remove());
  yTitle(f, 'Forecast of the June–September total, TMC');
  f.g.append('rect').attr('x', 0).attr('width', f.iw).attr('y', y(T.actual * 1.2)).attr('height', y(T.actual * 0.8) - y(T.actual * 1.2)).attr('fill', 'rgba(29,122,67,.08)');
  f.g.append('line').attr('x1', 0).attr('x2', f.iw).attr('y1', y(T.actual)).attr('y2', y(T.actual)).attr('stroke', C.ink).attr('stroke-dasharray', '5 4');
  f.g.append('text').attr('class', 'lbl2').attr('x', f.iw - 4).attr('y', y(T.actual) - 6).attr('text-anchor', 'end').text(`Actual ${T.actual} TMC`);
  f.g.append('path').datum(T.pts).attr('fill', 'none').attr('stroke', C.ink3).attr('stroke-width', 1.5).attr('d', d3.line().x((p) => x(p.dos)).y((p) => y(p.f)));
  const err = (p) => Math.round((p.f - T.actual) / T.actual * 100);
  const sgn = (v) => (v > 0 ? '+' : '') + v + '%';
  f.g.selectAll(null).data(T.pts).join('circle').attr('cx', (p) => x(p.dos)).attr('cy', (p) => y(p.f)).attr('r', (p) => p.heavy ? 6 : 5)
    .attr('fill', (p) => p.heavy ? C.orange : C.blue).attr('stroke', '#fff').attr('stroke-width', 2)
    .on('mousemove', (ev, p) => showTip(ev, `<b>${p.date} ${evYear}</b>` + (p.heavy ? `<div class="lbl2">heavy rain: ${p.rain} mm over the catchment</div>` : '<div class="lbl2">scheduled forecast</div>')
      + row(p.heavy ? C.orange : C.blue, 'Forecast', `${p.f} TMC (${sgn(err(p))})`)
      + (p.heavy ? row(C.blue, `Last scheduled (${p.prev_date})`, `${p.prev} TMC (${sgn(Math.round((p.prev - T.actual) / T.actual * 100))})`) : '')))
    .on('mouseleave', hideTip);
  const ry = d3.scaleLinear().domain([0, Math.max(40, d3.max(T.rain))]).range([0, rainH]);
  const r0 = top + gap + rainH, heavyDos = new Set(T.pts.filter((p) => p.heavy).map((p) => p.dos));
  f.g.selectAll(null).data(T.rain).join('rect').attr('x', (v, i) => x(i) - 1.2).attr('width', 2.4).attr('y', (v) => r0 - ry(v)).attr('height', (v) => ry(v))
    .attr('fill', (v, i) => heavyDos.has(i) ? C.orange : '#c3ccd5');
  f.g.append('text').attr('class', 'lbl2').attr('x', 0).attr('y', top + gap - 6).text('Daily catchment rain');
  const months = [[0, 'Jun'], [30, 'Jul'], [61, 'Aug'], [92, 'Sep']];
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${r0})`).call(d3.axisBottom(x).tickValues(months.map((m) => m[0])).tickFormat((v) => months.find((m) => m[0] === v)[1]).tickSizeOuter(0));
}
function selEvents() {
  const el = document.getElementById('sel-events'), yrs = Object.keys(D.events.traj).map(Number);
  evYear = D.events.default;
  el.innerHTML = yrs.map((y) => `<option value="${y}" ${y === +evYear ? 'selected' : ''}>${y} · ${D.events.traj[y].pts.filter((p) => p.heavy).length} heavy-rain updates</option>`).join('');
  el.onchange = () => { evYear = el.value; chartEvents(); };
}
const N15 = [{ k: 'Average year', n: 'Assume an average year', c: '#9aa6b2' }, { k: 'Same as last 15 days', n: 'Same as the last 15 days', c: '#9aa6b2' },
  { k: "Today's state, no rain forecast", n: "Today's state, no rain forecast", c: C.blue }, { k: '+ next 7 days of rain', n: '+ 7-day rain forecast', c: C.blue },
  { k: '+ next 15 days of rain', n: '+ 15-day rain forecast', c: C.orange }];
function chartN15Bars() {
  const M = D.next15.models;
  hbar('c-n15bars', N15.map((s) => ({ label: s.n, v: M[s.k].median_pct, c: s.c,
    tip: row('#fff', 'Typical error', `±${Math.round(M[s.k].median_pct)}%`) + row('#fff', '4 years in 5 within', `±${Math.round(M[s.k].p80_pct)}%`)
      + row('#fff', 'Within ±25%', pct(M[s.k].within_25 * 100)) + row('#fff', 'On heavy-rain days', `±${Math.round(M[s.k].heavy_median_pct)}%`)
      + row('#fff', 'NSE', M[s.k].NSE.toFixed(2)) })), (v) => `±${Math.round(v)}%`, C.blue);
}
function chartN15() {
  const S = D.next15.series[n15Year], el = document.getElementById('c-n15'), m = { t: 28, r: 16, b: 30, l: 48 }, f = frame(el, 320, m);
  const rows = S.dates.map((d, i) => ({ t: new Date(d), a: S.actual[i], n: S["Today's state, no rain forecast"][i], r: S['+ next 15 days of rain'][i], c: S['Average year'][i] }));
  const x = d3.scaleTime().domain(d3.extent(rows, (d) => d.t)).range([0, f.iw]);
  const y = d3.scaleLinear().domain([0, d3.max(rows, (d) => Math.max(d.a, d.n, d.r)) * 1.05]).nice().range([f.ih, 0]);
  yAxis(f, y, (v) => v); yTitle(f, 'Inflow over the next 15 days, TMC');
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(d3.timeMonth.every(1)).tickFormat(d3.timeFormat('%b')).tickSizeOuter(0));
  S.heavy.forEach((h) => f.g.append('line').attr('x1', x(new Date(h))).attr('x2', x(new Date(h))).attr('y1', f.ih).attr('y2', f.ih - 10).attr('stroke', C.orange).attr('stroke-width', 3));
  f.g.append('path').datum(rows).attr('fill', 'rgba(21,32,43,.08)').attr('d', d3.area().x((d) => x(d.t)).y0(f.ih).y1((d) => y(d.a)));
  const L = [{ k: 'a', n: 'Actual', c: C.ink, w: 1.5 }, { k: 'c', n: 'Average year', c: C.ink3, w: 1.5, dash: true },
    { k: 'n', n: "Today's state, no rain forecast", c: C.blue, w: 2 }, { k: 'r', n: '+ 15-day rain forecast', c: C.orange, w: 2 }];
  L.forEach((s) => f.g.append('path').datum(rows).attr('fill', 'none').attr('stroke', s.c).attr('stroke-width', s.w).attr('stroke-dasharray', s.dash ? '5 4' : null)
    .attr('d', d3.line().x((d) => x(d.t)).y((d) => y(d[s.k]))));
  legend(document.getElementById('l-n15'), L.map((s) => ({ n: s.n, c: s.c, cls: s.dash ? 'dash' : '' })).concat([{ n: 'Heavy-rain day', c: C.orange, cls: 'dot' }]));
  const bis = d3.bisector((d) => d.t).center, hv = new Set(S.heavy);
  const v = f.g.append('line').attr('y1', 0).attr('y2', f.ih).attr('stroke', C.ink3).attr('opacity', 0);
  f.g.append('rect').attr('width', f.iw).attr('height', f.ih).attr('fill', 'transparent')
    .on('mousemove', (ev) => { const d = rows[bis(rows, x.invert(d3.pointer(ev)[0]))];
      v.attr('x1', x(d.t)).attr('x2', x(d.t)).attr('opacity', .5);
      showTip(ev, `<b>Forecast made ${d3.timeFormat('%d %b %Y')(d.t)}</b>` + (hv.has(d3.timeFormat('%Y-%m-%d')(d.t)) ? '<div class="lbl2">heavy-rain day</div>' : '')
        + L.map((s) => row(s.c, s.n, d[s.k].toFixed(1) + ' TMC')).join('')); })
    .on('mouseleave', () => { v.attr('opacity', 0); hideTip(); });
}
function segN15() {
  const el = document.getElementById('seg-n15'), yrs = Object.keys(D.next15.series);
  n15Year = yrs.reduce((a, b) => d3.max(D.next15.series[b].actual) > d3.max(D.next15.series[a].actual) ? b : a);
  yrs.forEach((k) => { const b = document.createElement('button'); b.textContent = k; b.setAttribute('aria-pressed', k === n15Year);
    b.onclick = () => { n15Year = k; el.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b)); chartN15(); };
    el.appendChild(b); });
}

/* ---- comparison with the published Hemavathi study (Hemavathy report only) ----------- */
const PROWS = [
  { k: null, n: 'Paper, as reported (neural network)' },
  { k: 'Paper method: network on flows from the formula', n: 'Paper method on our data: network on formula flows' },
  { k: 'Paper method: network on measured flows (1-2 days before)', n: 'Paper method on our data: network on measured flows' },
  { k: 'Same as yesterday', n: '“Same as yesterday”' },
  { k: 'Our model: rain + soil (no inflow readings)', n: 'Our model: rain + soil (no inflow readings)' },
  { k: "Our model: + last week's inflow", n: 'Our model + last week’s inflow' }];
let paperMetric = 'NSE', paperProto = 'ours';
function chartPaperScores() {
  const P = D.paper, el = document.getElementById('c-pscores'), m = { t: 30, r: 40, b: 30, l: Math.min(330, el.clientWidth * 0.5) };
  const f = frame(el, PROWS.length * 40 + 40, m);
  const y = d3.scaleBand().domain(PROWS.map((r) => r.n)).range([0, f.ih]).padding(0.4);
  const val = (r, pr) => r.k === null ? (pr === 'paper' ? P.paper_reported.nn[paperMetric] : null) : P.protocols[pr].models[r.k][paperMetric];
  const lo = Math.min(0, d3.min(PROWS.flatMap((r) => ['paper', 'ours'].map((pr) => val(r, pr)).filter((v) => v != null))));
  const x = d3.scaleLinear().domain([Math.floor(lo * 10) / 10, 1]).range([0, f.iw]);
  f.g.append('g').attr('class', 'grid').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(5).tickSize(-f.ih).tickFormat(''));
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format('.1f')).tickSizeOuter(0));
  f.g.append('text').attr('class', 'lbl2').attr('x', 0).attr('y', -14).text(`${paperMetric} (1 = perfect)`);
  const ref = P.paper_reported.nn[paperMetric];
  f.g.append('line').attr('x1', x(ref)).attr('x2', x(ref)).attr('y1', -6).attr('y2', f.ih).attr('stroke', C.ink3).attr('stroke-dasharray', '4 4');
  PROWS.forEach((r) => {
    const cy = y(r.n) + y.bandwidth() / 2, a = val(r, 'paper'), b = val(r, 'ours');
    f.g.append('text').attr('class', r.k && r.k.startsWith('Our') ? 'lbl' : 'lbl2').attr('x', -12).attr('y', cy + 4).attr('text-anchor', 'end').text(r.n);
    if (a != null && b != null) f.g.append('line').attr('x1', x(a)).attr('x2', x(b)).attr('y1', cy).attr('y2', cy).attr('stroke', '#c3ccd5').attr('stroke-width', 3);
    [['paper', a, C.orange, 'Random test days (paper’s test)'], ['ours', b, C.blue, 'Unseen years 2020–2023 (our test)']].forEach(([pr, v, c, lab]) => {
      if (v == null) return;
      f.g.append('circle').attr('cx', x(v)).attr('cy', cy).attr('r', 7).attr('fill', c).attr('stroke', '#fff').attr('stroke-width', 2)
        .on('mousemove', (ev) => { const M = r.k ? P.protocols[pr].models[r.k] : P.paper_reported.nn;
          showTip(ev, `<b>${r.n}</b><div class="lbl2">${r.k ? lab : 'reported in the paper (random split)'}</div>` + ['NSE', 'KGE', 'R2'].map((k) => row(c, k === 'R2' ? 'R²' : k, M[k].toFixed(2))).join('')
            + (M.bias_pct != null ? row(c, 'Average bias', (M.bias_pct > 0 ? '+' : '') + M.bias_pct.toFixed(0) + '%') : '')); })
        .on('mouseleave', hideTip);
    });
  });
  f.g.append('text').attr('class', 'lbl2').attr('x', x(ref) + 4).attr('y', -2).text('paper');
}
function segPaper() {
  const el = document.getElementById('seg-pmetric');
  ['NSE', 'KGE', 'R2'].forEach((k) => { const b = document.createElement('button'); b.textContent = k === 'R2' ? 'R²' : k; b.setAttribute('aria-pressed', k === paperMetric);
    b.onclick = () => { paperMetric = k; el.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b)); chartPaperScores(); }; el.appendChild(b); });
  const el2 = document.getElementById('seg-pproto');
  [['ours', 'Unseen years 2020–2023'], ['paper', 'Random test days (paper’s test)']].forEach(([k, n]) => { const b = document.createElement('button'); b.textContent = n; b.setAttribute('aria-pressed', k === paperProto);
    b.onclick = () => { paperProto = k; el2.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b)); chartPaperMonths(); }; el2.appendChild(b); });
}
function chartPaperMonths() {
  const P = D.paper, Pr = P.protocols[paperProto], months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'];
  const S = [{ n: 'Paper, as reported (best formula: polynomial)', c: C.ink3, dash: true, v: P.paper_reported.formulas_nse.Polynomial },
    { n: 'Paper’s polynomial formula on our data', c: C.orange, v: Pr.formulas.Polynomial },
    { n: 'Our model: rain + soil', c: C.blue, v: months.map((mm) => Pr.models['Our model: rain + soil (no inflow readings)'].by_month[mm]) },
    { n: 'Our model + last week’s inflow', c: C.green, v: months.map((mm) => Pr.models["Our model: + last week's inflow"].by_month[mm]) }];
  const el = document.getElementById('c-pmonths'), m = { t: 28, r: 16, b: 30, l: 44 }, f = frame(el, 300, m);
  const x = d3.scalePoint().domain(months).range([0, f.iw]).padding(0.3);
  const lo = Math.min(0, d3.min(S.flatMap((s) => s.v)));
  const y = d3.scaleLinear().domain([Math.floor(lo * 5) / 5, 1]).range([f.ih, 0]);
  yAxis(f, y, d3.format('.1f')); yTitle(f, 'NSE by month (1 = perfect)');
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).tickSizeOuter(0));
  if (lo < 0) f.g.append('line').attr('x1', 0).attr('x2', f.iw).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', C.ink3);
  S.forEach((s) => { const pts = months.map((mm, i) => ({ mm, v: s.v[i] }));
    f.g.append('path').datum(pts).attr('fill', 'none').attr('stroke', s.c).attr('stroke-width', 2.5).attr('stroke-dasharray', s.dash ? '6 4' : null).attr('d', d3.line().x((p) => x(p.mm)).y((p) => y(p.v)));
    f.g.selectAll(null).data(pts).join('circle').attr('cx', (p) => x(p.mm)).attr('cy', (p) => y(p.v)).attr('r', 4).attr('fill', s.c).attr('stroke', '#fff').attr('stroke-width', 2); });
  legend(document.getElementById('l-pmonths'), S.map((s) => ({ n: s.n, c: s.c, cls: s.dash ? 'dash' : '' })));
  crosshair(f, x, months, (mm) => `<b>${mm}</b>` + S.map((s) => row(s.c, s.n, s.v[months.indexOf(mm)].toFixed(2))).join(''), (mm) => mm);
}

function drawAll() {
  chartForecast(); chartStack(); chartRelations(); chartScatter();
  const R = D.lag_rain, bands = [{ k: 'far', n: 'Far (Ghats)', c: C.blue }, { k: 'mid', n: 'Middle', c: C.orange }, { k: 'near', n: 'Near dam', c: C.green }];
  lagChart('c-lag', 'l-lag', bands.map((b) => ({ n: b.n, c: b.c, v: R[b.k] })), 'Extra inflow per mm of rain, cusecs', (v) => fmt0(v), (v) => fmt0(v) + ' cusecs');
  if (D.lag_up) lagChart('c-lagup', 'l-lagup', [{ n: 'Harangi', c: C.blue, v: D.lag_up.Harangi }, { n: 'Hemavathy', c: C.orange, v: D.lag_up.Hemavathy }],
    'Share of each released cusec reaching KRS', (v) => Math.round(v * 100) + '%', (v) => Math.round(v * 100) + '%');
  cellMap('c-mapg', 'l-mapg', 'gain_cusec_days_per_mm', d3.interpolate('#dbe8f8', '#0b3d91'), (v) => fmt0(v), 'Multiplier');
  cellMap('c-mapd', 'l-mapd', 'delay_days', d3.interpolate('#fde3d6', '#9c2f06'), (v) => v.toFixed(1) + ' d', 'Delay');
  chartHydro();
  const S = leadChart('c-lead1', 'high_flow_caught_pct', 'High-flow days caught', (v) => Math.round(v) + '%', [0, 100]);
  leadChart('c-lead2', 'peak_median_err_pct', 'Flood-peak size error (median)', (v) => Math.round(v) + '%', [-100, 0]);
  legend(document.getElementById('l-lead'), S.map((s) => ({ n: s.n, c: s.c })));
  if (D.importance) hbar('c-imp', Object.entries(D.importance).map(([k, v]) => ({ label: k.replace(' (API, since Jun 1)', ''), v: v * 100,
    tip: row(C.blue, 'Accuracy lost without it', `${(v).toFixed(2)} NSE`) })), (v) => v.toFixed(0), C.blue);
  if (D.river) hbar('c-river', D.river.map((r, i) => ({ label: r.label, v: r.rise, c: i === 0 ? C.ink3 : C.blue,
    tip: row('#fff', 'Peak size error', `${r.peak_err}%`) + row('#fff', 'Share of sharp rises caught', `${r.rise}%`) + row('#fff', 'Overall NSE', r.nse) })),
    (v) => `${v}%`, C.blue, 100);
  if (D.vs_krs) chartVsKrs();
  chartLadder(); chartMatrix(); chartSoil(); chartEvents(); chartN15Bars(); chartN15();
  if (D.paper) { chartPaperScores(); chartPaperMonths(); }
}
segIssue(); segYear(); selEvents(); segN15(); if (D.paper) segPaper(); drawAll();
let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(drawAll, 150); });
"""


def page(d):
    s, at = d["short"], d["at"]
    rf, rfr = d["reliable_from"], d["reliable_from_rain"]
    lead = {(r["lead_days"], r["variant"]): r for r in d["lead"]}
    caught = lambda h, v: round(lead[(h, v)]["high_flow_caught_pct"])
    good_days = max(h for h in [1, 2, 3, 5, 7] if lead[(h, "Known on the day")]["high_flow_caught_pct"] >= 65)
    tiles = [
        ("Reliable monsoon-total estimate", rf, f"4 years in 5 within ±{GOOD}% · {rfr} with a 30-day rain forecast"),
        (f"Error on {rf}", f"±{round(at[rf]['none_med'])}%", f"typical · vs ±{round(at[rf]['clim_med'])}% assuming an average year"),
        ("Rain reaches the dam in", f"{d['rain_peak_day']} day{'s' if d['rain_peak_day'] != 1 else ''}",
         f"peak response · {d['rain_week_share']}% of it arrives within a week"),
        ("High-flow warning", f"{good_days} day{'s' if good_days > 1 else ''} ahead",
         f"{caught(good_days, 'Known on the day')}% of high-flow days caught · {caught(7, '+ rain forecast')}% at 7 days with a rain forecast"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{t}</div></div>'
                         for k, v, t in tiles)
    yrs = d["years"]
    dry = sorted(yrs, key=lambda y: y["total"])[: len(yrs) // 3]
    lo_hi = f"{round(d['min_total'])}–{round(d['max_total'])} TMC"
    if d["dam"] == "krs":
        d["hydroNames"] = ["Model: rain + upstream releases", "+ last week's KRS inflow"]
        rv = d["river"]
        best = max(rv[1:], key=lambda r: r["rise"])
        same_peak = abs(best["peak_err"] - rv[0]["peak_err"]) < 3
        if best["rise"] > rv[0]["rise"] + 2:
            river_title = (f"Using the river network catches more of each sharp rise ({rv[0]['rise']}% → {best['rise']}%)"
                           + ("; flood-peak size and overall fit barely change" if same_peak else
                              f"; flood-peak size error {rv[0]['peak_err']:.0f}% → {best['peak_err']:.0f}%"))
        else:
            river_title = (f"Grouping rain by river distance does not beat straight-line distance on 2020–2023 "
                           f"(sharp rises caught: {rv[0]['rise']}% vs at best {best['rise']}%)")
        extra = f"""
<div class="card"><h2>Upstream releases and soil wetness drive KRS inflow, not today's rain</h2>
<p class="sub">How much the daily model's accuracy drops when each input is scrambled. Bigger bar = more important.</p>
<div class="chart" id="c-imp"></div></div>
<div class="card"><h2>{river_title}</h2>
<p class="sub">Share of each sharp rise in inflow (5,000+ cusecs in a day) that the daily model predicts on the day, 2020–2023, for different ways of using the river network. Hover a bar for flood-peak size error and overall fit (NSE).</p>
<div class="chart" id="c-river"></div></div>"""
        lag_html = """<div class="row2"><div><h3>Rain on the free-draining catchment</h3><div class="legend" id="l-lag"></div><div class="chart" id="c-lag"></div></div>
<div><h3>Releases from upstream dams</h3><div class="legend" id="l-lagup"></div><div class="chart" id="c-lagup"></div></div></div>"""
        lag_title = "Upstream releases arrive within 1 day; rain takes about 2 days and keeps flowing for a week"
    else:
        d["hydroNames"] = ["Model: rain only", "+ last week's Hemavathy inflow"]
        v = d["vs_krs_daily"]
        extra = f"""
<div class="card"><h2>Hemavathy and KRS are equally hard to call until August; Hemavathy pulls ahead from mid-August</h2>
<p class="sub">Monsoon-total error by forecast date. Daily inflow from rain alone scores NSE {v['hem_rain']:.2f} at Hemavathy vs {v['krs_rain']:.2f} at KRS; KRS reaches {v['krs_up']:.2f} only because it also sees upstream releases. Hemavathy has no gauged dam upstream.</p>
<div class="legend" id="l-vskrs"></div><div class="chart" id="c-vskrs"></div></div>"""
        lag_html = '<div class="legend" id="l-lag"></div><div class="chart" id="c-lag"></div>'
        lag_title = f"Rain shows up at Hemavathy within {d['rain_peak_day']} day; most comes from the middle and far (Ghats) cells"
    b, a = d["cell_cmp"]["before"], d["cell_cmp"]["after"]
    cmp_tiles = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{t}</div></div>' for k, v, t in [
        ("Daily accuracy (NSE)", f"{b['NSE']:.2f} → {a['NSE']:.2f}", "using per-cell routing instead of 3 distance bands"),
        ("Flood-peak size error", f"{b['peak_median_err_pct']:.0f}% → {a['peak_median_err_pct']:.0f}%", "median, same-day, 2020–2023"),
        ("Sharp rises caught", f"{round(b['rise_capture_median'] * 100)}% → {round(a['rise_capture_median'] * 100)}%", "share of each 5,000+ cusec jump predicted")])
    es, mean = d["eda_stats"], d["eda_stats"]["mean"]
    eda_title = (f"Dry years ripple through: {abs(es['dry_rain_pct'])}% less rain brings {abs(es['dry_inflow_pct'])}% less inflow, "
                 f"and {s} ends September {es['dry_full_pct']}% full instead of {es['wet_full_pct']}%")
    rel1 = f"Every extra 100 mm of rain adds about {es['rain_slope_per_100mm']:.0f} TMC of inflow (r = {es['r_rain_inflow']})"
    rel2 = (f"Above {es['fill_threshold']} TMC of inflow, {s} ends September ≥90% full in {es['fill_above']} years; below it, {es['fill_below']}"
            if es["fill_threshold"] else f"Inflow vs storage on 30 Sep (r = {es['r_inflow_storage']})")
    rel3 = f"Releases are about {es['release_share']}% of monsoon inflow"
    so, oth = d["soil"], d["soil_other"]
    gain = so["with"]["NSE"] - so["without"]["NSE"]
    soil_title = (f"Soil moisture data improves the daily forecast: NSE {so['without']['NSE']:.2f} → {so['with']['NSE']:.2f}"
                  + (", more than at KRS" if d["dam"] == "hemavathy" else ", less than at Hemavathy"))
    soil_why = ("<b>Hemavathy gains more because its inflow comes straight from rain on its own catchment</b>, so how wet the soil is "
                f"directly decides how much rain runs off (KRS gains only {oth['without']:.2f} → {oth['with']:.2f})."
                if d["dam"] == "hemavathy" else
                "<b>KRS gains less because it already gets most of its information from the upstream releases</b> (Harangi and Hemavathy), "
                f"which already reflect how wet their catchments are (Hemavathy gains {oth['without']:.2f} → {oth['with']:.2f}).")
    improved = sum(so["with"]["nse_by_year"][y] > so["without"]["nse_by_year"][y] + 0.005 for y in so["with"]["nse_by_year"])
    soil_tiles = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{t}</div></div>' for k, v, t in [
        ("Daily accuracy (NSE)", f"{so['without']['NSE']:.2f} → {so['with']['NSE']:.2f}", f"better in {improved} of {len(so['with']['nse_by_year'])} test years"),
        ("Monsoon months only", f"{so['without']['monsoon_NSE']:.2f} → {so['with']['monsoon_NSE']:.2f}", "NSE, June–September"),
        ("Flood peaks within ±25%", f"{round(so['without']['peak_within_25pct'] * 100)}% → {round(so['with']['peak_within_25pct'] * 100)}%", "same-day peak size")])
    mdiff = [b["with"] - b["without"] for b in so["monsoon"]]
    soil_note = (f"<b>The monsoon-total forecast does not improve:</b> with soil data its error moves by {min(mdiff):+.0f} to {max(mdiff):+.0f} points "
                 f"depending on the date, which is noise for 33 years, so the monsoon-total forecast stays without it. "
                 f"<b>Caveats:</b> this soil data is published with a long delay, so in real use a near-real-time soil-moisture product is needed; "
                 f"it ends in 2023, which is why daily scores in this report use 2020–2023. It overlaps our rain-based soil-wetness index "
                 f"(correlation {so['corr']}) but its month-to-month departures differ (correlation {so['corr_anom']}), which is where the extra information comes from.")
    es = d["events"]["summary"]
    same = abs(es["median_gain_points"] - es["ordinary_day_median_gain_points"]) < 1
    ev_title = ("An extra forecast on a heavy-rain day helps, but no more than one on any other day" if same else
                ("Heavy-rain days are worth an extra forecast: it beats an update on an ordinary day" if es["median_gain_points"] > es["ordinary_day_median_gain_points"]
                 else "An extra forecast on a heavy-rain day helps less than one on an ordinary day"))
    bm = {r["month"]: r for r in d["events"]["by_month"]}
    ev_tiles = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{t}</div></div>' for k, v, t in [
        ("Heavy-rain updates", f"{es['per_year']:.0f} a year", f"{es['events']} in 35 years · rain above {es['threshold_mm']:.0f} mm in a day over the catchment"),
        ("Typical error", f"±{es['err_last_scheduled_median']:.0f}% → ±{es['err_updated_median']:.0f}%", f"last scheduled forecast (median {es['median_wait_days']:.0f} days earlier) → updated on the heavy-rain day"),
        ("Updates that improve", f"{round(es['better_share'] * 100)}%", f"vs {round(es['ordinary_day_better_share'] * 100)}% for an update on an ordinary day after the same wait")])
    ev_note = (f"<b>Why heavy rain adds little:</b> on the day of the storm its water has not reached the dam yet, and most of the season's uncertainty is the rain still to come. "
               f"In June an update does not help at all (typical error {bm['June']['before']:.0f}% → {bm['June']['after']:.0f}%); in August it does "
               f"({bm['August']['before']:.0f}% → {bm['August']['after']:.0f}%), but so does any later update, because more of the season has already been measured. "
               "So: keep the regular dates, and treat a heavy-rain update as a fresh look, not a better forecast.")
    N = d["next15"]["models"]
    n0, n7, n15 = N["Today's state, no rain forecast"], N["+ next 7 days of rain"], N["+ next 15 days of rain"]
    n15_title = (f"Water arriving in the next 15 days: typical error ±{n15['median_pct']:.0f}% with a 15-day rain forecast, "
                 f"±{n0['median_pct']:.0f}% from today's state alone")
    n15_tiles = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{t}</div></div>' for k, v, t in [
        ("Accuracy (NSE)", f"{n0['NSE']:.2f} → {n7['NSE']:.2f} → {n15['NSE']:.2f}", "no rain forecast → 7-day → 15-day rain forecast"),
        ("Within ±25%", f"{round(n0['within_25'] * 100)}% → {round(n15['within_25'] * 100)}%", "share of forecast days, without → with a 15-day rain forecast"),
        ("On heavy-rain days", f"±{n15['heavy_median_pct']:.0f}%", f"typical error with a 15-day rain forecast ({d['next15']['n_heavy']} heavy-rain days in 2020–2023)")])
    jun = lambda r: r["by_month"].get("6", r["by_month"].get(6))
    n15_note = (f"<b>June is hardest:</b> the catchment is still dry, so the same rain gives very different inflow "
                f"(typical June error ±{jun(n15):.0f}% even with the rain forecast). "
                "The rain forecast here is the rain that actually fell, so these numbers are the best a real forecast could reach. "
                + ("Future releases from Harangi and Hemavathy are not known in advance, so the model only sees today's releases." if d["dam"] == "krs" else ""))
    paper_html = ""
    if d.get("paper"):
        PP = d["paper"]
        pm = lambda pr, k: PP["protocols"][pr]["models"][k]
        oa, orr = pm("paper", "Our model: + last week's inflow"), pm("paper", "Our model: rain + soil (no inflow readings)")
        ua, ur = pm("ours", "Our model: + last week's inflow"), pm("ours", "Our model: rain + soil (no inflow readings)")
        rep_ = PP["paper_reported"]["nn"]
        pf, pq = pm("paper", "Paper method: network on flows from the formula"), pm("paper", "Paper method: network on measured flows (1-2 days before)")
        uf, uq = pm("ours", "Paper method: network on flows from the formula"), pm("ours", "Paper method: network on measured flows (1-2 days before)")
        py, uy = pm("paper", "Same as yesterday"), pm("ours", "Same as yesterday")
        mo = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
        best_paper = PP["paper_reported"]["formulas_nse"]["Polynomial"]
        win = lambda r: sum(r["by_month"][m] > b for m, b in zip(mo, best_paper))
        month_title = (f"On unseen years, our model beats the paper's best monthly formula in {win(ua)} of 6 months with last week's inflow, "
                       f"and in {win(ur)} of 6 from rain alone; October and November are hardest for every approach")
        paper_tiles = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{t}</div></div>' for k, v, t in [
            ("Paper's test, our model", f"{orr['NSE']:.2f}", f"NSE from rain and soil alone (KGE {orr['KGE']:.2f}) vs {rep_['NSE']:.2f} reported by the paper"),
            ("Paper's method on our data", f"{pf['NSE']:.2f}–{pq['NSE']:.2f}", f"NSE on its own test; {uf['NSE']:.2f}–{uq['NSE']:.2f} on unseen years"),
            ("“Same as yesterday”", f"{py['NSE']:.2f}", f"NSE on the paper's test (KGE {py['KGE']:.2f}) with no model at all; the paper reports {rep_['NSE']:.2f} / {rep_['KGE']:.2f}")])
        paper_html = f"""
<h2 class="sec">Compared with a published study</h2>
<div class="card"><h2>Scored the paper's way, our model beats the published Hemavathi model: NSE {orr['NSE']:.2f}–{oa['NSE']:.2f} vs {rep_['NSE']:.2f}</h2>
<p class="sub">C. Chandre Gowda (2025), <i>Inflow prediction and modelling for a seasonal flow reservoir</i>, Water-Energy Nexus 8, 281–286, models daily inflow to the same reservoir. We rebuilt its method on our data and scored every approach two ways:
<b style="color:#eb6834">random test days</b>, as in the paper (70% / 15% / 15% of June–November days, so the days either side of each test day are in training), and
<b style="color:#2a78d6">unseen years</b>, as in the rest of this report (trained on 1991–2019, tested on all of June–November 2020–2023). Dashed line = the paper's reported score. Hover a dot for all scores.</p>
<div class="seg" id="seg-pmetric"></div>
<div class="legend"><span><i class="dot" style="background:#eb6834"></i>Random test days (paper’s test)</span><span><i class="dot" style="background:#2a78d6"></i>Unseen years 2020–2023 (our test)</span><span><i class="dash" style="background:#15202b"></i>Paper’s reported score</span></div>
<div class="chart" id="c-pscores"></div>
<div class="tiles" style="margin-top:14px">{paper_tiles}</div>
<p class="sub" style="margin-top:12px"><b>What this shows:</b> on random test days every model looks better, because the days just before and after each test day, from the same storms, are in its training data; “same as yesterday” alone scores NSE {py['NSE']:.2f}. On unseen years, the paper's network on formula flows drops to {uf['NSE']:.2f}, while our model holds {ua['NSE']:.2f} with last week's inflow.
<b>Where we fall short:</b> without any inflow readings, our model over-predicts unseen years by {ur['bias_pct']:.0f}% on average, which pulls its KGE down to {ur['KGE']:.2f}; that is the next thing to fix.</p></div>

<div class="card"><h2>{month_title}</h2>
<p class="sub">NSE for each month, June–November. The paper fits one formula per month from rain on the day and the 2 days before (its 2-day lag comes from a 12-hour time of concentration). Pick how the test days are chosen.</p>
<div class="seg" id="seg-pproto"></div>
<div class="legend" id="l-pmonths"></div><div class="chart" id="c-pmonths"></div>
<p class="sub" style="margin-top:12px"><b>How we rebuilt it:</b> catchment rain is our IMD grid weighted by area (the paper uses Thiessen weights for 4 rain gauges). The four formulas (linear, polynomial, logarithmic, exponential) are fitted per month on rain lags 0–2. The network has 2 inputs, 14 tanh hidden neurons and a linear output, best of 5 random starts on the validation days. The paper does not list the network's two inputs exactly, so we tried both readings: the formula's flows for today and yesterday, and measured inflow 1 and 2 days before.
<b>Not comparable:</b> the paper's RMSE (1.48 m³/s) and MAE (0.94 m³/s) are far below its own monthly tables (50–770 m³/s), so they were probably computed on rescaled data; it gives the catchment as 5,410 km², which looks like the whole Hemavathi basin, while {d['area_km2']:,} km² drains to the dam.</p></div>"""
    L = d["ladder"]
    fin, clim = next(r for r in L if r["final"]), L[0]
    ladder_title = (f"From NSE {clim['nse']:.2f} (seasonal average) to {fin['nse']:.2f}: "
                    + ("upstream releases add the most; per-cell rain response and soil moisture add a little more" if d["dam"] == "krs"
                       else (lambda b, l: f"per-cell rain response and soil moisture lift rain-only accuracy from {b['nse']:.2f} to {fin['nse']:.2f}; "
                             f"last week's inflow takes it to {l['nse']:.2f}")(next(r for r in L if r["label"] == "Rain in 3 distance bands"), L[-1])))
    M = d["matrix"]
    first_ok = lambda r: next((M["issues"][j] for j, v in enumerate(r["p80"]) if v <= GOOD), "never")
    matrix_title = (f"Without a rain forecast the estimate becomes reliable on {first_ok(M['rows'][1])}; a 30-day rain forecast brings that to "
                    f"{first_ok(M['rows'][4])}, while knowing only the next 7 days helps less")
    up = d["dam"] == "krs"
    node = lambda cls, title, body: f'<div class="node {cls}"><b>{title}</b>{body}</div>'
    stage = lambda n, title, sub, nodes, two=False, four=False, three=False: (f'<div class="stage"><div class="sh"><span>{n}</span>{title}<small>{sub}</small></div>'
                                                                  f'<div class="nodes{" two" if two else ""}{" four" if four else ""}{" three" if three else ""}">{"".join(nodes)}</div></div>')
    down = '<div class="down"></div>'
    flow = '<div class="flow">' + down.join([
        stage(1, "Collect data", "1991–2025, daily", four=True, nodes=[
            node("n-data", "Rainfall", f"IMD 0.25° gridded daily rainfall (cells of about 27 × 27 km). {d['cells_total']} cells cover the {'free-draining ' if up else ''}catchment."),
            node("n-data", "Dam records", "Daily storage, % full, inflow and outflow"
                 + (" for KRS and the two upstream dams, Harangi and Hemavathy." if up else " for Hemavathy.")),
            node("n-data", "Maps", f"Catchment boundary ({d['area_km2']:,} km²) and river lines with stream order."),
            node("n-data", "Soil moisture", f"TerraClimate monthly soil moisture, water deficit and evaporation for {len(d['soil']['sub_watersheds'])} sub-watersheds, 1991–2023.")]),
        stage(2, "Clean and check", f"{len(d['fixes'])} fixes for {s}, all logged", [
            node("n-prep", "Fix what is clearly wrong",
                 "<ul><li>Missing days re-added; gaps of up to 3 days filled</li><li>Storage above capacity and one-day spikes removed</li>"
                 "<li>Zero inflow in the monsoon treated as missing</li><li>Jul–Sep 2019 outflow was not in cusecs: rebuilt as inflow minus storage change</li></ul>"),
            node("n-prep", "Flag what looks odd",
                 "<ul><li>Days where storage change does not match inflow minus outflow</li><li>June 2012: very heavy rain but little inflow</li></ul>Flagged only, not changed.")]),
        stage(3, "Build the inputs", "what the models see each day", four=True, nodes=[
            node("n-prep", "Catchment rain", "Each cell weighted by its area inside the catchment, and grouped into near, middle and far bands from the dam."),
            node("n-prep", "Per-cell response", "For every cell: the share of its rain that reaches the dam on each of the next 14 days; wetter soil gives more. "
                 + ("Rain above Hemavathy dam is left out: it reaches KRS only through Hemavathy's releases. " if up else "")
                 + "Result: “routed rain”, the inflow expected from rain alone."),
            node("n-prep", "Soil wetness", "A running total of recent rain that slowly fades, plus last month's measured soil moisture and how far it is from normal."),
            node("n-prep", "Upstream releases" if up else "Time of season",
                 "Harangi and Hemavathy releases over the last 5 days, and how full they are." if up
                 else "Day of the year, so the model knows where in the monsoon it is.")]),
        stage(4, "Three models", "days, the next 15 days, the season", three=True, nodes=[
            node("n-model", "A. Daily inflow model",
                 "<ul><li>Gradient boosting (many small decision trees)</li><li>Predicts each day's inflow from the inputs above</li>"
                 "<li>Learns from 1991–2019</li><li>Also run 1–7 days ahead, with and without a rain forecast</li></ul>"),
            node("n-model", "B. Monsoon-total forecast",
                 "<ul><li>On each forecast date (1 Jun … 15 Sep): total = inflow already measured + predicted rest of season</li>"
                 "<li>Rest of season from inflow so far, last week's inflow, rain so far, May rain, last 30 days' rain, storage"
                 + (" and upstream storage" if up else "") + "</li><li>Simple linear (ridge) model, one per forecast date</li>"
                 "<li>Re-issued on every heavy-rain day too (top 5% of monsoon days), with a local model fitted on nearby dates</li>"
                 "<li>Variant: add the next 7–30 days' actual rain as a perfect rain forecast</li></ul>"),
            node("n-model", "C. Next 15 days",
                 "<ul><li>Gradient boosting on today's state: recent inflow and rain, soil wetness, storage"
                 + (", upstream releases" if up else "") + "</li><li>Predicts the total inflow of the next 15 days</li>"
                 "<li>With and without the rain of the next 7 or 15 days</li></ul>")]),
        stage(5, "Test fairly", "never score on data the model learned from", [
            node("n-model", "Daily model", "Learns from 1991–2019, scored only on 2020–2023 (soil data ends 2023). NSE: 1 = perfect, 0 = no better than the seasonal average. Flood peaks: size error on the peak day."),
            node("n-model", "Monsoon total", f"Each of the {len(yrs)} years is forecast by a model trained on the other years. Error = % off that year's actual total."),
            node("n-model", "Next 15 days", "Learns from 1991–2019, scored on every June–September day of 2020–2023. Error = % off the actual 15-day inflow.")], three=True),
        stage(6, "Results", "what an engineer gets", [
            node("n-out", "Daily inflow and flood warning",
                 f"NSE {d['daily_nse']:.2f}; high flows flagged {good_days} day{'s' if good_days > 1 else ''} ahead, up to a week with a rain forecast."),
            node("n-out", "Monsoon total by date", f"Reliable (4 years in 5 within ±{GOOD}%) from {rf}; from {rfr} with a 30-day rain forecast."),
            node("n-out", "Next 15 days", f"Typical error ±{n15['median_pct']:.0f}% with a 15-day rain forecast, ±{n0['median_pct']:.0f}% without.")], three=True),
    ]) + "</div>"
    fixes = "".join(f"<tr><td>{f['issue']}</td><td>{f['action']}</td><td class='n'>{f['count']}</td></tr>" for f in d["fixes"])
    a1 = at["1 Aug"]
    body = f"""
<div class="wrap">
<header>
  <h1>{d['name']}: monsoon inflow outlook</h1>
  <p>How early can engineers know the total June–September inflow? Daily IMD rainfall over the {d['area_km2']:,} km² catchment and daily dam records, 1991–2025.</p>
</header>
<div class="tiles">{tiles_html}</div>

<h2 class="sec">The monsoon, year by year</h2>
<div class="card"><h2>{eda_title}</h2>
<p class="sub">June–September totals for every year, 1991–2025. Orange = the driest third of years by inflow (the same years in every row); % differences in the title compare them with the other years. Dashed line = average. Hover a year to compare.</p>
<div class="legend"><span><i class="area" style="background:#eb6834"></i>Dry third of years</span><span><i class="area" style="background:#2a78d6"></i>Other years</span><span><i class="dash" style="background:#15202b"></i>Average</span><span><i style="background:#15202b;height:3px"></i>Storage on 1 Jun (start of monsoon)</span></div>
<div class="chart" id="c-stack"></div></div>

<div class="card"><h2>How rain, inflow, releases and end-of-monsoon storage relate</h2>
<p class="sub">Each dot is one year. Orange = dry third of years.</p>
<div class="row3">
<div><h3>{rel1}</h3><div class="chart" id="c-r1"></div></div>
<div><h3>{rel2}</h3><div class="chart" id="c-r2"></div></div>
<div><h3>{rel3}</h3><div class="chart" id="c-r3"></div></div>
</div></div>

<h2 class="sec">Forecasting the monsoon inflow</h2>
<div class="card"><h2>A reliable estimate of the monsoon total is possible from {rf}, or {rfr} with a 30-day rain forecast</h2>
<p class="sub">Error in the forecast of total June–September inflow, by the date it is made. Lower is better. Each year is forecast by a model that never saw it.</p>
<div class="legend" id="l-forecast"></div><div class="chart" id="c-forecast"></div></div>

<div class="card"><h2>Forecast vs actual, every year</h2>
<p class="sub"><span id="s-scatter"></span> Early forecasts struggle because a median {round(d['after_aug1_median'])}% of the season's inflow arrives after 1 August.</p>
<div class="seg" id="seg-issue"></div>
<div class="legend"><span><i class="dot" style="background:#2a78d6"></i>No rain forecast</span><span><i class="dot" style="background:#eb6834"></i>+ 30-day rain forecast</span><span><i class="area" style="background:rgba(29,122,67,.15)"></i>within ±20%</span></div>
<div class="chart" id="c-scatter" style="max-width:620px"></div></div>

<div class="card"><h2>{ev_title}</h2>
<p class="sub">The monsoon-total forecast, re-issued on the regular dates (blue) and on every heavy-rain day (orange): a day when catchment-average rain is in the top 5% of monsoon days. Pick a year; hover a dot. Green band = within ±20% of the actual total.</p>
<select class="yr" id="sel-events" aria-label="Year"></select>
<div class="legend"><span><i class="dot" style="background:#2a78d6"></i>Regular date (1st and 15th)</span><span><i class="dot" style="background:#eb6834"></i>Heavy-rain day</span><span><i class="dash" style="background:#15202b"></i>Actual total</span></div>
<div class="chart" id="c-events"></div>
<div class="tiles" style="margin-top:14px">{ev_tiles}</div>
<p class="sub" style="margin-top:12px">{ev_note}</p></div>

<h2 class="sec">The next 15 days</h2>
<div class="card"><h2>{n15_title}</h2>
<p class="sub">Total inflow over the 15 days after each forecast day, June–September of the test years 2020–2023 (never seen in training). Inputs: recent inflow and rain, soil wetness and soil moisture, storage{', upstream releases' if d['dam'] == 'krs' else ''}; plus, optionally, the rain of the coming days. Typical error, lower is better; hover a bar for more.</p>
<div class="chart" id="c-n15bars"></div>
<div class="tiles" style="margin-top:14px">{n15_tiles}</div></div>
<div class="card"><h2>Forecast vs actual, day by day</h2>
<p class="sub">Each point is the inflow expected over the next 15 days, plotted on the day the forecast is made. Orange ticks = heavy-rain days. Pick a year.</p>
<div class="seg" id="seg-n15"></div>
<div class="legend" id="l-n15"></div><div class="chart" id="c-n15"></div>
<p class="sub" style="margin-top:12px">{n15_note}</p></div>

<h2 class="sec">How the catchment responds</h2>

<div class="card"><h2>{lag_title}</h2>
<p class="sub">Extra daily inflow at the dam on each day after rain or a release (monsoon months).</p>
{lag_html}</div>

<div class="card"><h2>Each rain cell has its own multiplier and delay; {d['cells_quiet']} of {d['cells_total']} cells barely affect inflow</h2>
<p class="sub">Learned separately for each 0.25° IMD rain cell (monsoon months, 1991–2019). Multiplier = extra inflow at the dam from 1 mm of rain on that cell. Delay = average days for that cell's rain to arrive. Overall, {d['runoff_share']:.0f}% of rain on the {'free-draining ' if d['dam'] == 'krs' else ''}catchment reaches the dam.</p>
<div class="row2"><div><h3>Multiplier</h3><div class="legend" id="l-mapg"></div><div class="chart" id="c-mapg"></div></div>
<div><h3>Delay</h3><div class="legend" id="l-mapd"></div><div class="chart" id="c-mapd"></div></div></div>
<div class="tiles" style="margin-top:14px">{cmp_tiles}</div></div>

<div class="card"><h2>Daily inflow is tracked closely (NSE {d['daily_nse']:.2f}); flood peaks come out {abs(d['peaks']['peak_median_err_pct']):.0f}% low</h2>
<p class="sub">June–October of the test years 2020–2023, which the model never saw. Pick a year. <b>NSE</b> = how much of the day-to-day rise and fall the model gets right: 1 = perfect, 0 = no better than always guessing the seasonal average, above 0.75 is very good.</p>
<div class="seg" id="seg-year"></div>
<div class="legend" id="l-hydro"></div><div class="chart" id="c-hydro"></div></div>

<div class="card"><h2>{soil_title}</h2>
<p class="sub">{soil_why} Soil moisture is TerraClimate monthly data for {len(d['soil']['sub_watersheds'])} sub-watersheds ({', '.join(d['soil']['sub_watersheds'])}), {d['soil']['months'][0][:4]}–{d['soil']['months'][1][:4]}; the model only sees the last complete month. Daily accuracy (NSE) on 2020–2023, by year:</p>
<div class="legend" id="l-soil"></div><div class="chart" id="c-soil"></div>
<div class="tiles" style="margin-top:14px">{soil_tiles}</div>
<p class="sub" style="margin-top:12px">{soil_note}</p></div>

<h2 class="sec">Flood warning</h2>
<div class="card"><h2>Today's data warns of high flows {good_days} day{'s' if good_days > 1 else ''} ahead; a rain forecast stretches that to a week</h2>
<p class="sub">High-flow day = above the 95th percentile of daily inflow. Test years 2020–2023. Rain forecast = actual rainfall used as a perfect forecast (best case).</p>
<div class="legend" id="l-lead"></div>
<div class="row2"><div><div class="chart" id="c-lead1"></div></div><div><div class="chart" id="c-lead2"></div></div></div></div>
{extra}

<h2 class="sec">Comparing approaches</h2>
<div class="card"><h2>{ladder_title}</h2>
<p class="sub">Every way we tried to predict daily inflow, scored on the same unseen years (2020–2023). “Same as yesterday” needs today’s reading, so it only works one day ahead. NSE: 1 = perfect, 0 = no better than the seasonal average. Hover a row for more scores.</p>
<div class="chart" id="c-ladder"></div></div>
<div class="card"><h2>{matrix_title}</h2>
<p class="sub">Monsoon-total forecast error not exceeded in 4 of 5 years, for each approach (rows) and forecast date (columns). ✓ / bold = within ±{GOOD}%, i.e. reasonably informed. Rain-forecast rows use actual rainfall as a perfect forecast. Hover for the typical error.</p>
<div class="chart" id="c-matrix"></div></div>

{paper_html}

<h2 class="sec">How it works</h2>
<div class="card"><h2>From raw data to forecast</h2>
<p class="sub">The same steps run for both dams. Colours: grey = data, blue = preparing inputs, orange = models, green = results.</p>
{flow}</div>

<details><summary>Data fixes applied to {s} records ({len(d['fixes'])})</summary>
<table><thead><tr><th>Issue</th><th>Action</th><th class="n">Count</th></tr></thead><tbody>{fixes}</tbody></table></details>

<details><summary>How to reproduce</summary>
<p style="color:var(--ink2)">Python 3.12, packages in requirements.txt. Input data (not in the repository) goes in data/ as described in the README. Then:</p>
<pre>cd src
../.venv/bin/python prepare.py && ../.venv/bin/python inflow_model.py && ../.venv/bin/python peak_analysis.py
../.venv/bin/python dam_analysis.py && ../.venv/bin/python cell_response.py && ../.venv/bin/python soil_test.py
../.venv/bin/python event_forecast.py && ../.venv/bin/python paper_compare.py
../.venv/bin/python build_dam_reports.py</pre></details>
<p class="foot">WELL Labs Water Data Hackathon · Theme 2 / Question 3 · generated by src/build_dam_reports.py</p>
</div>"""
    d3js = (ROOT / "src" / "vendor" / "d3.v7.9.0.min.js").read_text()
    d["GOOD"] = GOOD
    payload = json.dumps(d, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{s} Monsoon Inflow Outlook</title><style>{CSS}</style>
<script>{d3js}</script><script>window.DAM = {payload};</script></head>
<body>{body}<script>{JS}</script></body></html>
"""


def main():
    R = json.loads((OUT / "dam_results.json").read_text())
    inflow = json.loads((OUT / "inflow_results.json").read_text())
    fixes = json.loads((OUT / "data_fixes.json").read_text())
    CR = json.loads((OUT / "cell_response.json").read_text())
    SOIL = json.loads((OUT / "soil_results.json").read_text())
    EV = json.loads((OUT / "event_results.json").read_text())
    EV["paper"] = json.loads((OUT / "paper_compare.json").read_text())  # Hemavathy only
    df = pd.read_parquet(ROOT / "data" / "processed" / "daily.parquet")
    for dam, meta in DAMS.items():
        html = page(dam_data(dam, R, inflow, fixes, CR, df, SOIL, EV))
        (OUT / meta["file"]).write_text(html)
        print(f"wrote outputs/{meta['file']} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
