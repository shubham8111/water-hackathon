"""Final hackathon deliverables: outputs/final_report.html (one scrolling page) and outputs/presentation.html
(slides, arrow keys to move). Both dams, three questions:
  1. How much water will the monsoon bring?   (June-September inflow, with a range of outcomes)
  2. How early can engineers know?            (error by forecast date; dry-season warning)
  3. What arrives in the next 15 days?        (15-day inflow volume)
Reads the JSON outputs of dam_analysis, cell_response, soil_test, event_forecast, esp_forecast and
paper_compare. Self-contained: d3 and data are inlined.
"""
import json

import numpy as np
import pandas as pd

from build_dam_reports import CSS, DAMS, GOOD, JS, NONE, F30, ROOT, OUT, eda, issue_label, ladder

HELPERS = JS.split("/* ---- 1. forecast error by date")[0]


def collect():
    R = json.loads((OUT / "dam_results.json").read_text())
    CR = json.loads((OUT / "cell_response.json").read_text())
    SOIL = json.loads((OUT / "soil_results.json").read_text())
    EV = json.loads((OUT / "event_results.json").read_text())
    ESP = json.loads((OUT / "esp_results.json").read_text())
    PC = json.loads((OUT / "paper_compare.json").read_text())
    df = pd.read_parquet(ROOT / "data" / "processed" / "daily.parquet")
    D = {"GOOD": GOOD, "dams": {}, "paper": None}
    for dam in ["krs", "hemavathy"]:
        mi = {issue_label(r["issue_date"]): r for r in R["monsoon_inflow"][dam]["by_issue"]}
        es = {r["issue"]: r for r in ESP[dam]["by_issue"]}
        tl = [{"issue": k, "clim": v["climatology_p80_abs_pct_err"], "clim_med": v["climatology_median_abs_pct_err"],
               "ridge": v[NONE]["p80_abs_pct_err"], "esp": es[k]["esp_p80"], "esp_med": es[k]["esp_median"],
               "f30": v[F30]["p80_abs_pct_err"], "f30_med": v[F30]["median_abs_pct_err"],
               "covers": es[k]["range_covers"], "dry_caught": es[k]["dry_caught"], "dry_false": es[k]["dry_false_alarms"],
               "dry_skill": es[k]["dry_brier_skill"], "seen": v["already_observed_pct"]} for k, v in mi.items()]
        nse = lambda o, p: round(float(1 - ((np.array(o) - np.array(p)) ** 2).sum() / ((np.array(o) - np.mean(o)) ** 2).sum()), 2)
        yrs = ESP[dam]["years"]
        for j, t in enumerate(tl):
            t["esp_nse"] = nse([yrs[y]["actual"] for y in yrs], [yrs[y]["issues"][j]["esp"] for y in yrs])
            ys = mi[t["issue"]]["years"]
            t["f30_nse"] = nse([y["actual"] for y in ys], [y[F30] for y in ys])
        rows, st = eda(dam, df, R)
        n15 = EV[dam]["next15"]["models"]
        so = SOIL[dam]["daily"]
        lad = ladder(dam, R, CR[dam]["daily_comparison"])
        first = lambda key: next((t["issue"] for t in tl if t[key] <= GOOD), None)
        D["dams"][dam] = {
            "name": DAMS[dam]["name"], "short": DAMS[dam]["short"], "timeline": tl,
            "reliable": first("esp"), "reliable_rain": first("f30"),
            "fans": ESP[dam]["years"], "eda": [{k: r[k] for k in ["year", "rain", "inflow", "pct_sep30", "dry"]} for r in rows],
            "eda_stats": st, "mean_total": R["monsoon_inflow"][dam]["mean_total_tmc"],
            "n15": {k: {x: n15[k][x] for x in ["median_pct", "p80_pct", "NSE", "within_25"]} for k in n15},
            "hz": EV[dam]["horizons"],
            "daily_nse": (R["daily"]["krs|Rain + upstream releases"] if dam == "krs" else R["daily"]["hemavathy|Rain only"])["all"]["NSE"],
            "daily_nse_ar": (R["daily"]["krs|Rain + upstream + past inflow"] if dam == "krs" else R["daily"]["hemavathy|Rain + past inflow"])["all"]["NSE"],
            "soil": [so["Without soil data"]["NSE"], so["With soil data"]["NSE"]],
            "ladder": [{"label": r["label"].replace(" (used in this report)", ""), "nse": r["nse"], "final": r["final"], "group": r["group"]} for r in lad],
            "lead": R["lead_time"][dam]}
    pm = lambda pr, k: PC["protocols"][pr]["models"][k]
    D["paper"] = {"reported": PC["paper_reported"]["nn"],
                  "rows": [{"n": n, "paper": pm("paper", k)["NSE"], "ours": pm("ours", k)["NSE"]} for n, k in [
                      ("Paper's method on our data", "Paper method: network on measured flows (1-2 days before)"),
                      ("“Same as yesterday”", "Same as yesterday"),
                      ("Our model, rain + soil", "Our model: rain + soil (no inflow readings)"),
                      ("Our model + last week's inflow", "Our model: + last week's inflow")]]}
    return D


CHART_JS = r"""
const DC = { krs: C.blue, hemavathy: C.green };
const ISS = (d) => D.dams[d].timeline.map((t) => t.issue);

/* error by forecast date, one dam */
function chartTimeline(id, dam, h = 300) {
  const T = D.dams[dam].timeline, el = document.getElementById(id), m = { t: 26, r: 12, b: 32, l: 40 }, f = frame(el, h, m);
  const x = d3.scalePoint().domain(T.map((t) => t.issue)).range([0, f.iw]).padding(0.2);
  const y = d3.scaleLinear().domain([0, 80]).range([f.ih, 0]).clamp(true);
  f.g.append('rect').attr('x', 0).attr('width', f.iw).attr('y', y(D.GOOD)).attr('height', f.ih - y(D.GOOD)).attr('fill', 'rgba(29,122,67,.08)');
  yAxis(f, y, (v) => v + '%', 4); yTitle(f, '4 years in 5 within ±…');
  f.g.append('text').attr('class', 'lbl2').attr('x', 4).attr('y', f.ih - 6).attr('fill', '#1d7a43').text(`reasonably informed (±${D.GOOD}%)`);
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).tickSizeOuter(0))
    .selectAll('text').text((d, i) => (el.clientWidth < 520 && i % 2) ? '' : d);
  const S = [{ k: 'clim', n: 'Assume an average year', c: C.ink3, dash: '5 4' }, { k: 'esp', n: 'Our forecast (no rain forecast)', c: DC[dam] },
    { k: 'f30', n: '+ 30-day rain forecast', c: C.orange, dash: '7 4' }];
  S.forEach((s) => {
    f.g.append('path').datum(T).attr('fill', 'none').attr('stroke', s.c).attr('stroke-width', s.k === 'esp' ? 3 : 2).attr('stroke-dasharray', s.dash || null)
      .attr('d', d3.line().x((t) => x(t.issue)).y((t) => y(t[s.k])));
    if (s.k !== 'clim') f.g.selectAll(null).data(T).join('circle').attr('cx', (t) => x(t.issue)).attr('cy', (t) => y(t[s.k])).attr('r', 4)
      .attr('fill', (t) => t[s.k] <= D.GOOD ? s.c : '#fff').attr('stroke', s.c).attr('stroke-width', 2);
  });
  const rel = D.dams[dam].reliable;
  if (rel) f.g.append('text').attr('class', 'lbl').attr('x', x(rel)).attr('y', y(T.find((t) => t.issue === rel).esp) - 12).attr('text-anchor', 'middle').text('✓ ' + rel);
  hover(f, x, T.map((t) => t.issue), (iss) => { const t = T.find((r) => r.issue === iss);
    return `<b>Forecast made ${iss}</b><div class="lbl2">${t.seen}% of the season already measured · NSE ${t.esp_nse.toFixed(2)}, with 30-day rain forecast ${t.f30_nse.toFixed(2)}</div>` + S.map((s) => row(s.c, s.n, `±${Math.round(t[s.k])}% (typical ±${Math.round(t[s.k + '_med'])}%)`)).join(''); });
  return S;
}
function hover(f, x, keys, html) {
  const v = f.g.append('line').attr('y1', 0).attr('y2', f.ih).attr('stroke', C.ink3).attr('opacity', 0);
  f.g.append('rect').attr('width', f.iw).attr('height', f.ih).attr('fill', 'transparent')
    .on('mousemove', (ev) => { const [mx] = d3.pointer(ev); const k = keys.reduce((a, b) => Math.abs(x(b) - mx) < Math.abs(x(a) - mx) ? b : a);
      v.attr('x1', x(k)).attr('x2', x(k)).attr('opacity', .5); showTip(ev, html(k)); })
    .on('mouseleave', () => { v.attr('opacity', 0); hideTip(); });
}

/* replay ensemble through one season */
function chartFan(id, dam, year, h = 320) {
  const F = D.dams[dam].fans[year], el = document.getElementById(id), m = { t: 26, r: 16, b: 32, l: 44 }, f = frame(el, h, m);
  const P = F.issues, x = d3.scalePoint().domain(P.map((p) => p.issue)).range([0, f.iw]).padding(0.2);
  const y = d3.scaleLinear().domain([0, d3.max([F.actual, ...P.map((p) => p.hi)]) * 1.08]).nice().range([f.ih, 0]);
  yAxis(f, y, (v) => v, 5); yTitle(f, 'June–September inflow, TMC');
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).tickSizeOuter(0))
    .selectAll('text').text((d, i) => (el.clientWidth < 520 && i % 2) ? '' : d);
  const col = DC[dam];
  f.g.append('path').datum(P).attr('fill', col).attr('opacity', .16).attr('d', d3.area().x((p) => x(p.issue)).y0((p) => y(p.lo)).y1((p) => y(p.hi)));
  f.g.append('line').attr('x1', 0).attr('x2', f.iw).attr('y1', y(F.actual)).attr('y2', y(F.actual)).attr('stroke', C.ink).attr('stroke-dasharray', '5 4');
  f.g.append('text').attr('class', 'lbl').attr('x', 4).attr('y', y(F.actual) + (y(F.actual) > f.ih - 20 ? -6 : 16)).text(`Actual ${F.actual} TMC`);
  f.g.append('path').datum(P).attr('fill', 'none').attr('stroke', col).attr('stroke-width', 3).attr('d', d3.line().x((p) => x(p.issue)).y((p) => y(p.esp)));
  f.g.selectAll(null).data(P).join('circle').attr('cx', (p) => x(p.issue)).attr('cy', (p) => y(p.esp)).attr('r', 5).attr('fill', col).attr('stroke', '#fff').attr('stroke-width', 2);
  hover(f, x, P.map((p) => p.issue), (iss) => { const p = P.find((r) => r.issue === iss);
    return `<b>${iss} ${year}</b>` + row(col, 'Forecast', `${p.esp} TMC (${p.esp > F.actual ? '+' : ''}${Math.round((p.esp - F.actual) / F.actual * 100)}%)`)
      + row(col, 'Likely range (80%)', `${Math.round(p.lo)}–${Math.round(p.hi)} TMC`) + row(C.orange, 'Chance of a dry season', Math.round(p.p_dry * 100) + '%')
      + row(C.ink, 'Actual', F.actual + ' TMC'); });
}

/* dry-season warning by date, both dams */
function chartDry(id, h = 260) {
  const el = document.getElementById(id), m = { t: 26, r: 16, b: 32, l: 40 }, f = frame(el, h, m);
  const iss = ISS('krs'), x = d3.scalePoint().domain(iss).range([0, f.iw]).padding(0.2), y = d3.scaleLinear().domain([0, 100]).range([f.ih, 0]);
  yAxis(f, y, (v) => v + '%', 4); yTitle(f, 'Dry seasons flagged in advance');
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).tickSizeOuter(0))
    .selectAll('text').text((d, i) => (el.clientWidth < 520 && i % 2) ? '' : d);
  const val = (t) => { const [a, b] = t.dry_caught.split('/').map(Number); return a / b * 100; };
  ['krs', 'hemavathy'].forEach((dam) => { const T = D.dams[dam].timeline;
    f.g.append('path').datum(T).attr('fill', 'none').attr('stroke', DC[dam]).attr('stroke-width', 2.5).attr('d', d3.line().x((t) => x(t.issue)).y((t) => y(val(t))));
    f.g.selectAll(null).data(T).join('circle').attr('cx', (t) => x(t.issue)).attr('cy', (t) => y(val(t))).attr('r', 4).attr('fill', DC[dam]).attr('stroke', '#fff').attr('stroke-width', 2); });
  hover(f, x, iss, (k) => `<b>Forecast made ${k}</b>` + ['krs', 'hemavathy'].map((dam) => { const t = D.dams[dam].timeline.find((r) => r.issue === k);
    return row(DC[dam], D.dams[dam].short, `${t.dry_caught} dry years · ${t.dry_false} false alarm${t.dry_false === 1 ? '' : 's'}`); }).join(''));
  return [{ n: 'KRS', c: DC.krs }, { n: 'Hemavathy', c: DC.hemavathy }];
}

/* monsoon inflow decides end-of-September storage */
function chartStorage(id, dam, h = 280) {
  const E = D.dams[dam].eda.filter((r) => r.pct_sep30 != null), st = D.dams[dam].eda_stats, el = document.getElementById(id);
  const m = { t: 26, r: 16, b: 40, l: 44 }, f = frame(el, h, m);
  const x = d3.scaleLinear().domain([0, d3.max(E, (r) => r.inflow) * 1.05]).nice().range([0, f.iw]), y = d3.scaleLinear().domain([0, 105]).range([f.ih, 0]);
  yAxis(f, y, (v) => v + '%', 5); yTitle(f, '% full on 30 September');
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
  f.g.append('text').attr('class', 'lbl2').attr('x', f.iw).attr('y', f.ih + 34).attr('text-anchor', 'end').text('June–September inflow, TMC');
  if (st.fill_threshold) { f.g.append('line').attr('x1', x(st.fill_threshold)).attr('x2', x(st.fill_threshold)).attr('y1', 0).attr('y2', f.ih).attr('stroke', C.ink3).attr('stroke-dasharray', '4 4');
    f.g.append('text').attr('class', 'lbl2').attr('x', x(st.fill_threshold) + 6).attr('y', y(8)).text(`above ${st.fill_threshold} TMC: ≥90% full in ${st.fill_above} years`); }
  f.g.selectAll(null).data(E).join('circle').attr('cx', (r) => x(r.inflow)).attr('cy', (r) => y(r.pct_sep30)).attr('r', 5.5)
    .attr('fill', (r) => r.dry ? C.orange : DC[dam]).attr('stroke', '#fff').attr('stroke-width', 1.5)
    .on('mousemove', (ev, r) => showTip(ev, `<b>${r.year}</b>` + row(DC[dam], 'Inflow', r.inflow + ' TMC') + row(DC[dam], 'Rain', r.rain + ' mm') + row(DC[dam], '30 Sep', r.pct_sep30 + '% full')))
    .on('mouseleave', hideTip);
}

/* next 15 days, both dams */
function chartN15(id, h = 250) {
  const el = document.getElementById(id), rows = [['Average year', 'Assume an average year'], ["Today's state, no rain forecast", "Today's state only"],
    ['+ next 7 days of rain', '+ 7-day rain forecast'], ['+ next 15 days of rain', '+ 15-day rain forecast']];
  const m = { t: 8, r: 50, b: 8, l: Math.min(170, el.clientWidth * 0.36) }, f = frame(el, h, m);
  const y0 = d3.scaleBand().domain(rows.map((r) => r[1])).range([0, f.ih]).padding(0.25);
  const y1 = d3.scaleBand().domain(['krs', 'hemavathy']).range([0, y0.bandwidth()]).padding(0.12);
  const x = d3.scaleLinear().domain([0, 65]).range([0, f.iw]);
  rows.forEach(([k, lab]) => {
    f.g.append('text').attr('class', 'lbl2').attr('x', -10).attr('y', y0(lab) + y0.bandwidth() / 2 + 4).attr('text-anchor', 'end').text(lab);
    ['krs', 'hemavathy'].forEach((dam) => { const v = D.dams[dam].n15[k];
      f.g.append('rect').attr('x', 0).attr('y', y0(lab) + y1(dam)).attr('height', y1.bandwidth()).attr('width', x(v.median_pct)).attr('rx', 3).attr('fill', DC[dam])
        .on('mousemove', (ev) => showTip(ev, `<b>${D.dams[dam].short}: ${lab}</b>` + row(DC[dam], 'Typical error', `±${Math.round(v.median_pct)}%`)
          + row(DC[dam], 'Within ±25%', `${Math.round(v.within_25 * 100)}% of days`) + row(DC[dam], 'NSE', v.NSE.toFixed(2)))).on('mouseleave', hideTip);
      f.g.append('text').attr('class', 'lbl').attr('x', x(v.median_pct) + 6).attr('y', y0(lab) + y1(dam) + y1.bandwidth() / 2 + 4).text(`±${Math.round(v.median_pct)}%`); });
  });
  return [{ n: 'KRS', c: DC.krs, cls: 'area' }, { n: 'Hemavathy', c: DC.hemavathy, cls: 'area' }];
}

/* next day / 7 days / 15 days, both dams */
const HZ = [['1', 'Next day'], ['7', 'Next 7 days'], ['15', 'Next 15 days']];
const HZROWS = [['Average year', 'Average year'], ["Today's state only", "Today's data only"], ['+ rain forecast for those days', '+ rain forecast']];
function chartHz(id, hz, h = 210) {
  const el = document.getElementById(id), m = { t: 8, r: 46, b: 8, l: Math.min(120, el.clientWidth * 0.36) }, f = frame(el, h, m);
  const y0 = d3.scaleBand().domain(HZROWS.map((r) => r[1])).range([0, f.ih]).padding(0.22);
  const y1 = d3.scaleBand().domain(['krs', 'hemavathy']).range([0, y0.bandwidth()]).padding(0.12);
  const x = d3.scaleLinear().domain([0, 82]).range([0, f.iw * 0.62]).clamp(true);
  HZROWS.forEach(([k, lab]) => {
    f.g.append('text').attr('class', 'lbl2').attr('x', -8).attr('y', y0(lab) + y0.bandwidth() / 2 + 4).attr('text-anchor', 'end').text(lab);
    ['krs', 'hemavathy'].forEach((dam) => { const H = D.dams[dam].hz[hz], v = H.models[k];
      f.g.append('rect').attr('x', 0).attr('y', y0(lab) + y1(dam)).attr('height', y1.bandwidth()).attr('width', x(v.median_pct)).attr('rx', 3).attr('fill', DC[dam])
        .on('mousemove', (ev) => showTip(ev, `<b>${D.dams[dam].short} · ${HZ.find((z) => z[0] === hz)[1].toLowerCase()} · ${lab}</b>` + row(DC[dam], 'Typical error', `±${Math.round(v.median_pct)}%`)
          + row(DC[dam], '4 days in 5 within', `±${Math.round(v.p80_pct)}%`) + row(DC[dam], 'NSE', v.NSE.toFixed(2)) + row(DC[dam], 'Average volume', `${H.mean_volume_tmc} TMC`))).on('mouseleave', hideTip);
      const t = f.g.append('text').attr('x', x(v.median_pct) + 5).attr('y', y0(lab) + y1(dam) + y1.bandwidth() / 2 + 4).attr('font-size', 12);
      t.append('tspan').attr('font-weight', 700).attr('fill', C.ink).text(`NSE ${v.NSE.toFixed(2)}`);
      t.append('tspan').attr('fill', C.ink3).text(`  ±${Math.round(v.median_pct)}%`); });
  });
}
function chartHzAll(prefix, h) { HZ.forEach(([k]) => chartHz(`${prefix}-${k}`, k, h));
  return [{ n: 'KRS', c: DC.krs, cls: 'area' }, { n: 'Hemavathy', c: DC.hemavathy, cls: 'area' }]; }

/* what each step added to daily accuracy */
function chartLadderMini(id, dam, h) {
  const L = D.dams[dam].ladder, el = document.getElementById(id), m = { t: 4, r: 44, b: 6, l: Math.min(240, el.clientWidth * 0.55) };
  const f = frame(el, h || L.length * 28 + 10, m);
  const y = d3.scaleBand().domain(L.map((r) => r.label)).range([0, f.ih]).padding(0.28), x = d3.scaleLinear().domain([0, 1]).range([0, f.iw]);
  L.forEach((r) => { f.g.append('rect').attr('x', 0).attr('y', y(r.label)).attr('height', y.bandwidth()).attr('width', x(Math.max(0, r.nse))).attr('rx', 3)
      .attr('fill', r.final ? DC[dam] : '#c3ccd5');
    f.g.append('text').attr('class', r.final ? 'lbl' : 'lbl2').attr('x', -8).attr('y', y(r.label) + y.bandwidth() / 2 + 4).attr('text-anchor', 'end').text(r.label);
    f.g.append('text').attr('class', r.final ? 'lbl' : 'lbl2').attr('x', x(Math.max(0, r.nse)) + 6).attr('y', y(r.label) + y.bandwidth() / 2 + 4).text(r.nse.toFixed(2)); });
}

/* vs the published Hemavathi model */
function chartPaper(id, h = 220) {
  const P = D.paper, el = document.getElementById(id), m = { t: 26, r: 30, b: 28, l: Math.min(230, el.clientWidth * 0.45) };
  const f = frame(el, h, m), y = d3.scaleBand().domain(P.rows.map((r) => r.n)).range([0, f.ih]).padding(0.4), x = d3.scaleLinear().domain([0.6, 1]).range([0, f.iw]);
  f.g.append('g').attr('class', 'grid').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(4).tickSize(-f.ih).tickFormat(''));
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.ih})`).call(d3.axisBottom(x).ticks(4).tickFormat(d3.format('.1f')).tickSizeOuter(0));
  f.g.append('line').attr('x1', x(P.reported.NSE)).attr('x2', x(P.reported.NSE)).attr('y1', -8).attr('y2', f.ih).attr('stroke', C.ink).attr('stroke-dasharray', '4 4');
  f.g.append('text').attr('class', 'lbl2').attr('x', x(P.reported.NSE)).attr('y', -12).attr('text-anchor', 'middle').text(`paper reports ${P.reported.NSE.toFixed(2)}`);
  P.rows.forEach((r) => { const cy = y(r.n) + y.bandwidth() / 2;
    f.g.append('text').attr('class', r.n.startsWith('Our') ? 'lbl' : 'lbl2').attr('x', -10).attr('y', cy + 4).attr('text-anchor', 'end').text(r.n);
    f.g.append('line').attr('x1', x(Math.max(.6, r.paper))).attr('x2', x(Math.max(.6, r.ours))).attr('y1', cy).attr('y2', cy).attr('stroke', '#c3ccd5').attr('stroke-width', 3);
    [['paper', C.orange, 'Random test days (paper’s test)'], ['ours', C.blue, 'Unseen years 2020–2023 (our test)']].forEach(([k, c, lab]) =>
      f.g.append('circle').attr('cx', x(Math.max(.6, r[k]))).attr('cy', cy).attr('r', 7).attr('fill', c).attr('stroke', '#fff').attr('stroke-width', 2)
        .on('mousemove', (ev) => showTip(ev, `<b>${r.n}</b>` + row(c, lab, 'NSE ' + r[k].toFixed(2)))).on('mouseleave', hideTip)); });
  return [{ n: 'Random test days (paper’s test)', c: C.orange, cls: 'dot' }, { n: 'Unseen years 2020–2023 (our test)', c: C.blue, cls: 'dot' }];
}

/* controls */
function segButtons(id, items, sel, onpick) {
  const el = document.getElementById(id); if (!el) return; el.innerHTML = '';
  items.forEach(([k, lab]) => { const b = document.createElement('button'); b.textContent = lab; b.setAttribute('aria-pressed', k === sel);
    b.onclick = () => { el.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b)); onpick(k); }; el.appendChild(b); });
}
const fanState = { dam: 'krs', year: null };
function fanDefaultYear(dam) { const F = D.dams[dam].fans; return Object.keys(F).reduce((a, b) => F[b].actual < F[a].actual ? b : a); }
function setupFan(prefix) {
  fanState.year = fanState.year || fanDefaultYear(fanState.dam);
  const sel = document.getElementById(prefix + '-year'), fill = () => { const yrs = Object.keys(D.dams[fanState.dam].fans);
    sel.innerHTML = yrs.map((y) => `<option value="${y}" ${y === String(fanState.year) ? 'selected' : ''}>${y} · ${D.dams[fanState.dam].fans[y].actual} TMC</option>`).join(''); };
  fill();
  sel.onchange = () => { fanState.year = sel.value; chartFan(prefix + '-chart', fanState.dam, fanState.year); };
  segButtons(prefix + '-dam', [['krs', 'KRS'], ['hemavathy', 'Hemavathy']], fanState.dam, (k) => { fanState.dam = k; fill(); chartFan(prefix + '-chart', k, fanState.year); });
}
"""

FINAL_CSS = """
.q { display:inline-block; background:var(--ink); color:#fff; border-radius:999px; padding:3px 12px; font-size:.78rem; letter-spacing:.04em; margin-bottom:8px; }
.answer { font-size:1.35rem; font-weight:650; line-height:1.3; margin:0 0 6px; }
.row2c { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
@media (max-width:760px) { .row2c { grid-template-columns:1fr; } }
.mini h3 { font-size:.95rem; margin:0 0 4px; }
.links a { color:var(--blue); }
.err { color:var(--ink3); font-weight:400; } .nse { color:var(--ink); font-weight:700; }
.tile .v .err { font-size:.6em; }
.nset { font-size:.85rem; } .nset th, .nset td { white-space:nowrap; }
"""


def hz_text(D):
    """Short answers for the next day / 7 days / 15 days (typical error and NSE, KRS · Hemavathy)."""
    g = lambda dam, h, m, k="median_pct": D["dams"][dam]["hz"][str(h)]["models"][m][k]
    st, rf = "Today's state only", "+ rain forecast for those days"
    pair = lambda h, m: f"<span class='err'>±{g('krs', h, m):.0f}% · ±{g('hemavathy', h, m):.0f}%</span>"
    npair = lambda h, m: f"<b class='nse'>NSE {g('krs', h, m, 'NSE'):.2f} · {g('hemavathy', h, m, 'NSE'):.2f}</b>"
    return {"day": pair(1, st), "d7": pair(7, rf), "d15": pair(15, rf), "d7_none": pair(7, st), "d15_none": pair(15, st),
            "n_day": npair(1, st), "n_d7": npair(7, rf), "n_d15": npair(15, rf), "n_d7_none": npair(7, st), "n_d15_none": npair(15, st)}


def nse_pair(a, b):
    return f"<b class='nse'>NSE {a:.2f} · {b:.2f}</b>"


def err(t):
    return f"<span class='err'>{t}</span>"


def nse_table(D):
    """Monsoon-total NSE by forecast date (across years), both dams, with and without a 30-day rain forecast."""
    T = {d: D["dams"][d]["timeline"] for d in ["krs", "hemavathy"]}
    head = "".join(f"<th class='n'>{t['issue']}</th>" for t in T["krs"])
    cell = lambda v: f"<td class='n'{' style=\"font-weight:650;color:#1d7a43\"' if v >= 0.9 else ''}>{v:.2f}</td>"
    rows = "".join(f"<tr><td>{lab}</td>{''.join(cell(t[key]) for t in T[d])}</tr>" for d, key, lab in [
        ("krs", "esp_nse", "KRS"), ("krs", "f30_nse", "KRS + 30-day rain forecast"),
        ("hemavathy", "esp_nse", "Hemavathy"), ("hemavathy", "f30_nse", "Hemavathy + 30-day rain forecast")])
    return f"<table class='nset'><thead><tr><th>NSE of the monsoon total, forecast made on</th>{head}</tr></thead><tbody>{rows}</tbody></table>"


HZ_HTML = """<div class="row3">
<div class="mini"><h3>Next day</h3><div class="chart" id="{p}-1"></div></div>
<div class="mini"><h3>Next 7 days</h3><div class="chart" id="{p}-7"></div></div>
<div class="mini"><h3>Next 15 days</h3><div class="chart" id="{p}-15"></div></div></div>"""


def final_page(D):
    k, h = D["dams"]["krs"], D["dams"]["hemavathy"]
    at = lambda d, iss: next(t for t in d["timeline"] if t["issue"] == iss)
    tile = lambda kk, v, s: f'<div class="tile"><div class="k">{kk}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
    dryk, dryh = at(k, "15 Aug")["dry_caught"], at(h, "15 Aug")["dry_caught"]
    n15k, n15h = k["n15"]["+ next 15 days of rain"], h["n15"]["+ next 15 days of rain"]
    hz = hz_text(D)
    tiles = "".join([
        tile("Reliable monsoon-total estimate", k["reliable"], f"{nse_pair(at(k, k['reliable'])['esp_nse'], at(h, h['reliable'])['esp_nse'])}, KRS · Hemavathy " + err(f"(4 years in 5 within ±{GOOD}%)") + f" · {k['reliable_rain']} with a 30-day rain forecast"),
        tile("Dry season flagged by 15 Aug", f"{dryk} · {dryh}", "dry years caught at KRS · Hemavathy"),
        tile("Water arriving tomorrow", hz["n_day"], f"KRS · Hemavathy, from today's data ({hz['day']}) · 7 days {hz['n_d7']} ({hz['d7']}), 15 days {hz['n_d15']} ({hz['d15']}) with a rain forecast"),
        tile("Daily inflow accuracy (NSE)", f"{k['daily_nse']:.2f} · {h['daily_nse']:.2f}", "on unseen years 2020–2023; 1 = perfect")])
    ka, ha = at(k, "1 Aug"), at(h, "1 Aug")
    st = k["eda_stats"]
    body = f"""
<div class="wrap">
<header>
  <h1>How much water will the monsoon bring, and how early can we know?</h1>
  <p>WELL Labs Water Data Hackathon · Theme 2, Question 3 · Krishnaraja Sagar (KRS) and Hemavathy dams, Cauvery basin.
  IMD daily gridded rainfall, daily dam records 1991–2025, TerraClimate soil moisture. Every score is on years the model never saw.</p>
</header>
<div class="tiles">{tiles}</div>

<h2 class="sec">Question 1 · How much water will the monsoon bring?</h2>
<div class="card"><span class="q">ANSWER</span>
<p class="answer">The June–September inflow decides how full the dams end the monsoon, and we forecast it as a number with a likely range.</p>
<p class="sub">KRS ends September at least 90% full in {st['fill_above']} years when monsoon inflow is above {st['fill_threshold']} TMC, and only {st['fill_below']} below it. So forecasting the inflow answers the storage question.</p>
<div class="row2c"><div class="mini"><h3>KRS: monsoon inflow vs storage on 30 September</h3><div class="chart" id="f-stor-k"></div></div>
<div class="mini"><h3>Hemavathy: the same</h3><div class="chart" id="f-stor-h"></div></div></div>
<div class="legend"><span><i class="dot" style="background:#eb6834"></i>Driest third of years</span><span><i class="dot" style="background:#2a78d6"></i>KRS, other years</span><span><i class="dot" style="background:#1baf7a"></i>Hemavathy, other years</span></div></div>

<div class="card"><h2>The forecast through one season: every past monsoon's remaining rain replayed on this year's catchment</h2>
<p class="sub">On each forecast date the inflow already measured and today's state are fixed, and the rest of the season is replayed with the rain of each of the other {len(k['fans']) - 1} years. Line = middle of the replays; band = 80% likely range; dashed = what actually came. Pick a dam and a year; hover for the chance of a dry season.</p>
<div class="seg" id="ffan-dam"></div> <select class="yr" id="ffan-year" aria-label="Year"></select>
<div class="chart" id="ffan-chart"></div></div>

<h2 class="sec">Question 2 · How early can engineers make a reasonably informed estimate?</h2>
<div class="card"><span class="q">ANSWER</span>
<p class="answer">From {k['reliable']} at both dams ({nse_pair(at(k, k['reliable'])['esp_nse'], at(h, h['reliable'])['esp_nse'])}; {err(f"4 years in 5 within ±{GOOD}%")}), or from {k['reliable_rain']} with a 30-day rain forecast ({nse_pair(at(k, k['reliable_rain'])['f30_nse'], at(h, h['reliable_rain'])['f30_nse'])}). Before July, NSE is about 0: no better than assuming an average year.</p>
<p class="sub">On 1 August the typical error is ±{ka['esp_med']:.0f}% at KRS and ±{ha['esp_med']:.0f}% at Hemavathy. A median of about half the season's inflow is still to come then, so the rain of August decides the outcome.</p>
<div class="legend" id="f-tl-leg"></div>
<div class="row2c"><div class="mini"><h3>KRS</h3><div class="chart" id="f-tl-k"></div></div><div class="mini"><h3>Hemavathy</h3><div class="chart" id="f-tl-h"></div></div></div>
<p class="sub" style="margin-top:14px">NSE of the forecast total across all {len(k['fans'])} years (1 = perfect, 0 = no better than assuming an average year; 0.90 and above in green):</p>
<div style="overflow-x:auto">{nse_table(D)}</div></div>
<div class="card"><h2>A dry season can be called by mid-August: {dryk} dry years at KRS and {dryh} at Hemavathy</h2>
<p class="sub">Dry season = the driest third of years. Flagged when at least half of the replays end below that line. Hover for false alarms.</p>
<div class="legend" id="f-dry-leg"></div><div class="chart" id="f-dry"></div></div>

<h2 class="sec">Question 3 · How much water will reach the dam in the next day, 7 days and 15 days?</h2>
<div class="card"><span class="q">ANSWER</span>
<p class="answer">Next day: {hz['n_day']} from today's data alone ({hz['day']}). Next 7 days: {hz['n_d7']} ({hz['d7']}), and next 15 days: {hz['n_d15']} ({hz['d15']}), with a rain forecast for those days.</p>
<p class="sub">NSE in bold, typical error in grey; KRS · Hemavathy. Total inflow over the period after each forecast day, June–September of 2020–2023, never seen in training. Tomorrow's inflow is mostly water already on its way, so today's data is enough; beyond a few days the coming rain decides it (without a rain forecast: {hz['n_d7_none']} ({hz['d7_none']}) for 7 days; {hz['n_d15_none']} ({hz['d15_none']}) for 15 days). The rain forecast is the rain that actually fell, so those are best-case numbers. Bar length = typical error (shorter is better); hover for volumes.</p>
<div class="legend" id="f-hz-leg"></div>{HZ_HTML.format(p="f-hz")}</div>

<h2 class="sec">How we got there</h2>
<div class="card"><h2>Each step that improved daily accuracy</h2>
<p class="sub">NSE on unseen years 2020–2023 (1 = perfect, 0 = no better than the seasonal average). Coloured = the model behind the daily results. Soil moisture: KRS {k['soil'][0]:.2f} → {k['soil'][1]:.2f}, Hemavathy {h['soil'][0]:.2f} → {h['soil'][1]:.2f}. Hemavathy gains more because its inflow comes straight from rain on its own catchment; KRS already gets most of its information from the upstream releases.</p>
<div class="row2c"><div class="mini"><h3>KRS</h3><div class="chart" id="f-lad-k"></div></div><div class="mini"><h3>Hemavathy</h3><div class="chart" id="f-lad-h"></div></div></div></div>
<div class="card"><h2>Scored the same way, our Hemavathy model beats the published one (Chandre Gowda, 2025)</h2>
<p class="sub">The paper reports NSE {D['paper']['reported']['NSE']:.2f} with test days picked at random. We rebuilt its method on our data and scored everything both ways. Random test days flatter every model; on unseen years ours holds up.</p>
<div class="legend" id="f-pap-leg"></div><div class="chart" id="f-pap"></div></div>
<div class="card"><h2>What did not help, and the limits</h2>
<ul class="sub" style="margin:0;padding-left:18px;line-height:1.6">
<li>Distance along the river network did not beat straight-line distance; an extra forecast on a heavy-rain day helps no more than one on any other day.</li>
<li>Soil moisture improves the daily model but not the monsoon total. It is published with a long delay, so real-time use needs a near-real-time product.</li>
<li>Rain-forecast results use the rain that actually fell, so they are an upper bound. Releases from upstream dams are operator decisions, taken as known.</li>
<li>Without inflow readings the Hemavathy daily model over-predicts unseen years; with last week's inflow it is accurate (NSE {h['daily_nse_ar']:.2f}).</li></ul></div>
<p class="foot links">Detailed reports: <a href="KRS_insights.html">KRS</a> · <a href="Hemavathy_insights.html">Hemavathy</a> · <a href="presentation.html">presentation</a> · generated by src/build_final.py</p>
</div>"""
    script = r"""
function drawAll() {
  chartStorage('f-stor-k', 'krs'); chartStorage('f-stor-h', 'hemavathy');
  chartFan('ffan-chart', fanState.dam, fanState.year);
  const S = chartTimeline('f-tl-k', 'krs'); chartTimeline('f-tl-h', 'hemavathy');
  legend(document.getElementById('f-tl-leg'), [{ n: 'Assume an average year', c: C.ink3, cls: 'dash' }, { n: 'Our forecast, KRS', c: DC.krs }, { n: 'Our forecast, Hemavathy', c: DC.hemavathy }, { n: '+ 30-day rain forecast', c: C.orange }, { n: `±${D.GOOD}% band`, c: 'rgba(29,122,67,.25)', cls: 'area' }]);
  legend(document.getElementById('f-dry-leg'), chartDry('f-dry'));
  legend(document.getElementById('f-hz-leg'), chartHzAll('f-hz'));
  chartLadderMini('f-lad-k', 'krs'); chartLadderMini('f-lad-h', 'hemavathy');
  legend(document.getElementById('f-pap-leg'), chartPaper('f-pap'));
}
setupFan('ffan'); drawAll();
let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(drawAll, 150); });
"""
    return doc("Monsoon Inflow Outlook", CSS + FINAL_CSS, body, D, script)


SLIDE_CSS = """
html, body { height:100%; overflow:hidden; }
.deck { height:100vh; position:relative; }
.slide { position:absolute; inset:0; padding:4vh 5vw 9vh; display:none; flex-direction:column; }
.slide.on { display:flex; }
.slide h1 { font-size:clamp(1.8rem,3.6vw,3rem); line-height:1.15; margin:0 0 14px; letter-spacing:-.01em; }
.slide h2 { font-size:clamp(1.3rem,2.4vw,2rem); line-height:1.2; margin:0 0 6px; }
.slide .lede { font-size:clamp(1rem,1.5vw,1.25rem); color:var(--ink2); margin:0 0 14px; max-width:75ch; }
.slide .body { flex:1; min-height:0; display:flex; flex-direction:column; justify-content:center; }
.slide .tiles { margin:10px 0; }
.slide .tile .v { font-size:clamp(1.5rem,2.6vw,2.4rem); }
.big3 { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
.big3 .tile .v { font-size:clamp(1.3rem,2.2vw,2rem); }
.hzt td.nn { font-size:.85em; }
.hzt { width:auto; margin:6px 0 4px; font-size:clamp(.95rem,1.25vw,1.2rem); } .hzt td { border:0; padding:3px 12px 3px 0; white-space:nowrap; } .hzt td + td { font-variant-numeric:tabular-nums; }
.slide .tile .s { font-size:clamp(.9rem,1.2vw,1.1rem); line-height:1.45; } .slide .tile { padding:22px 24px; }
.flowline { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }
.flowline .node { font-size:.95rem; }
.nav { position:fixed; bottom:2.5vh; left:5vw; right:5vw; display:flex; align-items:center; gap:12px; color:var(--ink3); font-size:.85rem; }
.nav button { font:inherit; border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:6px 14px; cursor:pointer; color:var(--ink); }
.nav .bar { flex:1; height:4px; background:var(--grid); border-radius:2px; overflow:hidden; } .nav .bar i { display:block; height:100%; background:var(--ink); }
.kicker { font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; color:var(--ink3); margin-bottom:8px; }
@media (max-width:760px) { .big3, .flowline { grid-template-columns:1fr; } .slide { overflow-y:auto; } }
"""


def presentation(D):
    k, h = D["dams"]["krs"], D["dams"]["hemavathy"]
    at = lambda d, iss: next(t for t in d["timeline"] if t["issue"] == iss)
    hz = hz_text(D)
    st = k["eda_stats"]
    tile = lambda kk, v, s: f'<div class="tile"><div class="k">{kk}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
    node = lambda cls, t, b: f'<div class="node {cls}"><b>{t}</b>{b}</div>'
    slides = [
        ("", f"""<div class="body"><div class="kicker">WELL Labs Water Data Hackathon · Theme 2 · Question 3</div>
<h1>How much water will the monsoon bring,<br>and how early can we know?</h1>
<p class="lede">Forecasting June–September inflow to Krishnaraja Sagar (KRS) and Hemavathy dams, Cauvery basin, from rainfall, dam records and soil moisture.</p>
<p class="lede" style="font-size:1rem">Use ← → to move between slides.</p></div>"""),
        ("", f"""<div class="kicker">The answers</div><h2>Three questions, three answers</h2><div class="body"><div class="big3">
{tile("1 · How much water?", "Inflow decides storage", f"KRS ends September ≥90% full in {st['fill_above']} years above {st['fill_threshold']} TMC of monsoon inflow. We forecast that inflow with a likely range.")}
{tile("2 · How early?", k['reliable'], f"{nse_pair(at(k, k['reliable'])['esp_nse'], at(h, h['reliable'])['esp_nse'])} at both dams " + err(f"(4 years in 5 within ±{GOOD}%)") + f"; {k['reliable_rain']} with a 30-day rain forecast ({nse_pair(at(k, k['reliable_rain'])['f30_nse'], at(h, h['reliable_rain'])['f30_nse'])}). A dry season is flagged by mid-August.")}
<div class="tile"><div class="k">3 · Water in the next day, 7 days, 15 days?</div>
<table class="hzt"><tr><td>Next day</td><td>{hz['n_day']}</td><td class="nn">{hz['day']}</td></tr><tr><td>Next 7 days</td><td>{hz['n_d7']}</td><td class="nn">{hz['d7']}</td></tr><tr><td>Next 15 days</td><td>{hz['n_d15']}</td><td class="nn">{hz['d15']}</td></tr></table>
<div class="s">NSE in bold, typical error in grey; KRS · Hemavathy. Next day from today's data, 7 and 15 days with a rain forecast</div></div>
</div></div>"""),
        ("", f"""<div class="kicker">Approach</div><h2>From raw data to a forecast an engineer can use</h2><p class="lede">Every score is on years the model never saw.</p><div class="body"><div class="flowline">
{node("n-data", "Data", "IMD 0.25° daily rain, dam records for KRS, Harangi and Hemavathy (1991–2025), catchment and river maps, TerraClimate soil moisture.")}
{node("n-prep", "Clean", "Missing days, storage spikes, zero monsoon inflow, Jul–Sep 2019 outflow in the wrong units: all fixed and logged.")}
{node("n-prep", "Inputs", "Rain by distance band and per grid cell (each with its own delay and multiplier), soil wetness, upstream releases.")}
{node("n-model", "Models", "Daily inflow (gradient boosting); monsoon total (ridge + replay of past monsoons); next 15 days.")}
{node("n-out", "Fair test", "Daily: trained 1991–2019, tested 2020–2023. Season: each year forecast by a model that never saw it.")}</div></div>"""),
        ("stor", f"""<div class="kicker">Question 1 · How much water?</div><h2>Monsoon inflow decides how full the dams end September</h2>
<p class="lede">Above {st['fill_threshold']} TMC of June–September inflow, KRS ends September at least 90% full in {st['fill_above']} years; below it, in {st['fill_below']}. Orange = driest third of years.</p>
<div class="body"><div class="row2c"><div class="mini"><h3>KRS</h3><div class="chart" id="p-stor-k"></div></div><div class="mini"><h3>Hemavathy</h3><div class="chart" id="p-stor-h"></div></div></div></div>"""),
        ("fan", f"""<div class="kicker">Question 1 · How much water?</div><h2>Replaying every past monsoon gives a forecast and a likely range</h2>
<p class="lede">On each date the inflow so far is fixed and the rest of the season is replayed with every other year's rain. Band = 80% likely range; dashed = what actually came.</p>
<div><div class="seg" id="pfan-dam"></div> <select class="yr" id="pfan-year" aria-label="Year"></select></div><div class="body"><div class="chart" id="pfan-chart"></div></div>"""),
        ("tl", f"""<div class="kicker">Question 2 · How early?</div><h2>Reasonably informed from {k['reliable']}; from {k['reliable_rain']} with a 30-day rain forecast</h2>
<p class="lede">Error not exceeded in 4 years out of 5, by the date the forecast is made. Filled dots = within ±{GOOD}%. On {k['reliable']}: {nse_pair(at(k, k['reliable'])['esp_nse'], at(h, h['reliable'])['esp_nse'])} (KRS · Hemavathy); about 0 before July.</p>
<div class="legend" id="p-tl-leg"></div><div class="body"><div class="row2c"><div class="mini"><h3>KRS</h3><div class="chart" id="p-tl-k"></div></div><div class="mini"><h3>Hemavathy</h3><div class="chart" id="p-tl-h"></div></div></div></div>"""),
        ("dry", f"""<div class="kicker">Question 2 · How early?</div><h2>A dry season can be called by mid-August</h2>
<p class="lede">Share of the driest third of years flagged in advance, by forecast date. By 15 August: {at(k, '15 Aug')['dry_caught']} at KRS and {at(h, '15 Aug')['dry_caught']} at Hemavathy.</p>
<div class="legend" id="p-dry-leg"></div><div class="body"><div class="chart" id="p-dry"></div></div>"""),
        ("n15", f"""<div class="kicker">Question 3 · Water in the next day, 7 days, 15 days</div><h2>Tomorrow from today's data; a week or two ahead needs a rain forecast</h2>
<p class="lede">KRS · Hemavathy, test years 2020–2023, NSE in bold, typical error in grey. Next day {hz['n_day']} {hz['day']}; next 7 days {hz['n_d7']} {hz['d7']}; next 15 days {hz['n_d15']} {hz['d15']} with a rain forecast.</p>
<div class="legend" id="p-hz-leg"></div><div class="body">{HZ_HTML.format(p="p-hz")}</div>"""),
        ("lad", f"""<div class="kicker">What improved performance</div><h2>Daily inflow: NSE {k['daily_nse']:.2f} at KRS, {h['daily_nse']:.2f} at Hemavathy on unseen years</h2>
<p class="lede">Each grid cell's own delay and multiplier, and soil moisture, lift accuracy; Hemavathy gains most from soil because its inflow comes straight from rain on its own catchment.</p>
<div class="body"><div class="row2c"><div class="mini"><h3>KRS</h3><div class="chart" id="p-lad-k"></div></div><div class="mini"><h3>Hemavathy</h3><div class="chart" id="p-lad-h"></div></div></div></div>"""),
        ("pap", f"""<div class="kicker">Against published work</div><h2>Scored the same way, we beat the published Hemavathi model</h2>
<p class="lede">Chandre Gowda (2025), Water-Energy Nexus 8, reports NSE {D['paper']['reported']['NSE']:.2f} on random test days. We rebuilt its method and scored all models both ways.</p>
<div class="legend" id="p-pap-leg"></div><div class="body"><div class="chart" id="p-pap"></div></div>"""),
        ("", f"""<div class="kicker">Limits and next steps</div><h2>What engineers should know</h2><div class="body"><div class="big3">
{tile("Rain forecasts", "Best case", "Rain-forecast results use the rain that actually fell. Next: plug in IMD extended-range forecasts.")}
{tile("Data delays", "Soil moisture", "Helps the daily model, not the monsoon total; needs a near-real-time product to use live.")}
{tile("Operations", "Releases", "Upstream releases are operator decisions, taken as known. Next: link the forecast to release rules.")}
</div></div>"""),
        ("", """<div class="body"><div class="kicker">Thank you</div><h1>Questions?</h1>
<p class="lede">Detailed reports for each dam, all code and the method: <span class="links"><a href="final_report.html">final report</a> · <a href="KRS_insights.html">KRS</a> · <a href="Hemavathy_insights.html">Hemavathy</a></span></p></div>"""),
    ]
    html = '<div class="deck">' + "".join(f'<section class="slide" data-k="{key}">{content}</section>' for key, content in slides) + "</div>"
    html += '<div class="nav"><button id="prev" aria-label="Previous slide">←</button><span id="count"></span><div class="bar"><i id="prog"></i></div><button id="next" aria-label="Next slide">→</button></div>'
    script = r"""
const slides = [...document.querySelectorAll('.slide')]; let cur = 0;
const H = () => Math.max(220, Math.min(460, window.innerHeight * 0.5));
const draw = {
  stor: () => { chartStorage('p-stor-k', 'krs', H()); chartStorage('p-stor-h', 'hemavathy', H()); },
  fan: () => chartFan('pfan-chart', fanState.dam, fanState.year, H()),
  tl: () => { chartTimeline('p-tl-k', 'krs', H()); chartTimeline('p-tl-h', 'hemavathy', H());
    legend(document.getElementById('p-tl-leg'), [{ n: 'Assume an average year', c: C.ink3, cls: 'dash' }, { n: 'Our forecast, KRS', c: DC.krs }, { n: 'Our forecast, Hemavathy', c: DC.hemavathy }, { n: '+ 30-day rain forecast', c: C.orange }]); },
  dry: () => legend(document.getElementById('p-dry-leg'), chartDry('p-dry', H())),
  n15: () => legend(document.getElementById('p-hz-leg'), chartHzAll('p-hz', H() * 0.8)),
  lad: () => { chartLadderMini('p-lad-k', 'krs'); chartLadderMini('p-lad-h', 'hemavathy'); },
  pap: () => legend(document.getElementById('p-pap-leg'), chartPaper('p-pap', H() * 0.7)),
};
function show(i) {
  cur = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((s, j) => s.classList.toggle('on', j === cur));
  document.getElementById('count').textContent = `${cur + 1} / ${slides.length}`;
  document.getElementById('prog').style.width = `${(cur + 1) / slides.length * 100}%`;
  const k = slides[cur].dataset.k; if (draw[k]) draw[k]();
  try { history.replaceState(null, '', '#' + (cur + 1)); } catch (e) {}
}
document.addEventListener('keydown', (e) => { if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); show(cur + 1); }
  if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); show(cur - 1); } });
document.getElementById('next').onclick = () => show(cur + 1); document.getElementById('prev').onclick = () => show(cur - 1);
setupFan('pfan');
const _cf = chartFan; chartFan = (id, dam, y) => _cf(id, dam, y, H());
show((parseInt(location.hash.slice(1)) || 1) - 1);
let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => show(cur), 150); });
"""
    return doc("Monsoon Inflow Presentation", CSS + FINAL_CSS + SLIDE_CSS, html, D, script)


def doc(title, css, body, D, script):
    d3js = (ROOT / "src" / "vendor" / "d3.v7.9.0.min.js").read_text()
    payload = json.dumps(D, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{css}</style>
<script>{d3js}</script><script>window.DAM = {payload};</script></head>
<body>{body}<script>{HELPERS}{CHART_JS}{script}</script></body></html>
"""


def main():
    D = collect()
    for name, html in [("final_report.html", final_page(D)), ("presentation.html", presentation(D))]:
        (OUT / name).write_text(html)
        print(f"wrote outputs/{name} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
