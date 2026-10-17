/* Peak prediction page */
(function () {
  const K = window.KX;
  const P = window.REPORT_DATA.peaks;
  const M = [
    { k: 'gb', label: 'Rain + upstream', color: dash.colors[0] },
    { k: 'gb_ar', label: '+ past KRS inflow', color: dash.colors[1] },
    { k: 'persistence', label: 'Same as yesterday', color: dash.colors[2] },
  ];
  const ui = { event: null, level: '95th pct' };
  const pct = (v) => (v == null ? '–' : `${v > 0 ? '+' : ''}${v}%`);
  const sum = (label) => P.summary.find((s) => s.model === label);

  const take = () => {
    const g = sum('Rain + upstream'), y = sum('Same as yesterday');
    const el = document.getElementById('peak-take');
    el.replaceChildren();
    const b = (t) => { const e = document.createElement('b'); e.textContent = t; return e; };
    el.append(
      `We found ${g.events} peak events in 2020–2025 (independent maxima above ${K.fmt(P.thresholds.p95)} cusecs, the training-period 95th percentile). `,
      'On days when inflow jumped by at least ', `${K.fmt(P.rise_cusecs)} cusecs`, ', the rain + upstream model predicted ',
      b(`${g.median_rise_captured_pct}% of the jump`), ' on the same day (median). "Same as yesterday" predicts 0% by construction. ',
      'The weakness is size: on the peak day the model is ', b(`${Math.abs(g.median_peak_error_pct)}% low`),
      ` (median), and ${Math.round(g.within_25pct * 100)}% of peaks fall within ±25%. That is typical of models trained on log flows, which pull extremes toward the middle.`,
    );
    const tiles = document.getElementById('peak-tiles');
    const tile = (lbl, big, sub) => {
      const t = document.createElement('div'); t.className = 'kx-tile';
      const a = document.createElement('span'); a.className = 'lbl'; a.textContent = lbl;
      const c = document.createElement('span'); c.className = 'big'; c.textContent = big;
      const d = document.createElement('span'); d.className = 'sub'; d.textContent = sub;
      t.append(a, c, d); return t;
    };
    tiles.replaceChildren(
      tile('Sharp rises caught on the day', `${g.median_rise_captured_pct}%`, `median share of the jump, ${g.rise_days} rise days. Yesterday: ${y.median_rise_captured_pct}%`),
      tile('Peak-day error, median', pct(g.median_peak_error_pct), `rain + upstream. Yesterday: ${pct(y.median_peak_error_pct)}`),
      tile('Peaks within ±25%', `${Math.round(g.within_25pct * 100)}%`, `rain + upstream. + past inflow: ${Math.round(sum('+ past KRS inflow').within_25pct * 100)}%`),
      tile('Largest peaks (above 99th pct)', pct(g.big_median_peak_error_pct), `median peak-day error over ${g.big_events} events`),
    );
  };

  const zoom = () => {
    const box = document.getElementById('chart-pzoom');
    const sel = document.getElementById('pevent');
    K.legend(document.getElementById('legend-pzoom'), [{ label: 'Observed', color: 'var(--color-fg)' }, ...M]);
    const evs = P.events.slice().sort((a, b) => b.observed - a.observed);
    if (!ui.event) ui.event = evs[0].date;
    if (sel.options.length !== evs.length) sel.replaceChildren(...evs.map((e) => new Option(`${d3.utcFormat('%b %-d, %Y')(new Date(e.date))} — ${K.fmt(e.observed)} cusecs`, e.date)));
    sel.value = ui.event;
    const hy = dash.data('hydrograph').data;
    const i0 = hy.findIndex((r) => r.date === ui.event);
    const rows = hy.slice(Math.max(1, i0 - 15), i0 + 16).map((r) => {
      const prev = hy[hy.indexOf(r) - 1];
      return { t: new Date(r.date), observed: r.observed, gb: r.gb, gb_ar: r.gb_ar, persistence: prev ? prev.observed : null };
    });
    const { svg, w, h } = K.svg(box, 320);
    const series = [{ k: 'observed', label: 'Observed', color: 'var(--color-fg)' }, ...M];
    const y = d3.scaleLinear([0, d3.max(rows, (r) => d3.max(series, (s) => +r[s.k] || 0))], [0, 1]).nice();
    const m = { t: 22, r: 8, b: 26, l: K.measure(svg, y.ticks(5).map(K.fmt)) + 12 };
    const x = d3.scaleUtc(d3.extent(rows, (r) => r.t), [m.l, w - m.r]);
    y.range([h - m.b, m.t]);
    K.axes(svg, x, y, w, h, m, { yLabel: 'KRS inflow, cusecs', xAxis: d3.axisBottom(x).ticks(w < 500 ? 4 : 8).tickFormat(d3.utcFormat('%b %-d')) });
    const tPeak = new Date(ui.event);
    svg.append('line').attr('x1', x(tPeak)).attr('x2', x(tPeak)).attr('y1', m.t).attr('y2', h - m.b).attr('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '4 4');
    series.slice().reverse().forEach((s) => {
      svg.append('path').datum(rows).attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', s.k === 'observed' ? 2.5 : 2)
        .attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round')
        .attr('d', d3.line((r) => x(r.t), (r) => y(+r[s.k])).defined((r) => r[s.k] != null));
      svg.append('g').selectAll('circle').data(rows.filter((r) => r[s.k] != null)).join('circle')
        .attr('cx', (r) => x(r.t)).attr('cy', (r) => y(+r[s.k])).attr('r', 2.5).attr('fill', s.color);
    });
    const tip = K.tip(box);
    const rule = svg.append('line').attr('y1', m.t).attr('y2', h - m.b).attr('stroke', 'var(--cds-chart-reference)').attr('visibility', 'hidden');
    const bis = d3.bisector((r) => r.t).center;
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', w - m.l - m.r).attr('height', h - m.t - m.b).attr('fill', 'transparent')
      .on('pointermove', (e) => {
        const [px] = d3.pointer(e); const r = rows[bis(rows, x.invert(px))];
        rule.attr('x1', x(r.t)).attr('x2', x(r.t)).attr('visibility', 'visible');
        const [bx, by] = d3.pointer(e, box);
        tip.show(d3.utcFormat('%b %-d, %Y')(r.t), series.map((s) => ({ color: K.cssColor(s.color), label: `${s.label}: ${r[s.k] == null ? '–' : K.fmt(+r[s.k])}` })), bx, by);
      }).on('pointerleave', () => { tip.hide(); rule.attr('visibility', 'hidden'); });
  };
  document.getElementById('pevent').addEventListener('change', (e) => { ui.event = e.target.value; zoom(); });

  const scatter = () => {
    const box = document.getElementById('chart-pscatter');
    K.legend(document.getElementById('legend-pscatter'), M.map((s) => ({ ...s, dot: true })));
    const pts = P.events.flatMap((e) => M.filter((s) => e[`${s.k}_sameday`] != null).map((s) => ({ s, x: e.observed, y: e[`${s.k}_sameday`], date: e.date })));
    const { svg, w, h } = K.svg(box, 320);
    const top = d3.max(pts, (p) => Math.max(p.x, p.y));
    const x = d3.scaleLinear([0, top], [0, 1]).nice();
    const y = d3.scaleLinear(x.domain(), [0, 1]);
    const m = { t: 22, r: 12, b: 34, l: K.measure(svg, y.ticks(5).map(K.fmt)) + 12 };
    x.range([m.l, w - m.r]); y.range([h - m.b, m.t]);
    K.axes(svg, x, y, w, h, m, { yLabel: 'predicted on peak day, cusecs', xLabel: 'observed peak, cusecs', xAxis: d3.axisBottom(x).ticks(Math.max(3, Math.floor(w / 90))).tickFormat(K.fmt) });
    svg.append('line').attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(x.domain()[1])).attr('y2', y(x.domain()[1])).attr('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '4 4');
    svg.append('text').attr('x', x(x.domain()[1]) - 4).attr('y', y(x.domain()[1]) + 14).attr('text-anchor', 'end').text('perfect');
    svg.append('g').selectAll('circle').data(pts).join('circle').attr('cx', (p) => x(p.x)).attr('cy', (p) => y(p.y)).attr('r', 5)
      .attr('fill', (p) => p.s.color).attr('fill-opacity', 0.85).attr('stroke', 'var(--color-panel)').attr('stroke-width', 2);
    const tip = K.tip(box);
    const del = d3.Delaunay.from(pts, (p) => x(p.x), (p) => y(p.y));
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', w - m.l - m.r).attr('height', h - m.t - m.b).attr('fill', 'transparent')
      .on('pointermove', (e) => {
        const [px, py] = d3.pointer(e); const p = pts[del.find(px, py)];
        const [bx, by] = d3.pointer(e, box);
        tip.show(d3.utcFormat('%b %-d, %Y')(new Date(p.date)), [{ color: K.cssColor(p.s.color), label: `${p.s.label}: ${K.fmt(p.y)}` }, { label: `Observed: ${K.fmt(p.x)}` }], bx, by);
      }).on('pointerleave', () => tip.hide());
  };

  const band = () => {
    const box = document.getElementById('chart-pband');
    K.legend(document.getElementById('legend-pband'), M);
    const bands = [...new Set(P.by_band.map((r) => r.flow_band))];
    const { svg, w, h } = K.svg(box, 320);
    const ext = d3.extent(P.by_band, (r) => r.bias_pct);
    const y = d3.scaleLinear([Math.min(0, ext[0]), Math.max(0, ext[1])], [0, 1]).nice();
    const fmtP = (v) => `${v > 0 ? '+' : ''}${v}%`;
    const m = { t: 22, r: 8, b: 26, l: K.measure(svg, y.ticks(5).map(fmtP)) + 12 };
    const x0 = d3.scaleBand(bands, [m.l, w - m.r]).padding(0.25);
    const x1 = d3.scaleBand(M.map((s) => s.label), [0, x0.bandwidth()]).padding(0.12);
    y.range([h - m.b, m.t]);
    K.axes(svg, x0, y, w, h, m, { yLabel: 'volume bias, %', yFmt: fmtP, xAxis: d3.axisBottom(x0) });
    svg.append('line').attr('x1', m.l).attr('x2', w - m.r).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', 'var(--cds-chart-axis)');
    const tip = K.tip(box);
    svg.append('g').selectAll('rect').data(P.by_band).join('rect')
      .attr('x', (r) => x0(r.flow_band) + x1(r.model)).attr('width', x1.bandwidth())
      .attr('y', (r) => y(Math.max(0, r.bias_pct))).attr('height', (r) => Math.max(1, Math.abs(y(r.bias_pct) - y(0))))
      .attr('rx', Math.min(3, x1.bandwidth() / 2)).attr('fill', (r) => M.find((s) => s.label === r.model).color)
      .on('pointermove', (e, r) => {
        const [bx, by] = d3.pointer(e, box);
        tip.show(`${r.flow_band} cusecs, ${K.fmt(r.days)} days`, [{ color: K.cssColor(M.find((s) => s.label === r.model).color), label: `${r.model}: ${fmtP(r.bias_pct)} volume, median error ${r.mape_pct}%` }], bx, by);
      }).on('pointerleave', () => tip.hide());
  };

  const seg = document.getElementById('seg-level');
  seg.replaceChildren(...[['95th pct', 'Above 95th pct'], ['99th pct', 'Above 99th pct']].map(([v, l]) => {
    const b = document.createElement('button'); b.textContent = l; b.dataset.v = v;
    b.addEventListener('click', () => { ui.level = v; cont(); }); return b;
  }));
  const cont = () => {
    seg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === ui.level)));
    const rows = P.contingency.filter((r) => r.level === ui.level);
    const td = (t, c) => { const e = document.createElement('td'); if (c) e.className = c; e.textContent = t; return e; };
    document.querySelector('#table-cont tbody').replaceChildren(...rows.map((r) => {
      const tr = document.createElement('tr');
      tr.append(td(r.model), td(`${K.fmt(r.observed_days)} (≥ ${K.fmt(r.threshold)})`, 'num'), td(`${Math.round(r.POD * 100)}%`, 'num'), td(`${Math.round(r.FAR * 100)}%`, 'num'), td(r.CSI.toFixed(2), 'num'));
      return tr;
    }));
  };
  const events = () => {
    const td = (t, c) => { const e = document.createElement('td'); if (c) e.className = c; e.textContent = t; return e; };
    const cell = (v, e) => td(v == null ? '–' : `${K.fmt(v)} (${pct(e)})`, 'num');
    document.querySelector('#table-events tbody').replaceChildren(...P.events.map((e) => {
      const tr = document.createElement('tr');
      tr.append(td(d3.utcFormat('%b %-d, %Y')(new Date(e.date))), td(K.fmt(e.observed), 'num'),
        cell(e.gb_sameday, e.gb_sameday_err_pct), cell(e.gb_ar_sameday, e.gb_ar_sameday_err_pct), cell(e.persistence_sameday, e.persistence_sameday_err_pct),
        td(e.gb_lag_days == null ? '–' : `${e.gb_lag_days > 0 ? '+' : ''}${e.gb_lag_days} d`, 'num'));
      return tr;
    }));
  };
  K.pages.peaks = [take, zoom, scatter, band, cont, events];
})();
