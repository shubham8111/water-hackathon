"""Build the self-contained light-theme report: outputs/KRS_inflow_report.html.

Reuses the dashboard page (src/report/page.html) with a tiny local stand-in for its data API,
adds the peak-prediction and approach tabs, and inlines d3 + all data so it opens offline.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "report"
DASH = ROOT / "outputs" / "dash"
OUT = ROOT / "outputs" / "KRS_inflow_report.html"
DATASETS = ["fixes", "annual", "monthly", "cells", "watershed", "inflow_metrics", "hydrograph",
            "lag_response", "importance", "season_skill", "season_years"]

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KRS Inflow & Storage Forecast</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono&display=swap">
<style>
/* Light, paper-and-river report: cool off-white ground, white panels, Cauvery blue accent */
:root {
  color-scheme: light;
  --font-anthropic-sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-anthropic-serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
  --color-bg: #f5f7f8;
  --color-panel: #ffffff;
  --color-fg: #14202b;
  --color-fg-muted: #56636e;
  --color-border-line: #dde3e8;
  --color-ok: #1d7a43;
  --color-warn: #9a6400;
  --color-bad: #c43c3b;
  --cds-text-secondary: #56636e;
  --cds-font-size-caption: 0.8125rem;
  --cds-font-size-body: 0.9375rem;
  --cds-font-size-heading: 1.0625rem;
  --cds-font-size-title: 1.875rem;
  --cds-font-weight-medium: 600;
  --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 16px; --cds-gap-lg: 24px; --cds-gap-xl: 40px;
  --cds-pad-xs: 4px; --cds-pad-sm: 8px; --cds-pad-md: 12px; --cds-pad-lg: 20px; --cds-pad-xl: 32px;
  --cds-radius: 8px;
  --cds-chart-grid: #edf0f2;
  --cds-chart-axis: #b9c2c9;
  --cds-chart-reference: #8a959e;
  --cds-chart-muted: #b4bec6;
  --cds-chart-status-critical: #e34948;
  --cds-surface-popover: #ffffff;
  --cds-border: #d3dae0;
  --cds-shadow-popover: 0 4px 16px rgba(20, 32, 43, 0.12);
  --cds-dur-slow: 240ms;
}
html { background: var(--color-bg); }
body { margin: 0; background: var(--color-bg); color: var(--color-fg); font-family: var(--font-anthropic-sans); font-size: var(--cds-font-size-body); line-height: 1.5; }
.kx-wrap { max-width: 1240px; margin-inline: auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.dash-skeleton { }
[hidden] { display: none !important; }
.kx-doc { background: var(--color-panel); border: 1px solid var(--color-border-line); border-radius: var(--cds-radius); padding: var(--cds-pad-xl); max-width: 80ch; }
.kx-doc h2 { font-family: var(--font-anthropic-serif); font-weight: 600; font-size: 1.375rem; margin: 2rem 0 0.5rem; text-wrap: balance; }
.kx-doc h2:first-child { margin-top: 0; }
.kx-doc p, .kx-doc li { max-width: 72ch; }
.kx-doc li { margin-block: 0.3rem; }
.kx-doc code { font-family: var(--font-mono); font-size: 0.85em; background: var(--color-bg); padding: 1px 4px; border-radius: 4px; }
.kx-doc pre { background: var(--color-bg); border: 1px solid var(--color-border-line); border-radius: var(--cds-radius); padding: var(--cds-pad-md); overflow-x: auto; }
.kx-doc pre code { background: none; padding: 0; font-size: 0.8125rem; }
.kx-doc .kx-spec td { vertical-align: top; }
.kx-doc .kx-spec { margin-block: 0.5rem; }
.kx-doc b, .kx-doc strong { font-weight: 600; }
@media (max-width: 600px) { .kx-doc { padding: var(--cds-pad-lg); } }
</style>
"""

SHIM = """<script>
/* Local stand-in for the dashboard data API: all data is embedded below. */
(function () {
  const D = window.REPORT_DATA;
  const subs = [];
  const params = { target: 'KRS only' };
  const run = () => subs.forEach((f) => f());
  let t;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(run, 120); });
  window.dash = {
    colors: ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e87ba4', '#008300', '#eda100', '#e34948'],
    data: (id) => ({ status: 'ok', data: D[id] || [] }),
    calc: (id, { inputs, fn }) => { D[id] = fn(...inputs.map((i) => D[i])); },
    onData: (fn) => { subs.push(fn); fn(); },
    params: () => params,
    setParams: (o) => { Object.assign(params, o); run(); },
    resetParams: () => {},
    setLink: (h) => { try { history.replaceState(null, '', '#' + h); } catch (e) {} },
    refresh: () => {},
  };
})();
</script>
"""


def main():
    data = {k: json.loads((DASH / f"{k}.json").read_text()) for k in DATASETS}
    data["peaks"] = json.loads((ROOT / "outputs" / "peak_results.json").read_text())
    page = (SRC / "page.html").read_text()
    sections = (SRC / "peaks.html").read_text() + (SRC / "approach.html").read_text()
    close = "\n</div>\n\n<script>\ndash.calc("
    assert close in page
    page = page.replace(close, "\n" + sections + "</div>\n\n<script>\ndash.calc(", 1)
    final = "<script>\n(function () {\n  const K = window.KX;\n  dash.onData(() => K.drawAll());"
    assert final in page
    page = page.replace(final, "<script>\n" + (SRC / "peaks.js").read_text() + "</script>\n\n" + final, 1)
    page = page.replace('<div class="kx">', '<div class="kx kx-wrap">', 1)
    page = page.replace(
        "Krishnaraja Sagar, Cauvery basin.",
        "WELL Labs Water Data Hackathon, Theme 2 / Question 3. Krishnaraja Sagar, Cauvery basin.", 1)

    d3 = (ROOT / "src" / "vendor" / "d3.v7.9.0.min.js").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = (HEAD + "<script>" + d3 + "</script>\n"
            + "<script>window.REPORT_DATA = " + payload + ";</script>\n" + SHIM
            + "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
