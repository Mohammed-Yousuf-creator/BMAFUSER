from pathlib import Path
import json, html

src = Path("./fxx24BMA.txt")
data = json.loads(src.read_text(encoding="utf-8"))
assert data["location"] == "New Delhi"
assert data["variable"] == "temperature"

points = data["distribution"]
payload = json.dumps(points, separators=(",", ":"))
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>New Delhi — F24 Temperature Distribution</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 24px; background: #f3f6fa; color: #172033; font: 15px/1.5 system-ui, sans-serif; }}
  main {{ width: min(900px, 100%); background: white; border: 1px solid #e4e9f0; border-radius: 18px; padding: 26px; box-shadow: 0 12px 36px #17203312; }}
  h1 {{ margin: 0; font-size: 1.45rem; }}
  .sub {{ margin: 4px 0 22px; color: #667085; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
  .stat {{ flex: 1 1 170px; padding: 13px 16px; border-radius: 12px; background: #f5f7fb; }}
  .stat span {{ display:block; color:#667085; font-size:.82rem; }}
  .stat strong {{ display:block; font-size:1.25rem; margin-top:3px; }}
  .chart-wrap {{ position: relative; height: 390px; }}
  .note {{ color:#667085; font-size:.83rem; margin:14px 0 0; }}
  @media (max-width: 560px) {{ main {{ padding:18px; }} .chart-wrap {{ height:300px; }} }}
</style>
</head>
<body>
<main>
  <h1>New Delhi</h1>
  <p class="sub">Temperature probability distribution · Forecast horizon F24</p>
  <section class="stats">
    <div class="stat"><span>Expected temperature</span><strong id="mean">—</strong></div>
    <div class="stat"><span>Most probable bin</span><strong id="mode">—</strong></div>
    <div class="stat"><span>Distribution points</span><strong id="count">—</strong></div>
  </section>
  <div class="chart-wrap"><canvas id="chart" role="img" aria-label="New Delhi temperature probability distribution"></canvas></div>
  <p class="note">Bars show probability mass at each temperature point from the BMA output. This is the supplied distribution, not a newly calculated forecast.</p>
</main>
<script>
const distribution = {payload}; // [temperature °C, probability mass]
const total = distribution.reduce((s, p) => s + p[1], 0);
const mean = distribution.reduce((s, p) => s + p[0] * p[1], 0) / total;
const modePoint = distribution.reduce((best, p) => p[1] > best[1] ? p : best, distribution[0]);
document.getElementById("mean").textContent = mean.toFixed(1) + " °C";
document.getElementById("mode").textContent = modePoint[0].toFixed(1) + " °C";
document.getElementById("count").textContent = distribution.length;
new Chart(document.getElementById("chart"), {{
  type: "bar",
  data: {{
    datasets: [{{
      label: "Probability mass",
      data: distribution.map(([x, y]) => ({{x, y}})),
      backgroundColor: "rgba(45, 112, 214, 0.48)",
      borderColor: "#2d70d6",
      borderWidth: 1,
      barPercentage: 1,
      categoryPercentage: 1
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: "index", intersect: false }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{
        title: items => items.length ? items[0].parsed.x.toFixed(2) + " °C" : "",
        label: item => "Probability mass: " + item.parsed.y.toFixed(5)
      }}}}
    }},
    scales: {{
      x: {{ type: "linear", title: {{ display: true, text: "Temperature (°C)" }},
        ticks: {{ callback: value => Number(value).toFixed(0) }} }},
      y: {{ beginAtZero: true, title: {{ display: true, text: "Probability mass" }} }}
    }}
  }}
}});
</script>
</body>
</html>
"""
out = Path("./new_delhi_distribution.html")
out.write_text(page, encoding="utf-8")
print(f"Created {out} ({len(points)} distribution points)")
