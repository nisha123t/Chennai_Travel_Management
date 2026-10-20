import { useEffect, useState } from "react";
import { api } from "../api.js";
import { mode } from "../modes.js";

const MODELS = [
  { key: "baseline_timetable", label: "Timetable only", color: "#b5c1bf" },
  { key: "linear_regression", label: "Linear Regression", color: "#7a6a9e" },
  { key: "random_forest", label: "Random Forest", color: "#0f6b66" },
];

export default function ModelPanel() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.metrics().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="notice notice-error" role="alert">{error}</div>;
  if (!data) return <div className="notice" aria-busy="true">Loading model results…</div>;

  const rf = data.random_forest;
  const lr = data.linear_regression;
  const base = data.baseline_timetable;
  const gain = base ? Math.round((1 - rf.mae / base.mae) * 100) : null;

  return (
    <section className="models" aria-labelledby="models-title">
      <header className="models-head">
        <h2 id="models-title">How well do the models predict journey time?</h2>
        <p>
          Random Forest is off by <strong>{rf.mae.toFixed(1)} min</strong> on average for routes it never saw in
          training{gain != null && <>, {gain}% closer than the timetable alone</>}. Linear Regression
          ({lr.mae.toFixed(1)} min) can’t capture that traffic delay grows with trip length, so it barely beats the timetable.
        </p>
      </header>

      <div className="models-grid">
        <div className="panel">
          <h3>Test-set scores</h3>
          <table className="metrics">
            <thead>
              <tr><th scope="col">Model</th><th scope="col">MAE</th><th scope="col">RMSE</th><th scope="col">R²</th><th scope="col">5-fold MAE</th></tr>
            </thead>
            <tbody>
              {MODELS.filter((m) => data[m.key]).map((m) => {
                const r = data[m.key];
                const cv = r.cv_5fold;
                return (
                  <tr key={m.key}>
                    <th scope="row"><i className="swatch" style={{ background: m.color }} />{m.label}</th>
                    <td>{r.mae.toFixed(2)}</td>
                    <td>{r.rmse.toFixed(2)}</td>
                    <td>{r.r2_score.toFixed(3)}</td>
                    <td>{cv ? `${cv.mae_mean.toFixed(2)} ± ${cv.mae_std.toFixed(2)}` : "n/a"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="fineprint">
            Errors in minutes. {data.test_samples.toLocaleString()} test rides on routes held out of training
            ({data.train_samples.toLocaleString()} training rides). Journey times are simulated from timetable
            times with a time-of-day traffic profile, because no open Chennai trip-time data exists.
          </p>
        </div>

        <div className="panel">
          <h3>Average error by mode (min)</h3>
          <ModeBars data={data} />
        </div>

        <div className="panel">
          <h3>Predicted vs actual, Random Forest</h3>
          <Scatter sample={data.test_predictions_sample} />
        </div>

        <div className="panel">
          <h3>What the Random Forest relies on</h3>
          <Importances values={rf.feature_importances || {}} />
        </div>
      </div>
    </section>
  );
}

function ModeBars({ data }) {
  const modes = Object.keys(data.random_forest.per_mode).sort(
    (a, b) => data.random_forest.per_mode[b].n - data.random_forest.per_mode[a].n
  );
  const max = Math.max(...modes.flatMap((m) => MODELS.map((x) => data[x.key]?.per_mode?.[m]?.mae || 0)));
  return (
    <div className="bars">
      <div className="bars-legend">
        {MODELS.map((m) => <span key={m.key}><i style={{ background: m.color }} />{m.label}</span>)}
      </div>
      {modes.map((m) => (
        <div className="bars-row" key={m}>
          <div className="bars-label">
            {mode(m).label}
            <small>{data.random_forest.per_mode[m].n} rides</small>
          </div>
          <div className="bars-group">
            {MODELS.map((x) => {
              const v = data[x.key]?.per_mode?.[m]?.mae;
              if (v == null) return null;
              return (
                <div className="bar" key={x.key} title={`${x.label}: ${v.toFixed(2)} min`}>
                  <span style={{ width: `${(v / max) * 100}%`, background: x.color }} />
                  <em>{v.toFixed(1)}</em>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function Scatter({ sample }) {
  if (!sample) return null;
  const W = 320, H = 260, P = 34;
  const actual = sample.actual;
  const pred = sample.random_forest;
  const max = Math.ceil(Math.max(...actual, ...pred) / 20) * 20;
  const x = (v) => P + (v / max) * (W - P - 10);
  const y = (v) => H - P - (v / max) * (H - P - 10);
  const ticks = [0, max / 2, max];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="scatter" role="img" aria-label="Scatter of predicted against actual journey time">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(0)} x2={x(max)} y1={y(t)} y2={y(t)} className="grid" />
          <text x={P - 6} y={y(t) + 4} textAnchor="end" className="tick">{t}</text>
          <text x={x(t)} y={H - P + 16} textAnchor="middle" className="tick">{t}</text>
        </g>
      ))}
      <line x1={x(0)} y1={y(0)} x2={x(max)} y2={y(max)} className="diag" />
      {actual.map((a, i) => (
        <circle key={i} cx={x(a)} cy={y(pred[i])} r={2.6}
          fill={mode(sample.transport_type[i]).color} fillOpacity={0.6} />
      ))}
      <text x={x(max / 2)} y={H - 4} textAnchor="middle" className="axis">Actual (min)</text>
      <text transform={`translate(10 ${y(max / 2)}) rotate(-90)`} textAnchor="middle" className="axis">Predicted (min)</text>
    </svg>
  );
}

const FEATURE_NAMES = {
  distance_km: "Distance",
  stops_count: "Number of stops",
  departure_hour: "Departure hour",
};

function prettyFeature(f) {
  if (FEATURE_NAMES[f]) return FEATURE_NAMES[f];
  const [, group, val] = f.match(/^(transport_type|time_of_day|day_type)_(.+)$/) || [];
  if (group === "transport_type") return `Mode: ${mode(val).label}`;
  if (group) return `${group === "day_type" ? "Day" : "Time"}: ${val.replace(/_/g, " ")}`;
  return f;
}

function Importances({ values }) {
  const entries = Object.entries(values).slice(0, 7);
  const max = Math.max(...entries.map((e) => e[1]), 0.0001);
  return (
    <ul className="importances">
      {entries.map(([k, v]) => (
        <li key={k}>
          <span className="imp-name">{prettyFeature(k)}</span>
          <span className="imp-bar"><span style={{ width: `${(v / max) * 100}%` }} /></span>
          <span className="imp-val">{(v * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}
