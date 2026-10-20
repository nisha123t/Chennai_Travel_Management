import { useEffect, useState } from "react";
import { api } from "./api.js";
import { fmtHour, MODES } from "./modes.js";
import StopPicker from "./components/StopPicker.jsx";
import RouteCard from "./components/RouteCard.jsx";
import JourneyMap from "./components/JourneyMap.jsx";
import ModelPanel from "./components/ModelPanel.jsx";

const PREFERENCES = [
  { value: "balanced", label: "Balanced", hint: "Weighs time, fare, walking and changes" },
  { value: "fastest", label: "Fastest", hint: "Shortest predicted journey time" },
  { value: "cheapest", label: "Cheapest", hint: "Lowest total fare" },
  { value: "least_walking", label: "Least walking", hint: "Fewest kilometres on foot" },
];

const nowHour = () => new Date().getHours();

export default function App() {
  const [tab, setTab] = useState("plan");
  const [stops, setStops] = useState([]);
  const [stopsError, setStopsError] = useState(null);

  const [start, setStart] = useState("");
  const [dest, setDest] = useState("");
  const [preference, setPreference] = useState("balanced");
  const [hour, setHour] = useState(nowHour());
  const [dayType, setDayType] = useState([0, 6].includes(new Date().getDay()) ? "weekend" : "weekday");
  const [mlOn, setMlOn] = useState(true);
  const [model, setModel] = useState("random_forest");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(0);
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    api.stops().then(setStops).catch((e) => setStopsError(e.message));
  }, []);

  const search = async (overrides = {}) => {
    const s = overrides.start ?? start;
    const d = overrides.dest ?? dest;
    if (!s || !d) return setError("Choose where you’re starting from and where you’re going.");
    if (s === d) return setError("Start and destination are the same stop. Pick a different destination.");
    setLoading(true);
    setError(null);
    try {
      const r = await api.routes({
        start: s, destination: d,
        preference: overrides.preference ?? preference,
        departure_hour: Number(overrides.hour ?? hour),
        day_type: overrides.dayType ?? dayType,
        ml_enhanced: overrides.mlOn ?? mlOn,
        model: overrides.model ?? model,
        max_options: 3,
      });
      setResult(r);
      setSelected(0);
      setExpanded(null);
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // Once a search has run, changing a setting re-runs it straight away.
  const update = (setter, key) => (value) => {
    setter(value);
    if (result) search({ [key]: value });
  };

  const swap = () => {
    setStart(dest);
    setDest(start);
    if (result) search({ start: dest, dest: start });
  };

  const option = result?.options?.[selected];

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <svg viewBox="0 0 40 40" className="brand-mark" aria-hidden="true">
            <path d="M5 29 H14 L26 11 H35" fill="none" stroke={MODES.metro.color} strokeWidth="4.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M5 11 H14 L26 29 H35" fill="none" stroke={MODES.bus.color} strokeWidth="4.5" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="20" cy="20" r="5" fill="#fff" stroke="currentColor" strokeWidth="2.5" />
          </svg>
          <div>
            <h1>Chennai Route Planner</h1>
            <p>Metro, suburban rail, MTC buses and walking, with journey times predicted by machine learning.</p>
          </div>
        </div>
        <nav className="tabs" aria-label="Sections">
          <button className={tab === "plan" ? "is-on" : ""} aria-current={tab === "plan"} onClick={() => setTab("plan")}>Plan a journey</button>
          <button className={tab === "models" ? "is-on" : ""} aria-current={tab === "models"} onClick={() => setTab("models")}>Model results</button>
        </nav>
      </header>

      {tab === "models" ? (
        <main className="main-models"><ModelPanel /></main>
      ) : (
        <main className="layout">
          <form className="planner" onSubmit={(e) => { e.preventDefault(); search(); }}>
            {stopsError && <div className="notice notice-error" role="alert">{stopsError}</div>}
            <div className="pickers">
              <StopPicker label="From" stops={stops} value={start} onChange={setStart} dotColor="var(--ink)" />
              <button type="button" className="swap" onClick={swap} aria-label="Swap start and destination" title="Swap">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4v14M7 18l-3-3M7 18l3-3M17 20V6M17 6l-3 3M17 6l3 3" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </button>
              <StopPicker label="To" stops={stops} value={dest} onChange={setDest} dotColor="var(--dest)" />
            </div>

            <fieldset className="prefs">
              <legend>Prioritise</legend>
              <div className="segmented">
                {PREFERENCES.map((p) => (
                  <label key={p.value} className={preference === p.value ? "is-on" : ""} title={p.hint}>
                    <input type="radio" name="pref" value={p.value} checked={preference === p.value}
                      onChange={() => update(setPreference, "preference")(p.value)} />
                    {p.label}
                  </label>
                ))}
              </div>
              <p className="hint">{PREFERENCES.find((p) => p.value === preference).hint}</p>
            </fieldset>

            <div className="when">
              <label className="field">
                <span>Leaving at</span>
                <select value={hour} onChange={(e) => update(setHour, "hour")(Number(e.target.value))}>
                  {Array.from({ length: 24 }, (_, h) => <option key={h} value={h}>{fmtHour(h)}</option>)}
                </select>
              </label>
              <label className="field">
                <span>Day</span>
                <select value={dayType} onChange={(e) => update(setDayType, "dayType")(e.target.value)}>
                  <option value="weekday">Weekday</option>
                  <option value="weekend">Weekend</option>
                </select>
              </label>
            </div>

            <div className="ml-row">
              <label className="switch">
                <input type="checkbox" checked={mlOn} onChange={(e) => update(setMlOn, "mlOn")(e.target.checked)} />
                <span>Predict times with ML</span>
              </label>
              <select aria-label="Prediction model" value={model} disabled={!mlOn}
                onChange={(e) => update(setModel, "model")(e.target.value)}>
                <option value="random_forest">Random Forest</option>
                <option value="linear_regression">Linear Regression</option>
              </select>
            </div>

            <button type="submit" className="go" disabled={loading || !stops.length}>
              {loading ? "Finding routes…" : "Find routes"}
            </button>
          </form>

          <section className="results" aria-live="polite" aria-busy={loading}>
            {error && <div className="notice notice-error" role="alert">{error}</div>}
            {result?.ml_warning && <div className="notice">{result.ml_warning}</div>}

            {!result && !loading && !error && (
              <div className="empty">
                {stops.length > 0 && <JourneyMap stops={stops} option={null} startId={start} destId={dest} />}
                <p>Choose two stops to compare up to three ways of getting there.</p>
              </div>
            )}

            {loading && !result && (
              <div className="skeletons" aria-hidden="true">
                {[0, 1, 2].map((i) => <div key={i} className="skeleton" />)}
              </div>
            )}

            {result && (
              <div className={`result-grid${loading ? " is-stale" : ""}`}>
                <div className="result-map">
                  <JourneyMap stops={stops} option={option} startId={result.start.stop_id} destId={result.destination.stop_id} />
                </div>
                <div className="result-list">
                  <p className="result-meta">
                    {result.options.length} route{result.options.length > 1 ? "s" : ""} from {result.start.stop_name} to{" "}
                    {result.destination.stop_name}, leaving {fmtHour(result.departure_hour)} on a {result.day_type}.
                    {" "}Chosen from {result.candidates_considered} possible journeys.
                  </p>
                  {result.options.map((o, i) => (
                    <RouteCard
                      key={o.option_id}
                      option={o}
                      mlOn={result.ml_enhanced}
                      selected={i === selected}
                      expanded={expanded === i}
                      onSelect={() => setSelected(i)}
                      onToggle={() => { setSelected(i); setExpanded(expanded === i ? null : i); }}
                    />
                  ))}
                  <CompareTable options={result.options} mlOn={result.ml_enhanced} selected={selected} onSelect={setSelected} />
                </div>
              </div>
            )}
          </section>
        </main>
      )}
    </div>
  );
}

function CompareTable({ options, mlOn, selected, onSelect }) {
  if (options.length < 2) return null;
  const time = (o) => (mlOn && o.ml_total_time_min != null ? o.ml_total_time_min : o.total_time_min);
  const best = {
    time: Math.min(...options.map(time)),
    fare: Math.min(...options.map((o) => o.total_fare_inr)),
    walk: Math.min(...options.map((o) => o.walking_distance_km)),
    tr: Math.min(...options.map((o) => o.transfers)),
  };
  return (
    <table className="compare">
      <caption>Side by side</caption>
      <thead>
        <tr><th scope="col">Route</th><th scope="col">Time</th><th scope="col">Fare</th><th scope="col">Walk</th><th scope="col">Changes</th></tr>
      </thead>
      <tbody>
        {options.map((o, i) => (
          <tr key={o.option_id} className={i === selected ? "is-selected" : ""} onClick={() => onSelect(i)}>
            <th scope="row">{o.summary}</th>
            <td className={time(o) === best.time ? "best" : ""}>{Math.round(time(o))} min</td>
            <td className={o.total_fare_inr === best.fare ? "best" : ""}>₹{o.total_fare_inr.toFixed(0)}</td>
            <td className={o.walking_distance_km === best.walk ? "best" : ""}>{o.walking_distance_km.toFixed(1)} km</td>
            <td className={o.transfers === best.tr ? "best" : ""}>{o.transfers}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
