import { fmtMin, mode } from "../modes.js";

const TAG_TEXT = {
  fastest: "Fastest",
  cheapest: "Cheapest",
  least_walking: "Least walking",
  fewest_transfers: "Fewest changes",
};

const segTime = (s) => s.ml_predicted_time_min ?? s.time_min;

/** The journey drawn as a transit line: one coloured run per ride, a station ring at each change. */
export function LineStrip({ segments }) {
  const total = segments.reduce((a, s) => a + segTime(s) + s.wait_min, 0) || 1;
  return (
    <div className="strip" aria-hidden="true">
      {segments.map((s, i) => {
        const m = mode(s.transport_type);
        const grow = (segTime(s) + s.wait_min) / total;
        return (
          <div key={i} className={`strip-run strip-${s.transport_type}`} style={{ flexGrow: grow, "--c": m.color }}>
            <span className="strip-ring" />
            <span className="strip-bar" />
            <span className="strip-label">
              {s.transport_type === "walk" ? "Walk" : shortRoute(s)}
            </span>
          </div>
        );
      })}
      <span className="strip-ring strip-end" />
    </div>
  );
}

function shortRoute(s) {
  if (s.transport_type === "bus") {
    const m = s.route_name.match(/MTC\s+(\S+)/);
    return m ? m[1] : "Bus";
  }
  if (s.transport_type === "metro") {
    const m = s.route_name.match(/(Blue|Green)/);
    return m ? `${m[1]} Line` : "Metro";
  }
  if (s.transport_type === "rail") return s.route_name.startsWith("MRTS") ? "MRTS" : "Suburban";
  return mode(s.transport_type).label;
}

export default function RouteCard({ option, selected, expanded, onSelect, onToggle, mlOn }) {
  const time = mlOn && option.ml_total_time_min != null ? option.ml_total_time_min : option.total_time_min;
  return (
    <article
      className={`card${selected ? " is-selected" : ""}`}
      aria-label={`Option ${option.rank}: ${fmtMin(time)}, ₹${option.total_fare_inr}`}
    >
      <button className="card-main" onClick={onSelect} aria-pressed={selected}>
        <div className="card-head">
          <div className="card-time">
            <span className="card-time-value">{fmtMin(time)}</span>
            {mlOn && option.ml_total_time_min != null && (
              <span className="card-time-note">timetable says {fmtMin(option.total_time_min)}</span>
            )}
          </div>
          <div className="card-tags">
            {option.tags.map((t) => (
              <span key={t} className={`tag tag-${t}`}>{TAG_TEXT[t] || t}</span>
            ))}
          </div>
        </div>
        <LineStrip segments={option.segments} />
        <dl className="card-facts">
          <div><dt>Fare</dt><dd>₹{Number(option.total_fare_inr).toFixed(0)}</dd></div>
          <div><dt>Walking</dt><dd>{option.walking_distance_km > 0 ? `${option.walking_distance_km.toFixed(1)} km` : "None"}</dd></div>
          <div><dt>Changes</dt><dd>{option.transfers}</dd></div>
          <div><dt>Distance</dt><dd>{option.total_distance_km.toFixed(1)} km</dd></div>
        </dl>
      </button>
      <button className="card-toggle" onClick={onToggle} aria-expanded={expanded}>
        {expanded ? "Hide steps" : "Show steps"}
      </button>
      {expanded && <Steps option={option} mlOn={mlOn} />}
    </article>
  );
}

function Steps({ option, mlOn }) {
  return (
    <ol className="steps">
      {option.segments.map((s, i) => {
        const m = mode(s.transport_type);
        const t = mlOn && s.ml_predicted_time_min != null ? s.ml_predicted_time_min : s.time_min;
        return (
          <li key={i} style={{ "--c": m.color }}>
            <div className="step-line" aria-hidden="true" />
            <div className="step-body">
              <div className="step-title">
                {s.transport_type === "walk"
                  ? `Walk to ${s.to_name}`
                  : `${m.label}: ${s.route_name}`}
              </div>
              <div className="step-detail">
                {s.transport_type === "walk"
                  ? `${s.distance_km.toFixed(2)} km, about ${Math.round(t)} min`
                  : `Board at ${s.from_name}, ride ${s.num_stops} stop${s.num_stops > 1 ? "s" : ""} to ${s.to_name}`}
              </div>
              {s.transport_type !== "walk" && (
                <div className="step-numbers">
                  <span>{Math.round(t)} min ride</span>
                  <span>~{Math.round(s.wait_min)} min wait</span>
                  <span>{s.distance_km.toFixed(1)} km</span>
                  <span>
                    {s.shares_ticket_with_previous ? "same ticket" : `₹${Number(s.fare_inr).toFixed(0)}`}
                  </span>
                </div>
              )}
            </div>
          </li>
        );
      })}
      <li className="step-arrive" style={{ "--c": "var(--ink)" }}>
        <div className="step-line" aria-hidden="true" />
        <div className="step-body"><div className="step-title">Arrive at {option.segments.at(-1).to_name}</div></div>
      </li>
    </ol>
  );
}
