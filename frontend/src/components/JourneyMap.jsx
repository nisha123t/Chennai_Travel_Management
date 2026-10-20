import { useMemo } from "react";
import { mode } from "../modes.js";

const W = 480;
const H = 400;
const PAD = 30;

/**
 * Schematic map: every stop in the network as a faint dot, the selected
 * journey drawn on top in its mode colours. Pure SVG, no tiles, works offline.
 */
export default function JourneyMap({ stops, option, startId, destId }) {
  const byId = useMemo(() => Object.fromEntries(stops.map((s) => [s.stop_id, s])), [stops]);

  const focus = useMemo(() => {
    const ids = option ? option.stops.map((s) => s.stop_id) : [startId, destId].filter(Boolean);
    const pts = ids.map((id) => byId[id]).filter(Boolean);
    return pts.length ? pts : stops;
  }, [option, startId, destId, byId, stops]);

  const project = useMemo(() => {
    if (!focus.length) return null;
    let minLat = Math.min(...focus.map((s) => s.latitude));
    let maxLat = Math.max(...focus.map((s) => s.latitude));
    let minLon = Math.min(...focus.map((s) => s.longitude));
    let maxLon = Math.max(...focus.map((s) => s.longitude));
    // keep at least ~6 km of context around short journeys
    const minSpan = 0.055;
    const padSpan = (lo, hi) => {
      const span = Math.max(hi - lo, minSpan) * 1.15;
      const mid = (lo + hi) / 2;
      return [mid - span / 2, mid + span / 2];
    };
    [minLat, maxLat] = padSpan(minLat, maxLat);
    [minLon, maxLon] = padSpan(minLon, maxLon);
    const kx = Math.cos((13.05 * Math.PI) / 180); // longitude shrink at Chennai's latitude
    const spanX = (maxLon - minLon) * kx;
    const spanY = maxLat - minLat;
    const scale = Math.min((W - 2 * PAD) / spanX, (H - 2 * PAD) / spanY);
    const offX = (W - spanX * scale) / 2;
    const offY = (H - spanY * scale) / 2;
    return (lat, lon) => [offX + (lon - minLon) * kx * scale, offY + (maxLat - lat) * scale];
  }, [focus]);

  if (!project) return <div className="map map-empty">Loading map…</div>;

  const inView = ([x, y]) => x > -20 && x < W + 20 && y > -20 && y < H + 20;
  const start = byId[startId];
  const dest = byId[destId];

  const segPaths = option
    ? (() => {
        let li = 0;
        return option.segments.map((seg) => {
          const legs = option.legs.slice(li, li + seg.num_stops);
          li += seg.num_stops;
          const ids = [legs[0]?.from_stop, ...legs.map((l) => l.to_stop)].filter(Boolean);
          const pts = ids.map((id) => byId[id]).filter(Boolean).map((s) => project(s.latitude, s.longitude));
          return { seg, pts };
        });
      })()
    : [];

  const changeStops = option ? option.segments.slice(1).map((s) => byId[s.from_stop]).filter(Boolean) : [];

  return (
    <figure className="map">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={option ? `Map of the selected journey: ${option.summary}` : "Map of the Chennai network"}>
        <rect width={W} height={H} className="map-bg" />
        {stops.map((s) => {
          const p = project(s.latitude, s.longitude);
          return inView(p) ? <circle key={s.stop_id} cx={p[0]} cy={p[1]} r={2.2} className="map-stop" /> : null;
        })}
        {segPaths.map(({ seg, pts }, i) => (
          <polyline
            key={i}
            points={pts.map((p) => p.join(",")).join(" ")}
            fill="none"
            stroke={mode(seg.transport_type).color}
            strokeWidth={seg.transport_type === "walk" ? 3 : 5}
            strokeDasharray={seg.transport_type === "walk" ? "2 6" : undefined}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
        {option && option.stops.slice(1, -1).map((s) => {
          const p = project(s.latitude, s.longitude);
          return <circle key={s.stop_id} cx={p[0]} cy={p[1]} r={2.6} className="map-pass" />;
        })}
        {changeStops.map((s) => {
          const [x, y] = project(s.latitude, s.longitude);
          return (
            <g key={`c-${s.stop_id}`}>
              <circle cx={x} cy={y} r={6} className="map-change" />
              <Label x={x} y={y + 4} gap={10} className="map-label map-label-small">{s.stop_name}</Label>
            </g>
          );
        })}
        {[[start, "start"], [dest, "dest"]].map(([s, kind]) => {
          if (!s) return null;
          const [x, y] = project(s.latitude, s.longitude);
          return (
            <g key={kind}>
              <circle cx={x} cy={y} r={8} className={`map-end map-${kind}`} />
              <Label x={x} y={y + 5} gap={12} className="map-label">{s.stop_name}</Label>
            </g>
          );
        })}
      </svg>
      {option && (
        <figcaption className="map-legend">
          {option.modes_used.map((m) => (
            <span key={m}><i style={{ background: mode(m).color }} />{mode(m).label}</span>
          ))}
        </figcaption>
      )}
    </figure>
  );
}

/** Stop name beside a point, flipped to the left when it would run off the right edge. */
function Label({ x, y, gap, className, children }) {
  const flip = x > W - 120;
  return (
    <text x={flip ? x - gap : x + gap} y={y} textAnchor={flip ? "end" : "start"} className={className}>
      {children}
    </text>
  );
}
