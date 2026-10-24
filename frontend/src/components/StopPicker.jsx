import { useId, useMemo, useRef, useState } from "react";
import { mode } from "../modes.js";

function matchStops(stops, query) {
  const q = query.trim().toLowerCase();
  if (!q) return stops.slice(0, 8);
  const scored = [];
  for (const s of stops) {
    const name = s.stop_name.toLowerCase();
    let rank = -1;
    if (name.startsWith(q)) rank = 0;
    else if (name.split(/\s+/).some((w) => w.startsWith(q))) rank = 1;
    else if (name.includes(q)) rank = 2;
    else if (String(s.area).toLowerCase().includes(q) || s.stop_id.toLowerCase() === q) rank = 3;
    if (rank >= 0) scored.push([rank, name.length, s]);
  }
  scored.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return scored.slice(0, 8).map((x) => x[2]);
}

export default function StopPicker({ label, stops, value, onChange, dotColor }) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const inputRef = useRef(null);

  const selected = stops.find((s) => s.stop_id === value);
  const results = useMemo(() => matchStops(stops, query), [stops, query]);

  const choose = (s) => {
    onChange(s.stop_id);
    setQuery("");
    setOpen(false);
    inputRef.current?.blur();
  };

  const onKeyDown = (e) => {
    if (!open && (e.key === "ArrowDown" || e.key === "Enter")) return setOpen(true);
    if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(a + 1, results.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)); }
    else if (e.key === "Enter" && results[active]) { e.preventDefault(); choose(results[active]); }
    else if (e.key === "Escape") setOpen(false);
  };

  return (
    <div className="picker">
      <label htmlFor={id} className="picker-label">
        <span className="picker-dot" style={{ background: dotColor }} aria-hidden="true" />
        {label}
      </label>
      <input
        id={id}
        ref={inputRef}
        className="picker-input"
        role="combobox"
        aria-expanded={open}
        aria-controls={`${id}-list`}
        aria-autocomplete="list"
        autoComplete="off"
        placeholder="Search a station or bus stop"
        value={open ? query : selected ? selected.stop_name : ""}
        onFocus={() => { setOpen(true); setActive(0); }}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
        onChange={(e) => { setQuery(e.target.value); setActive(0); setOpen(true); }}
        onKeyDown={onKeyDown}
      />
      {selected && !open && <div className="picker-area">{selected.area}</div>}
      {open && (
        <ul id={`${id}-list`} className="picker-list" role="listbox">
          {results.length === 0 && <li className="picker-empty">No stop matches “{query}”. Try part of the name or the area.</li>}
          {results.map((s, i) => (
            <li
              key={s.stop_id}
              role="option"
              aria-selected={i === active}
              className={i === active ? "is-active" : ""}
              onMouseDown={(e) => { e.preventDefault(); choose(s); }}
              onMouseEnter={() => setActive(i)}
            >
              <span className="picker-name">{s.stop_name}</span>
              <span className="picker-meta">
                {s.area}
                <span className="picker-modes">
                  {s.modes.map((m) => (
                    <span key={m} className="mode-pip" style={{ background: mode(m).color }} title={mode(m).label} />
                  ))}
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
