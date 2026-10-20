"""
Dataset builder for the journey-time prediction models.

Each training sample is one *continuous ride segment* (board at stop A, ride
N stops on the same route, alight at stop B) - the same unit that
router.find_route() returns in "segments". This makes `stops_count` and
`distance_km` real quantities taken from the network data rather than
placeholders.

IMPORTANT - what is real and what is simulated
-----------------------------------------------
* Real (from backend/data): stops, route stop sequences, hop distances and
  timetable / schedule base times.
* Simulated: the *observed* travel time. No open per-trip travel-time feed
  exists for Chennai MTC / CMRL / suburban rail, so we simulate observations
  from the timetable base time using a time-of-day congestion profile,
  mode-specific congestion sensitivity, per-stop dwell time and random noise.
  Swap `travel_time_min` for real observations (e.g. GPS logs or a maps API)
  and the rest of the pipeline works unchanged.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MAX_SEGMENT_STOPS = 12  # longest continuous ride enumerated per route

# How strongly each mode is affected by road congestion (0 = immune).
CONGESTION_SENSITIVITY = {
    "bus": 1.00, "cab": 0.90, "auto": 0.75,
    "rail": 0.08, "metro": 0.05, "walk": 0.0,
}
# Extra dwell minutes per intermediate stop (boarding/alighting, signals).
DWELL_PER_STOP = {"bus": 0.30, "rail": 0.35, "metro": 0.25}
# Multiplicative noise (std of log) - road modes are less predictable.
NOISE_SIGMA = {"bus": 0.10, "cab": 0.08, "auto": 0.08,
               "rail": 0.05, "metro": 0.03, "walk": 0.06}


def time_of_day_bucket(hour: float, day_type: str = "weekday") -> str:
    """Map a departure hour (0-23) to the coarse bucket used by the API."""
    h = int(hour) % 24
    if day_type == "weekend":
        return "weekend_day" if 7 <= h < 21 else "weekend_night"
    if 8 <= h < 11:
        return "morning_peak"
    if 17 <= h < 21:
        return "evening_peak"
    if 11 <= h < 17:
        return "afternoon"
    return "off_peak"


# Representative hour for each bucket (used when a caller only sends a bucket).
BUCKET_TO_HOUR = {
    "morning_peak": 9, "evening_peak": 18, "afternoon": 14,
    "off_peak": 22, "weekend_day": 13, "weekend_night": 22,
}


def congestion_level(hour: float, day_type: str) -> float:
    """
    Smooth road-congestion index (0 = free flow) over the day.
    Weekdays: morning peak ~09:00, evening peak ~18:30, small midday bump.
    Weekends: one broad, lower afternoon/evening hump.
    """
    def bump(centre, width, amp):
        d = min(abs(hour - centre), 24 - abs(hour - centre))  # circular
        return amp * np.exp(-0.5 * (d / width) ** 2)

    if day_type == "weekend":
        return bump(16.5, 3.0, 0.30)
    return bump(9.0, 1.3, 0.55) + bump(18.5, 1.6, 0.65) + bump(13.5, 1.5, 0.12)


def build_segments() -> pd.DataFrame:
    """Enumerate every continuous ride segment from the network data."""
    conns = pd.read_csv(DATA_DIR / "connections.csv")
    seqs = pd.read_csv(DATA_DIR / "stop_sequences.csv").sort_values(["route_id", "sequence_no"])
    hop = {(r.route_id, r.from_stop, r.to_stop): r for r in conns.itertuples()}

    rows = []
    for route_id, grp in seqs.groupby("route_id"):
        stops = grp["stop_id"].tolist()
        for i in range(len(stops) - 1):
            dist = base = 0.0
            mode = None
            for j in range(i + 1, min(len(stops), i + 1 + MAX_SEGMENT_STOPS)):
                c = hop.get((route_id, stops[j - 1], stops[j]))
                if c is None:  # gap in data - stop extending this segment
                    break
                mode = str(c.transport_type).lower()
                dist += float(c.distance_km)
                base += float(c.estimated_time_min)
                rows.append({
                    "route_id": route_id,
                    "from_stop": stops[i],
                    "to_stop": stops[j],
                    "transport_type": mode,
                    "distance_km": round(dist, 3),
                    "stops_count": j - i,
                    "timetable_time_min": round(base, 2),
                })
    return pd.DataFrame(rows)


def simulate_observed_time(seg, hour, day_type, rng) -> float:
    mode = seg["transport_type"]
    if mode == "walk":
        speed = float(np.clip(rng.normal(4.5, 0.4), 3.0, 6.0))
        t = seg["distance_km"] / speed * 60.0
    else:
        cong = congestion_level(hour, day_type)
        t = seg["timetable_time_min"] * (1.0 + CONGESTION_SENSITIVITY.get(mode, 0.5) * cong)
        # crowding lengthens dwell in peaks
        t += DWELL_PER_STOP.get(mode, 0.0) * seg["stops_count"] * (1.0 + cong)
    t *= float(np.exp(rng.normal(0.0, NOISE_SIGMA.get(mode, 0.08))))
    return max(1.0, t)


def generate_travel_time_dataset(output_path=None, samples_per_segment=4, random_state=42):
    rng = np.random.default_rng(random_state)
    segs = build_segments()

    # departure hours: mostly 05:00-23:00, a few night trips
    hours = np.arange(24)
    hour_w = np.where((hours >= 5) & (hours <= 22), 1.0, 0.15)
    hour_w = hour_w / hour_w.sum()

    records = []
    for seg in segs.to_dict("records"):
        for _ in range(samples_per_segment):
            day_type = "weekend" if rng.random() < 2 / 7 else "weekday"
            hour = int(rng.choice(hours, p=hour_w))
            records.append({
                **seg,
                "departure_hour": hour,
                "time_of_day": time_of_day_bucket(hour, day_type),
                "day_type": day_type,
                "travel_time_min": round(simulate_observed_time(seg, hour, day_type, rng), 2),
            })

    df = pd.DataFrame(records)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Dataset generated: {len(df)} samples from {len(segs)} ride segments -> {out}")
    return df


if __name__ == "__main__":
    generate_travel_time_dataset(output_path=Path(__file__).parent / "data" / "travel_time_dataset.csv")
