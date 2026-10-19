"""
Dataset builder for Machine Learning Travel Time Prediction models.
Extracts trip and leg data from backend/data/connections.csv, augmenting
with realistic Chennai urban traffic conditions (peak vs off-peak hours,
weather/congestion factors for road modes vs dedicated track for rail/metro).
"""
import random
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def generate_travel_time_dataset(output_path=None, samples_per_connection=4, random_state=42):
    """
    Builds a robust, realistic training dataset for travel time prediction.
    Uses real connections data as baseline and models real-world Chennai transit variations:
    - Metro & Suburban Rail: Highly reliable, minor dwell delays in peak hours (+5% to +15%)
    - MTC Bus: Sensitive to road traffic (+25% to +55% during morning/evening peak hours)
    - Auto & Cab: Highly sensitive to peak congestion (+30% to +65% during peak hours)
    - Walking: Fixed human walking pace (4.0 - 5.0 km/h) with minimal variance
    """
    random.seed(random_state)
    np.random.seed(random_state)

    conns_df = pd.read_csv(DATA_DIR / "connections.csv")
    
    time_of_day_options = [
        ("morning_peak", 1.35, "weekday"),   # 08:00 - 11:00
        ("evening_peak", 1.40, "weekday"),   # 17:00 - 20:30
        ("afternoon", 1.05, "weekday"),      # 12:00 - 16:00
        ("off_peak", 0.95, "weekday"),       # 21:00 - 06:00
        ("weekend_day", 1.10, "weekend"),    # Saturday / Sunday Day
        ("weekend_night", 0.90, "weekend")   # Saturday / Sunday Late Night
    ]

    records = []

    for _, conn in conns_df.iterrows():
        base_dist = float(conn["distance_km"])
        base_time = float(conn["estimated_time_min"])
        base_fare = float(conn["fare_inr"])
        ttype = str(conn["transport_type"]).lower()

        for _ in range(samples_per_connection):
            tod, traffic_mult, day_type = random.choice(time_of_day_options)
            
            # Mode-specific realistic traffic sensitivity
            if ttype == "walk":
                # Walking speed: ~4.5 km/h +/- small human variation
                speed_kmh = np.random.normal(4.5, 0.4)
                speed_kmh = max(3.0, min(6.0, speed_kmh))
                sim_time = (base_dist / speed_kmh) * 60.0
                stops_count = 1
                sim_fare = 0.0

            elif ttype == "metro":
                # Dedicated elevated/underground grade-separated track
                # Dwell time increases slightly with passenger crowding during peak hours
                crowd_factor = 1.05 if "peak" in tod else 1.0
                dwell_jitter = np.random.normal(0.0, 0.5)
                sim_time = max(2.0, base_time * crowd_factor + dwell_jitter)
                stops_count = 1
                sim_fare = base_fare

            elif ttype in ("rail", "suburban_rail", "mrts"):
                # Dedicated rail track; small signal / platform delay in peak hours
                rail_factor = 1.08 if "peak" in tod else 1.0
                rail_jitter = np.random.normal(0.0, 0.8)
                sim_time = max(2.0, base_time * rail_factor + rail_jitter)
                stops_count = 1
                sim_fare = base_fare

            elif ttype == "bus":
                # Surface road traffic on Mount Road, GST Road, OMR, Poonamallee High Road
                bus_mult = traffic_mult + np.random.normal(0.0, 0.08)
                bus_mult = max(0.9, bus_mult)
                sim_time = max(3.0, base_time * bus_mult)
                stops_count = random.choice([1, 2, 3])
                sim_fare = base_fare

            elif ttype == "auto":
                # Mixed road traffic, agile lane navigation
                auto_mult = (traffic_mult * 0.95) + np.random.normal(0.0, 0.07)
                auto_mult = max(0.85, auto_mult)
                sim_time = max(3.0, base_time * auto_mult)
                stops_count = 1
                sim_fare = base_fare

            elif ttype == "cab":
                # Arterial road/highway traffic; peak hour slowdowns
                cab_mult = traffic_mult + np.random.normal(0.0, 0.06)
                cab_mult = max(0.85, cab_mult)
                sim_time = max(8.0, base_time * cab_mult)
                stops_count = 1
                sim_fare = base_fare

            else:
                sim_time = max(2.0, base_time * traffic_mult)
                stops_count = 1
                sim_fare = base_fare

            records.append({
                "from_stop": conn["from_stop"],
                "to_stop": conn["to_stop"],
                "transport_type": ttype,
                "distance_km": round(base_dist, 2),
                "fare_inr": round(sim_fare, 2),
                "stops_count": stops_count,
                "time_of_day": tod,
                "day_type": day_type,
                "travel_time_min": round(sim_time, 2)
            })

    df = pd.DataFrame(records)
    
    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        print(f"Dataset generated with {len(df)} samples at {out_file}")

    return df

if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "travel_time_dataset.csv"
    generate_travel_time_dataset(output_path=out)
