"""Add MTC adjacent-stage segments whose endpoints already map to project stops."""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AUDIT_PATH = DATA_DIR / "mtc_route_stage_audit.csv"
SOURCE_URL = "https://mtcbus.tn.gov.in/Home/routewiseinfo"


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def haversine_km(a: dict, b: dict) -> float:
    lat1, lon1 = math.radians(float(a["latitude"])), math.radians(float(a["longitude"]))
    lat2, lon2 = math.radians(float(b["latitude"])), math.radians(float(b["longitude"]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def bus_fare(distance_km: float) -> float:
    stage = math.ceil(distance_km / 2.0)
    if stage <= 1:
        return 5.0
    if stage == 2:
        return 6.0
    if stage == 3:
        return 7.0
    if stage == 4:
        return 8.0
    if stage == 5:
        return 9.0
    if stage == 6:
        return 10.0
    if stage == 7:
        return 11.0
    if stage == 8:
        return 12.0
    if stage == 9:
        return 13.0
    if stage == 10:
        return 14.0
    if stage <= 12:
        return 15.0
    if stage <= 14:
        return 16.0
    if stage <= 16:
        return 17.0
    if stage <= 18:
        return 18.0
    if stage <= 21:
        return 19.0
    if stage <= 24:
        return 21.0
    if stage <= 27:
        return 23.0
    return 25.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write safe segments to the three network CSVs")
    args = parser.parse_args()

    audit_fields, audit = read_csv(AUDIT_PATH)
    stop_fields, stops = read_csv(DATA_DIR / "stops.csv")
    route_fields, routes = read_csv(DATA_DIR / "routes.csv")
    sequence_fields, sequences = read_csv(DATA_DIR / "stop_sequences.csv")
    connection_fields, connections = read_csv(DATA_DIR / "connections.csv")

    stops_by_id = {row["stop_id"]: row for row in stops}
    existing_route_ids = {row["route_id"] for row in routes}
    existing_connection_ids = {row["connection_id"] for row in connections}
    last_connection_number = max(
        (int(match.group(1)) for value in existing_connection_ids
         if (match := re.fullmatch(r"C(\d+)", value))),
        default=0,
    )

    by_route = defaultdict(list)
    for row in audit:
        if row["match_status"] in ("matched_existing", "matched_existing_alias") \
                and row["mapped_stop_id"] in stops_by_id:
            by_route[row["route_code"]].append(row)

    new_routes, new_sequences, new_connections = [], [], []
    for code, stage_rows in by_route.items():
        for first, second in zip(stage_rows, stage_rows[1:]):
            first_sequence = int(first["sequence_no"])
            second_sequence = int(second["sequence_no"])
            if second_sequence != first_sequence + 1:
                continue

            start_id, end_id = first["mapped_stop_id"], second["mapped_stop_id"]
            if start_id == end_id:
                continue

            safe_code = re.sub(r"[^A-Za-z0-9]+", "_", code).strip("_")
            route_id = f"MTC_{safe_code}_S{first_sequence:03d}"
            if route_id in existing_route_ids:
                continue

            start, end = stops_by_id[start_id], stops_by_id[end_id]
            distance = max(0.3, round(haversine_km(start, end) * 1.22, 2))
            travel_time = max(2.0, round(distance * 2.8 + 0.8, 1))
            fare = bus_fare(distance)
            url = f"{SOURCE_URL}?{urlencode({'selroute': code, 'submit': ''})}"
            name = f"MTC {code} (partial stages {first_sequence}-{second_sequence})"

            new_routes.append({
                "route_id": route_id, "transport_type": "bus", "route_name": name,
                "source_stop": start_id, "destination_stop": end_id,
                "fare_inr": int(fare), "source_url": url, "data_date": "2026-08",
                "status": "official_stage_partial_segment",
            })
            new_sequences.extend((
                {"route_id": route_id, "sequence_no": 1, "stop_id": start_id},
                {"route_id": route_id, "sequence_no": 2, "stop_id": end_id},
            ))
            last_connection_number += 1
            connection_id = f"C{last_connection_number:04d}"
            while connection_id in existing_connection_ids:
                last_connection_number += 1
                connection_id = f"C{last_connection_number:04d}"
            existing_connection_ids.add(connection_id)
            new_connections.append({
                "connection_id": connection_id, "from_stop": start_id, "to_stop": end_id,
                "transport_type": "bus", "route_id": route_id, "distance_km": distance,
                "estimated_time_min": travel_time, "fare_inr": fare,
                "source_url": url, "data_date": "2026-08",
                "status": "official_stages_geo_derived_time_fare",
            })
            existing_route_ids.add(route_id)

    print(f"Safe adjacent MTC segments: {len(new_routes)}")
    print("Will update routes.csv, stop_sequences.csv, and connections.csv; stops.csv is unchanged.")
    if not args.apply:
        print("Dry run only. Re-run with --apply to write the segments.")
        return

    write_csv(DATA_DIR / "routes.csv", route_fields, routes + new_routes)
    write_csv(DATA_DIR / "stop_sequences.csv", sequence_fields, sequences + new_sequences)
    write_csv(DATA_DIR / "connections.csv", connection_fields, connections + new_connections)
    print(f"Added {len(new_routes)} partial MTC route segments.")


if __name__ == "__main__":
    main()