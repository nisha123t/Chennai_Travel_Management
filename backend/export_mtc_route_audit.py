"""Export official MTC route stages with coordinate-match provenance."""
import argparse
import csv
import json
import re
import ssl
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "mtc_route_stage_audit.csv"
MTC_URL = "https://mtcbus.tn.gov.in/Home/routewiseinfo"
PHOTON_URL = "https://photon.komoot.io/api/"
PHOTON_CACHE_PATH = DATA_DIR / "photon_geocode_cache.json"
USER_AGENT = "ChennaiRoutePlannerDataset/1.0"
MTC_UPDATE_DATE = "2026-08-07"
PHOTON_DELAY_SECONDS = 1.1
STAGE_ALIASES = {
    "mgrcentral": "Chennai Central",
    "mgrkoyambedu": "Koyambedu (CMBT)",
    "royapurambs": "Royapuram",
    "guindyrs": "Guindy",
    "tbsanatorium": "Tambaram Sanatorium",
    "vallalarnagar": "Vallalar Nagar (Mint)",
    "thiagarayanagar": "T. Nagar",
    "stthomasmountpo": "St. Thomas Mount",
    "dms": "AG-DMS",
    "kilambakkambs": "Kilambakkam Bus Terminus",
    "kundrathurbs": "Kundrathur",
    "ambatturot": "Ambattur OT Bus Terminus",
    "tondaiayrpet": "Tondiarpet",
    "vyasarpadi": "Vyasarpadi Jeeva",
    "madhavaram": "Madhavaram MIBT",
    "thiruvotriyur": "Tiruvottiyur",
    "guduvanchery": "Guduvancheri",
    "urapakkam": "Urappakkam",
    "porur": "Porur Junction",
    "tambaramwest": "Tambaram",
    "tambarameast": "Tambaram East Bus Stand",
    "adyarbs": "Adyar",
    "pallavaramrs": "Pallavaram",
}


def verified_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    if hasattr(ssl, "enum_certificates"):
        for certificate, encoding, _ in ssl.enum_certificates("CA"):
            if encoding == "x509_asn":
                context.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(certificate))
    return context


CONTEXT = verified_context()


def fetch(url: str, timeout: int = 20) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, context=CONTEXT, timeout=timeout) as response:
        return response.read()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(char for char in value if not unicodedata.combining(char))
    return "".join(char for char in value if char.isalnum())


class RoutePageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_select = False
        self.in_option = False
        self.in_route_list = False
        self.in_stage = False
        self.option_value = ""
        self.option_text: list[str] = []
        self.stage_text: list[str] = []
        self.route_codes: list[str] = []
        self.stages: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select" and attrs.get("id") == "selroute":
            self.in_select = True
        elif tag == "option" and self.in_select:
            self.in_option = True
            self.option_value = attrs.get("value", "")
            self.option_text = []
        elif tag == "ul" and "route" in attrs.get("class", "").split():
            self.in_route_list = True
        elif tag == "li" and self.in_route_list:
            self.in_stage = True
            self.stage_text = []

    def handle_data(self, data):
        if self.in_option:
            self.option_text.append(data)
        if self.in_stage:
            self.stage_text.append(data)

    def handle_endtag(self, tag):
        if tag == "option" and self.in_option:
            text = " ".join(self.option_text).strip()
            if text and text != "--Route--":
                self.route_codes.append(self.option_value.strip())
            self.in_option = False
        elif tag == "select":
            self.in_select = False
        elif tag == "ul" and self.in_route_list:
            self.in_route_list = False
        elif tag == "li" and self.in_stage:
            text = " ".join(self.stage_text).strip()
            text = re.sub(r"^\s*\d+\s*", "", text).strip()
            if text:
                self.stages.append(text)
            self.in_stage = False


def read_existing_stops() -> dict[str, dict]:
    with (DATA_DIR / "stops.csv").open(encoding="utf-8-sig", newline="") as source:
        return {normalize(row["stop_name"]): row for row in csv.DictReader(source)}


def read_route_codes() -> list[str]:
    parser = RoutePageParser()
    parser.feed(fetch(MTC_URL).decode("utf-8", "replace"))
    if not parser.route_codes:
        raise RuntimeError("The official MTC page did not return route codes")
    return list(dict.fromkeys(parser.route_codes))


def get_route_stages(code: str) -> list[str]:
    parser = RoutePageParser()
    query = urlencode({"selroute": code, "submit": ""})
    last_error = None
    for _ in range(2):
        try:
            parser.feed(fetch(f"{MTC_URL}?{query}").decode("utf-8", "replace"))
            break
        except Exception as error:
            last_error = error
    else:
        raise RuntimeError(f"Could not fetch MTC route {code}: {last_error}")
    return parser.stages


def coordinate_match(stage: str, existing: dict[str, dict]) -> dict:
    key = normalize(stage)
    if key in existing:
        stop = existing[key]
        return {
            "stop_id": stop["stop_id"], "latitude": stop["latitude"],
            "longitude": stop["longitude"], "source": "existing_project_stop",
            "status": "matched_existing",
        }
    alias_target = STAGE_ALIASES.get(key)
    if alias_target and normalize(alias_target) in existing:
        stop = existing[normalize(alias_target)]
        return {
            "stop_id": stop["stop_id"], "latitude": stop["latitude"],
            "longitude": stop["longitude"], "source": "existing_project_stop_alias",
            "status": "matched_existing_alias",
        }

    return {"stop_id": "", "latitude": "", "longitude": "",
            "source": "", "status": "needs_photon_lookup"}


def photon_geocode_stage(stage: str) -> dict:
    query = urlencode({
        "q": f"{stage}, Chennai, Tamil Nadu, India",
        "lat": 12.8,
        "lon": 80.2,
        "limit": 5,
        "lang": "en",
    })
    response = json.loads(fetch(f"{PHOTON_URL}?{query}", timeout=8))
    candidates = []
    for feature in response.get("features", []):
        properties = feature.get("properties", {})
        longitude, latitude = feature.get("geometry", {}).get("coordinates", [None, None])
        if latitude is None or longitude is None:
            continue
        candidates.append({
            "name": properties.get("name", ""),
            "city": properties.get("city", ""),
            "county": properties.get("county", ""),
            "type": properties.get("type", ""),
            "latitude": latitude,
            "longitude": longitude,
            "osm_id": properties.get("osm_id", ""),
        })

    exact = [candidate for candidate in candidates
             if normalize(candidate["name"]) == normalize(stage)]
    if len(exact) == 1:
        chosen = exact[0]
        return {"latitude": chosen["latitude"], "longitude": chosen["longitude"],
                "source": "Photon / OpenStreetMap", "status": "photon_exact_candidate_review",
                "photon_candidates": json.dumps(candidates, ensure_ascii=False)}
    status = "photon_ambiguous_candidates" if candidates else "photon_no_results"
    return {"latitude": "", "longitude": "", "source": "Photon / OpenStreetMap",
            "status": status, "photon_candidates": json.dumps(candidates, ensure_ascii=False)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photon-geocode", action="store_true",
                        help="Geocode a small batch of unresolved stages using Photon / OpenStreetMap")
    parser.add_argument("--photon-limit", type=int, default=50,
                        help="Maximum uncached stage lookups per run (default: 50)")
    parser.add_argument("--refresh-routes", action="store_true",
                        help="Fetch all MTC route pages again instead of reusing the existing audit")
    args = parser.parse_args()
    if args.photon_limit < 1:
        parser.error("--photon-limit must be at least 1")

    existing = read_existing_stops()
    routes = []
    if OUTPUT_PATH.exists() and not args.refresh_routes:
        with OUTPUT_PATH.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                routes.append((row["route_code"], int(row["sequence_no"]), row["stage_name"]))
    else:
        codes = read_route_codes()
        stages_by_route = {}
        failures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            pending = {executor.submit(get_route_stages, code): code for code in codes}
            for future in as_completed(pending):
                code = pending[future]
                try:
                    stages_by_route[code] = future.result()
                except Exception as error:
                    failures.append(str(error))
        if failures:
            raise RuntimeError("Official MTC route fetch failed: " + "; ".join(failures[:5]))
        for code in codes:
            stages = stages_by_route[code]
            if not stages:
                raise RuntimeError(f"No official stages returned for MTC route {code}")
            routes.extend((code, sequence, stage) for sequence, stage in enumerate(stages, 1))

    codes = list(dict.fromkeys(code for code, _, _ in routes))

    matches = {stage: coordinate_match(stage, existing)
               for _, _, stage in routes}
    cache = json.loads(PHOTON_CACHE_PATH.read_text(encoding="utf-8")) if PHOTON_CACHE_PATH.exists() else {}
    unresolved = [stage for stage, match in matches.items() if not match["latitude"]]
    geocoded_count = 0
    if args.photon_geocode:
        to_lookup = [stage for stage in unresolved if stage not in cache][:args.photon_limit]
        for stage in to_lookup:
            try:
                result = photon_geocode_stage(stage)
            except Exception as error:
                result = {"latitude": "", "longitude": "", "source": "Photon / OpenStreetMap",
                          "status": f"photon_error_{type(error).__name__}", "photon_candidates": "[]"}
                print(f"Photon lookup failed for {stage}: {type(error).__name__}")
            else:
                cache[stage] = result
                PHOTON_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            matches[stage].update(result)
            geocoded_count += 1
            time.sleep(PHOTON_DELAY_SECONDS)
    for stage in unresolved:
        if stage in cache:
            matches[stage].update(cache[stage])

    maps_base = "https://www.google.com/maps/search/?api=1&"
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=[
            "route_code", "sequence_no", "stage_name", "mapped_stop_id",
            "latitude", "longitude", "coordinate_source", "match_status",
            "photon_candidates",
            "google_maps_search_url", "source_url", "source_updated",
        ])
        writer.writeheader()
        for code, sequence, stage in routes:
            match = matches[stage]
            query = f"{stage}, Chennai, Tamil Nadu, India"
            writer.writerow({
                "route_code": code,
                "sequence_no": sequence,
                "stage_name": stage,
                "mapped_stop_id": match["stop_id"],
                "latitude": match["latitude"],
                "longitude": match["longitude"],
                "coordinate_source": match["source"],
                "match_status": match["status"],
                "photon_candidates": match.get("photon_candidates", ""),
                "google_maps_search_url": maps_base + urlencode({"query": query}),
                "source_url": MTC_URL,
                "source_updated": MTC_UPDATE_DATE,
            })

    counts = defaultdict(int)
    for match in matches.values():
        counts[match["status"]] += 1
    print(f"Official MTC route codes: {len(codes)}")
    print(f"Route-stage rows: {len(routes)}")
    print(f"Unique stage names: {len(matches)}")
    if args.photon_geocode:
        print(f"Photon lookups this run: {geocoded_count} (cache: {PHOTON_CACHE_PATH})")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print(f"Audit spreadsheet: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()