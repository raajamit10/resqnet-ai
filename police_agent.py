"""
ResQNet AI - Specialist Agent (Fire / Medical / Police)

Receives a FacilityRequest from the ResQNet Orchestrator, finds the nearest real
facilities for its service (OpenStreetMap), ranks them by real road drive time (OSRM),
decides how many units the incident needs, adds service-specific advice, and replies
with a FacilityResponse.

>>> Set SERVICE below to "fire", "medical" or "police" for each hosted agent. <<<
"""
import json
import math
import time

import requests
from uagents import Agent, Context, Model, Protocol

SERVICE = "police"   # ready for the Police Agent - do not change

agent = Agent()

HEADERS = {"User-Agent": "ResQNet-AI/0.1 (Fetch-A-Thon 3.0 hackathon prototype)"}
RADIUS_M = {"fire_station": 25000, "hospital": 10000, "police": 10000}
AMENITY = {"fire": "fire_station", "medical": "hospital", "police": "police"}
AVG_SPEED_KMPH = 30
ROAD_FACTOR = 1.4


# ---- Message models: must be IDENTICAL in the orchestrator and all specialists ----
class FacilityRequest(Model):
    request_id: str
    service: str
    lat: float
    lon: float
    emergency_type: str
    severity: str
    injured: str


class FacilityResponse(Model):
    request_id: str
    service: str
    ok: bool
    facilities_json: str
    units_needed: int
    advice: str
    error: str


dispatch_protocol = Protocol(name="resqnet-dispatch", version="0.1.0")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


FILLER_WORDS = r"\b(near|behind|opposite|opp\.?|next to|in front of|beside|around|close to)\b"


OVERPASS_URLS = ["https://overpass-api.de/api/interpreter"]
NAME_QUERIES = {
    "fire_station": ["fire station", "agnishaman"],
    "hospital": ["hospital"],
    "police": ["police station", "kotwali", "thana", "police chowki"],
}
NAME_KEYWORDS = {
    "fire_station": ["fire", "agnishaman"],
    "hospital": ["hospital", "medical", "clinic"],
    "police": ["police", "kotwali", "thana", "chowki"],
}
PHOTON_QUERIES = {
    "fire_station": ["fire station", "fire"],
    "hospital": ["hospital"],
    "police": ["police station", "police"],
}
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"


def overpass_nearby(lat, lon, amenity):
    radius = RADIUS_M[amenity]
    query = (
        f'[out:json][timeout:20];('
        f'node["amenity"="{amenity}"](around:{radius},{lat},{lon});'
        f'way["amenity"="{amenity}"](around:{radius},{lat},{lon});'
        f');out center 40;'
    )
    last_err = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=5)
            r.raise_for_status()
            found = []
            for el in r.json().get("elements", []):
                flat = el.get("lat", el.get("center", {}).get("lat"))
                flon = el.get("lon", el.get("center", {}).get("lon"))
                if flat is None or flon is None:
                    continue
                tags = el.get("tags", {})
                found.append({
                    "name": tags.get("name", "Unnamed facility"),
                    "lat": flat, "lon": flon,
                    "phone": tags.get("phone") or tags.get("contact:phone"),
                    "km": haversine_km(lat, lon, flat, flon),
                })
            return found
        except Exception as e:
            last_err = e
            print(f"[facilities] Overpass {url} failed: {e}")
    raise RuntimeError(f"All Overpass servers failed: {last_err}")


def photon_nearby(lat, lon, amenity):
    found, seen, ok = [], set(), False
    for q in PHOTON_QUERIES[amenity]:
        try:
            r = requests.get(
                "https://photon.komoot.io/api/",
                params={"q": q, "lat": lat, "lon": lon,
                        "osm_tag": f"amenity:{amenity}", "limit": 15},
                headers=HEADERS,
                timeout=10,
            )
            r.raise_for_status()
            ok = True
        except Exception as e:
            print(f"[facilities] Photon '{q}' failed: {e}")
            continue
        for feat in r.json().get("features", []):
            flon, flat = feat["geometry"]["coordinates"][:2]
            key = (round(flat, 4), round(flon, 4))
            if key in seen:
                continue
            seen.add(key)
            km = haversine_km(lat, lon, flat, flon)
            if km > RADIUS_M[amenity] / 1000:
                continue
            props = feat.get("properties", {})
            found.append({
                "name": props.get("name", "Unnamed facility"),
                "lat": flat, "lon": flon, "phone": None, "km": km,
            })
    if not found:
        # Many Indian facilities are mapped without the amenity tag: search by name instead
        for q in NAME_QUERIES[amenity]:
            try:
                r = requests.get(
                    "https://photon.komoot.io/api/",
                    params={"q": q, "lat": lat, "lon": lon, "limit": 20},
                    headers=HEADERS,
                    timeout=8,
                )
                r.raise_for_status()
                ok = True
            except Exception as e:
                print(f"[facilities] Photon name search '{q}' failed: {e}")
                continue
            for feat in r.json().get("features", []):
                props = feat.get("properties", {})
                name = props.get("name") or ""
                if not any(k in name.lower() for k in NAME_KEYWORDS[amenity]):
                    continue
                flon, flat = feat["geometry"]["coordinates"][:2]
                key = (round(flat, 4), round(flon, 4))
                if key in seen:
                    continue
                seen.add(key)
                km = haversine_km(lat, lon, flat, flon)
                if km <= RADIUS_M[amenity] / 1000:
                    found.append({"name": name, "lat": flat, "lon": flon, "phone": None, "km": km})
            if found:
                break
    if not ok:
        raise RuntimeError("Photon facility search failed")
    return found


def add_road_times(lat, lon, facilities):
    """Real road distance + drive time from each facility to the incident (OSRM shortest path).
    OSRM computes fastest routes on the OSM road graph (Dijkstra-family algorithm).
    No live traffic. Leaves values empty if OSRM is unreachable."""
    if not facilities:
        return
    coords = f"{lon},{lat};" + ";".join(f"{f['lon']},{f['lat']}" for f in facilities)
    sources = ";".join(str(i + 1) for i in range(len(facilities)))
    try:
        r = requests.get(
            OSRM_TABLE_URL + coords,
            params={"sources": sources, "destinations": "0", "annotations": "duration,distance"},
            headers=HEADERS,
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        for f, dur, dist in zip(facilities, data["durations"], data["distances"]):
            if dur[0] is not None and dist[0] is not None:
                f["road_sec"] = dur[0]
                f["road_min"] = max(1, round(dur[0] / 60))
                f["road_km"] = dist[0] / 1000
    except Exception as e:
        print(f"[routing] OSRM failed: {e}")


def nearby(lat: float, lon: float, amenity: str, limit: int = 4) -> list:
    """Real OSM facilities (Photon first - fast from hosted agents; Overpass as backup),
    ranked by real road drive time when available."""
    photon_ok = True
    try:
        found = photon_nearby(lat, lon, amenity)
    except Exception as e:
        print(f"[facilities] Photon failed, trying Overpass: {e}")
        found, photon_ok = [], False
    if not found:
        try:
            found = overpass_nearby(lat, lon, amenity)
        except Exception as e:
            print(f"[facilities] Overpass also failed: {e}")
            if not photon_ok:
                raise  # both sources down -> caller reports "lookup failed"
    candidates = sorted(found, key=lambda f: f["km"])[:8]
    add_road_times(lat, lon, candidates)
    candidates.sort(key=lambda f: f.get("road_sec", f["km"] * ROAD_FACTOR / AVG_SPEED_KMPH * 3600))
    return candidates[:limit]



def decide_units(req: FacilityRequest) -> int:
    """Service-specific decision: how many units this incident needs."""
    high = req.severity == "high"
    if SERVICE == "fire":
        return 2 if high else 1
    if SERVICE == "medical":
        return 2 if (high and req.injured == "yes") else 1
    return 1  # police


def advice_for(req: FacilityRequest) -> str:
    if SERVICE == "fire":
        tip = "Keep the road clear for fire tenders; switch off electricity and gas only if it is safe."
        if req.severity == "high":
            tip += " High severity: 2 fire tenders recommended."
        return tip
    if SERVICE == "medical":
        if req.injured == "yes":
            return "Injured people reported: check breathing first, press on heavy bleeding, do not move them unless in danger."
        return "Keep a medical team on standby in case of smoke inhalation or injuries."
    if req.emergency_type in ("accident", "fire"):
        return "Traffic control needed: keep a lane clear for responders and keep onlookers back."
    return "Secure the area and keep people at a safe distance."


@dispatch_protocol.on_message(FacilityRequest, replies=FacilityResponse)
async def handle_request(ctx: Context, sender: str, req: FacilityRequest):
    ctx.logger.info(f"[{SERVICE}] request {req.request_id} from {sender}")
    started = time.time()
    try:
        facilities = nearby(req.lat, req.lon, AMENITY[SERVICE])
        resp = FacilityResponse(
            request_id=req.request_id, service=SERVICE, ok=True,
            facilities_json=json.dumps(facilities), units_needed=decide_units(req),
            advice=advice_for(req), error="",
        )
    except Exception as e:
        ctx.logger.exception(f"[{SERVICE}] lookup failed")
        resp = FacilityResponse(
            request_id=req.request_id, service=SERVICE, ok=False,
            facilities_json="[]", units_needed=decide_units(req),
            advice=advice_for(req), error=str(e)[:200],
        )
    ctx.logger.info(f"[{SERVICE}] replying in {time.time() - started:.1f}s")
    await ctx.send(sender, resp)


agent.include(dispatch_protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
