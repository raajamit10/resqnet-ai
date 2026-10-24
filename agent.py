"""
ResQNet AI - Orchestrator Agent (Fetch-A-Thon 3.0)

Flow: user report in ASI:One (Agent Chat Protocol)
-> ASI1 LLM extracts the incident -> OSM geocodes the location
-> Orchestrator dispatches FacilityRequests to the Fire / Medical / Police specialist agents
-> each specialist finds real facilities, ranks them by OSRM road time, decides units + advice
-> Orchestrator combines the responses (falls back to its own lookup if a specialist times out)
-> response plan with assigned units, routes, safety tips and a 112 script.

PROTOTYPE: nothing is dispatched. Facility open status / vehicles are NOT verified.
"""
import json
import math
import os
import re
import time
from datetime import datetime
from uuid import uuid4

import requests
from openai import OpenAI
from uagents import Agent, Context, Model, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

agent = Agent()
protocol = Protocol(spec=chat_protocol_spec)

# Agentverse injects these into Hosted Agents automatically
client = OpenAI(base_url=os.environ["ASI1_BASE_URL"], api_key=os.environ["ASI1_API_KEY"])

HEADERS = {"User-Agent": "ResQNet-AI/0.1 (Fetch-A-Thon 3.0 hackathon prototype)"}
SEARCH_RADIUS_M = 10000
# Fire stations are sparse in smaller cities, so search further for them
RADIUS_M = {"fire_station": 25000, "hospital": 10000, "police": 10000}

AMENITY = {"fire": "fire_station", "medical": "hospital", "police": "police"}
LABEL = {"fire": "🚒 Fire stations", "medical": "🏥 Hospitals", "police": "🚓 Police"}

UNIT_PREFIX = {"fire": "FIRE", "medical": "AMB", "police": "POL"}
FACILITY_WORD = {"fire": "fire station", "medical": "hospital", "police": "police station"}
UNIT_NAME = {"fire": "Fire tender", "medical": "Ambulance", "police": "Police unit"}
AVG_SPEED_KMPH = 30   # rough city speed for ETA estimates
ROAD_FACTOR = 1.4     # road distance is usually ~1.4x straight-line

# ---- Specialist agents: paste each hosted specialist agent's address here ----
SPECIALISTS = {
    "fire": "agent1qf7572xcq62m5jwqu7wqa43znskqm3a8qayxre06ggej4vf3nma2k07x4kh",
    "medical": "agent1qd7gf6fjqfzfhv2arsu4rnnymqhjd9qau58uz55jy3spw0x78a63vekdecz",
    "police": "agent1q0zthsypd7dct7jw0lg5ecyqfhdwj29q0h75tmj2sdwltqyvfxkeg3juqgc",
}
SPECIALIST_NAME = {"fire": "🚒 Fire Agent", "medical": "🏥 Medical Agent", "police": "🚓 Police Agent"}
JOB_TIMEOUT_S = 20


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

SAFETY = "⚠️ **Real emergency? Call 112 now.** ResQNet is a prototype and does not dispatch anyone."

TIPS = {
    "fire": "Get everyone out, stay low under smoke, do not use lifts, do not go back inside.",
    "medical": "Keep the person still and breathing freely; do not give food or water; stay with them.",
    "accident": "Move away from traffic; do not move injured people unless there is fire or other danger.",
    "crime": "Get to a safe place first; do not confront anyone.",
}

PARSE_PROMPT = """You extract details from an emergency report.
Reply with ONLY one JSON object, no markdown, no extra text:
{"emergency_type": "fire|medical|accident|crime|other|unclear",
 "location": "the place exactly as the user wrote it (area, street, landmark, city), or \\"\\"",
 "severity": "low|medium|high|unknown",
 "injured_people": true or false or null,
 "services": subset of ["fire","medical","police"],
 "summary": "one short sentence"}
Rules: fires need fire and medical; road accidents need medical and police;
crimes need police; never invent or guess a location."""


def make_reply(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )


def keyword_parse(text: str) -> dict:
    """Fallback if the LLM call fails, so the agent still works."""
    t = text.lower()
    if any(w in t for w in ["fire", "smoke", "burning", "aag"]):
        etype, services = "fire", ["fire", "medical"]
    elif any(w in t for w in ["accident", "crash", "collision", "hit"]):
        etype, services = "accident", ["medical", "police"]
    elif any(w in t for w in ["theft", "robbery", "attack", "fight"]):
        etype, services = "crime", ["police"]
    else:
        etype, services = "medical", ["medical"]
    m = re.search(r"location:\s*(.+)$", text, flags=re.I | re.M)
    return {
        "emergency_type": etype,
        "location": m.group(1).strip() if m else "",
        "severity": "unknown",
        "injured_people": None,
        "services": services,
        "summary": text[:120],
        "source": "keyword fallback",
    }


def parse_incident(text: str) -> dict:
    try:
        r = client.chat.completions.create(
            model="asi1",
            messages=[
                {"role": "system", "content": PARSE_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=400,
        )
        raw = str(r.choices[0].message.content).strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        data = json.loads(raw)
        data["source"] = "ASI1"
        return data
    except Exception:
        return keyword_parse(text)


EMERGENCY_WORDS = ["fire", "smoke", "burning", "accident", "crash", "collapsed", "injured",
                   "bleeding", "unconscious", "theft", "robbery", "attack", "emergency", "help"]


def looks_like_new_report(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in EMERGENCY_WORDS)


def regex_location(text: str) -> str:
    """Last-resort location extraction, e.g. '... near Clock Tower, Roorkee'."""
    m = re.search(r"location:\s*(.+)$", text, flags=re.I | re.M)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(?:near|opposite|behind|next to)\s+([^.\n]+)", text, flags=re.I)
    if not m:
        m = re.search(r"\b(?:at|in|on)\s+([A-Z][^.\n]*)", text)
    return m.group(1).strip() if m else ""


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


FILLER_WORDS = r"\b(near|behind|opposite|opp\.?|next to|in front of|beside|around|close to)\b"


def geocode(place: str):
    """Try the exact text, then without filler words, then broader areas (approximate)."""
    cleaned = re.sub(FILLER_WORDS, " ", place, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    candidates = [place, cleaned] + [", ".join(parts[i:]) for i in range(1, len(parts))]

    # If the user named a city (last comma part), reject matches outside it
    city = parts[-1].lower() if len(parts) > 1 else ""

    def valid(h):
        if h and city and city not in h[2].lower():
            print(f"[geocode] rejected '{h[2]}' (not in '{city}')")
            return None
        return h

    seen = set()
    for i, query in enumerate(candidates):
        if not query or query.lower() in seen:
            continue
        if seen:
            time.sleep(1)  # Nominatim usage policy: max 1 request per second
        seen.add(query.lower())
        hit = valid(nominatim_lookup(query)) or valid(photon_lookup(query))
        if hit:
            lat, lon, name = hit
            if i >= 2:
                name += f" — APPROXIMATE (only matched \"{query}\"; exact spot not found)"
            return lat, lon, name
    return None


def nominatim_lookup(query: str):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "countrycodes": "in"},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        results = r.json()
        if results:
            top = results[0]
            return float(top["lat"]), float(top["lon"]), top.get("display_name", query)
    except Exception as e:
        print(f"[geocode] Nominatim failed for '{query}': {e}")
    return None


def photon_lookup(query: str):
    """Backup OSM-based geocoder in case Nominatim fails or has no match."""
    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": query, "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        feats = r.json().get("features", [])
        if feats:
            lon, lat = feats[0]["geometry"]["coordinates"][:2]
            props = feats[0].get("properties", {})
            name = ", ".join(
                str(props[k]) for k in ("name", "street", "city", "state", "country") if props.get(k)
            )
            return float(lat), float(lon), name or query
    except Exception as e:
        print(f"[geocode] Photon failed for '{query}': {e}")
    return None


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


def distance_text(f: dict) -> str:
    if f.get("road_min") is not None:
        return f"~{f['road_km']:.1f} km by road, ~{f['road_min']} min drive _[OSRM road route, no live traffic]_"
    return (f"~{f['km'] * ROAD_FACTOR:.1f} km by road, ETA ~{eta_minutes(f['km'])} min "
            "_[estimate from straight-line distance]_")


def directions_link(lat, lon, flat, flon) -> str:
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={lat},{lon}&destination={flat},{flon}"
    )


def eta_minutes(km: float) -> int:
    return max(1, round(km * ROAD_FACTOR / AVG_SPEED_KMPH * 60))


def assign_next(case: dict, svc: str) -> list:
    """Assign the nearest available unit for a service. Returns message lines."""
    for u in case["units"].get(svc, []):
        if u["status"] == "available":
            u["status"] = "assigned"
            link = directions_link(u["lat"], u["lon"], case["lat"], case["lon"])
            return [
                f"- {UNIT_NAME[svc]} **{u['id']}** from {u['base']} → incident · "
                f"{distance_text(u)} · [Best route (live traffic in Google Maps)]({link})"
            ]
    if not case["units"].get(svc):
        return [f"- No {FACILITY_WORD[svc]} found within "
                f"{RADIUS_M[AMENITY[svc]] // 1000} km in our data, so no {UNIT_NAME[svc].lower()} "
                "could be assigned. **Call 112 now.**"]
    return [f"- No more {UNIT_NAME[svc].lower()}s available nearby in our data. **Call 112 now.**"]


STUCK_WORDS = ["stuck", "delayed", "late", "blocked", "jam", "reassign", "not coming", "breakdown"]
SERVICE_WORDS = {
    "fire": ["fire", "tender", "brigade"],
    "police": ["police", "cop", "pol"],
    "medical": ["ambulance", "amb", "medical", "hospital"],
}


def handle_reassign(case: dict, text: str):
    t = text.lower()
    if not any(w in t for w in STUCK_WORDS):
        return None
    if not any(case["units"].values()):
        return (f"{SAFETY}\n\nNo responder units were found for this incident "
                "(the facility lookup failed or found nothing nearby). **Call 112 now.**")
    svcs = [s for s, words in SERVICE_WORDS.items() if any(w in t for w in words) and case["units"].get(s)]
    if not svcs:
        svcs = [s for s in case["units"] if case["units"][s]][:1]
    lines = [SAFETY, "", "**🔁 Reassignment [SIMULATED]**"]
    for svc in svcs:
        for u in case["units"][svc]:
            if u["status"] == "assigned":
                u["status"] = "stuck"
                lines.append(f"- {u['id']} marked as stuck/delayed and released.")
                break
        lines += assign_next(case, svc)
    lines += ["", f"Incident location: {case['place']}"]
    return "\n".join(lines)


def injured_text(inc: dict) -> str:
    return {True: "yes", False: "no"}.get(inc.get("injured_people"), "not sure")


def local_result(job: dict, svc: str) -> dict:
    """Fallback: orchestrator does the lookup itself if a specialist did not answer."""
    try:
        facs = nearby(job["lat"], job["lon"], AMENITY[svc])
        return {"ok": True, "facilities": facs, "units_needed": 1, "advice": "",
                "by": f"{SPECIALIST_NAME[svc]}: no reply, orchestrator did the lookup itself", "secs": None}
    except Exception as e:
        return {"ok": False, "facilities": [], "units_needed": 1, "advice": "",
                "by": f"{SPECIALIST_NAME[svc]}: no reply, orchestrator fallback failed", "secs": None, "error": str(e)}


def build_plan(job: dict):
    inc = job["inc"]
    lat, lon, place_name = job["lat"], job["lon"], job["place"]
    etype = inc.get("emergency_type", "unclear")
    severity = inc.get("severity", "unknown")
    results = job["results"]

    lines = [
        SAFETY,
        "",
        f"**Incident:** {etype} · severity: {severity}  _[analysis: {inc.get('source')}]_",
        f"**Summary:** {inc.get('summary', '')}",
        f"**Location matched:** {place_name}  _[OSM geocode — please confirm this is correct]_",
        "",
        "**🤝 Agents coordinated**",
    ]
    for svc in job["services"]:
        r = results[svc]
        took = f" in {r['secs']:.1f}s" if r.get("secs") is not None else ""
        lines.append(f"- {r['by']}{took}")
    lines.append("")

    case = {"lat": lat, "lon": lon, "place": place_name, "units": {}}

    for svc in job["services"]:
        r = results[svc]
        lines.append(f"**{LABEL[svc]} nearby**")
        if not r["ok"]:
            lines += ["- Lookup failed right now. Rely on 112 for this service.", ""]
            continue
        facilities = r["facilities"]
        if not facilities:
            lines.append(f"- None found within {RADIUS_M[AMENITY[svc]] // 1000} km in OpenStreetMap data.")
        # simulated responder units, one based at each real facility (fastest first)
        case["units"][svc] = [
            {"id": f"{UNIT_PREFIX[svc]}-{n + 1}", "base": f["name"], "lat": f["lat"],
             "lon": f["lon"], "km": f["km"], "road_km": f.get("road_km"),
             "road_min": f.get("road_min"), "status": "available"}
            for n, f in enumerate(facilities)
        ]
        for f in facilities[:2]:
            phone = f" · 📞 {f['phone']} _[OSM, unverified]_" if f.get("phone") else ""
            lines.append(
                f"- {f['name']} — {distance_text(f)} · "
                f"[Directions]({directions_link(lat, lon, f['lat'], f['lon'])}){phone}"
            )
        if r.get("advice"):
            lines.append(f"- _{SPECIALIST_NAME[svc]} advice:_ {r['advice']}")
        lines.append("")

    assignment_lines = []
    for svc in job["services"]:
        if svc not in case["units"]:
            continue
        for _ in range(max(1, results[svc].get("units_needed", 1))):
            assignment_lines += assign_next(case, svc)
            if not any(u["status"] == "available" for u in case["units"][svc]):
                break
    if assignment_lines:
        lines += ["**🚦 Responder assignment [SIMULATED]**"] + assignment_lines + [
            "_If a unit is stuck or delayed, reply e.g. \"ambulance stuck\" and I will reassign the next fastest unit._",
            "",
        ]

    if etype in TIPS:
        lines += [f"**Immediate safety:** {TIPS[etype]}", ""]

    loc_text = (inc.get("location") or "").strip()
    if not re.match(r"(?i)(near|at|in|on|opposite|behind|next to|beside)\b", loc_text):
        loc_text = f"at {loc_text}"
    lines += [
        "**📞 Read this to the 112 operator:**",
        f"\"There is {'an' if etype[:1] in 'aeiou' else 'a'} {etype} emergency {loc_text} ({place_name}). "
        f"It looks {severity} severity. Injured people: {injured_text(inc)}. "
        "My name is ____ and my phone number is ____.\"",
        "",
        "_Data notes: facilities come from OpenStreetMap. Opening status, beds, vehicles "
        "and live traffic are NOT verified by this agent. Units are ranked by OSRM road drive time when available. "
        "No service has been contacted. Responder units are simulated, based at real facilities._",
    ]
    return "\n".join(lines), case


# ---------------- Job tracking (multi-agent coordination) ----------------

def active_jobs(ctx: Context) -> list:
    raw = ctx.storage.get("active-jobs")
    return json.loads(raw) if raw else []


def save_active_jobs(ctx: Context, jobs: list):
    ctx.storage.set("active-jobs", json.dumps(jobs))


def specialist_configured(svc: str) -> bool:
    addr = SPECIALISTS.get(svc, "")
    return addr.startswith("agent1") and "PASTE" not in addr


async def finalize_job(ctx: Context, job: dict):
    """Fill any missing services with the local fallback, then answer the user."""
    if job.get("done"):
        return
    job["done"] = True
    ctx.storage.set(f"job-{job['id']}", json.dumps(job))
    save_active_jobs(ctx, [j for j in active_jobs(ctx) if j != job["id"]])
    for svc in job["services"]:
        if svc not in job["results"]:
            ctx.logger.info(f"[orchestrator] {svc} specialist missing, using fallback")
            job["results"][svc] = local_result(job, svc)
    plan, case = build_plan(job)
    ctx.storage.set(f"case-{job['user']}", json.dumps(case))
    await ctx.send(job["user"], make_reply(plan))


async def start_job(ctx: Context, sender: str, inc: dict, geo):
    lat, lon, place_name = geo
    services = [s for s in inc.get("services", []) if s in AMENITY] or ["medical", "police"]
    job = {"id": str(uuid4()), "user": sender, "inc": inc, "lat": lat, "lon": lon,
           "place": place_name, "services": services, "results": {},
           "sent_at": {}, "created": time.time(), "done": False}

    for svc in services:
        if specialist_configured(svc):
            job["sent_at"][svc] = time.time()
        else:
            job["results"][svc] = local_result(job, svc)
    ctx.storage.set(f"job-{job['id']}", json.dumps(job))

    if all(svc in job["results"] for svc in services):
        await finalize_job(ctx, job)
        return

    save_active_jobs(ctx, active_jobs(ctx) + [job["id"]])
    for svc in services:
        if svc in job["sent_at"]:
            ctx.logger.info(f"[orchestrator] dispatching {svc} request to {SPECIALISTS[svc]}")
            await ctx.send(SPECIALISTS[svc], FacilityRequest(
                request_id=job["id"], service=svc, lat=lat, lon=lon,
                emergency_type=inc.get("emergency_type", "unclear"),
                severity=inc.get("severity", "unknown"), injured=injured_text(inc),
            ))


@dispatch_protocol.on_message(FacilityResponse)
async def handle_specialist(ctx: Context, sender: str, resp: FacilityResponse):
    raw = ctx.storage.get(f"job-{resp.request_id}")
    if not raw:
        return
    job = json.loads(raw)
    if job.get("done"):
        return
    sent = job["sent_at"].get(resp.service, job["created"])
    job["results"][resp.service] = {
        "ok": resp.ok,
        "facilities": json.loads(resp.facilities_json or "[]"),
        "units_needed": resp.units_needed,
        "advice": resp.advice,
        "by": f"{SPECIALIST_NAME.get(resp.service, resp.service)} ({sender[:14]}…)",
        "secs": time.time() - sent,
        "error": resp.error,
    }
    ctx.logger.info(f"[orchestrator] got {resp.service} response for {resp.request_id}")
    ctx.storage.set(f"job-{job['id']}", json.dumps(job))
    if all(svc in job["results"] for svc in job["services"]):
        await finalize_job(ctx, job)


@agent.on_interval(period=5.0)
async def check_timeouts(ctx: Context):
    for job_id in active_jobs(ctx):
        raw = ctx.storage.get(f"job-{job_id}")
        if not raw:
            save_active_jobs(ctx, [j for j in active_jobs(ctx) if j != job_id])
            continue
        job = json.loads(raw)
        if not job.get("done") and time.time() - job["created"] > JOB_TIMEOUT_S:
            ctx.logger.info(f"[orchestrator] job {job_id} timed out, finalizing with fallback")
            await finalize_job(ctx, job)


@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id),
    )
    text = msg.text()
    if not text:
        return
    text = re.sub(r"@\S+", "", text).strip()  # drop @agent mentions
    if not text:
        return
    ctx.logger.info(f"Report received: {text}")

    # Follow-up on an active incident: "ambulance stuck" -> reassign next nearest unit
    case_key = f"case-{sender}"
    saved = ctx.storage.get(case_key)
    if saved:
        case = json.loads(saved)
        reply = handle_reassign(case, text)
        if reply:
            ctx.storage.set(case_key, json.dumps(case))
            await ctx.send(sender, make_reply(reply))
            return

    # "ambulance stuck" with no active incident: don't treat it as a location
    if any(w in text.lower() for w in STUCK_WORDS) and not saved:
        await ctx.send(sender, make_reply(
            f"{SAFETY}\n\nThere is no active incident to reassign yet. "
            "Please describe the emergency and its location first."
        ))
        return

    # Merge with an earlier report ONLY if this message looks like a location-only reply.
    key = f"pending-{sender}"
    pending = ctx.storage.get(key)
    if pending:
        ctx.storage.set(key, "")
        if not looks_like_new_report(text):
            text = f"{pending}\nLocation: {text}"
            ctx.logger.info("Merged with pending report")

    inc = parse_incident(text)
    location = (inc.get("location") or "").strip() or regex_location(text)
    inc["location"] = location
    ctx.logger.info(f"Parsed incident: {inc} | location used: '{location}'")

    geo = None
    if location:
        try:
            geo = geocode(location)
        except Exception:
            ctx.logger.exception("Geocoding failed")
        ctx.logger.info(f"Geocode result: {geo}")

    if not geo:
        ctx.storage.set(key, text)
        missing = f' I could not find "{location}" on the map.' if location else ""
        await ctx.send(sender, make_reply(
            f"{SAFETY}\n\nI need a precise location to find the nearest help.{missing} "
            "Please reply with the area or street, a nearby landmark, and the city "
            "(e.g. \"Civil Lines, near Clock Tower, Roorkee\")."
        ))
        return

    await start_job(ctx, sender, inc, geo)


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)
agent.include(dispatch_protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
