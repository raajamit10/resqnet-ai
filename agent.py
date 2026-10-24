"""
ResQNet AI - Emergency Response Coordinator (Fetch-A-Thon 3.0)

Flow: user report -> ASI1 LLM extracts incident -> OSM Nominatim geocodes location
-> OSM Overpass finds nearest fire stations / hospitals / police -> agent returns
a response plan with directions links and a ready-to-read 112 script.

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
from uagents import Agent, Context, Protocol
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

AMENITY = {"fire": "fire_station", "medical": "hospital", "police": "police"}
LABEL = {"fire": "🚒 Fire stations", "medical": "🏥 Hospitals", "police": "🚓 Police"}

UNIT_PREFIX = {"fire": "FIRE", "medical": "AMB", "police": "POL"}
UNIT_NAME = {"fire": "Fire tender", "medical": "Ambulance", "police": "Police unit"}
AVG_SPEED_KMPH = 30   # rough city speed for ETA estimates
ROAD_FACTOR = 1.4     # road distance is usually ~1.4x straight-line

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
    m = re.search(r"\b(?:near|at|in|on|opposite|behind)\s+([A-Z][^.\n]*)", text)
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


OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
PHOTON_QUERY = {"fire_station": "fire station", "hospital": "hospital", "police": "police station"}


def overpass_nearby(lat, lon, amenity):
    query = (
        f'[out:json][timeout:20];('
        f'node["amenity"="{amenity}"](around:{SEARCH_RADIUS_M},{lat},{lon});'
        f'way["amenity"="{amenity}"](around:{SEARCH_RADIUS_M},{lat},{lon});'
        f');out center 40;'
    )
    last_err = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=20)
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
    r = requests.get(
        "https://photon.komoot.io/api/",
        params={"q": PHOTON_QUERY[amenity], "lat": lat, "lon": lon,
                "osm_tag": f"amenity:{amenity}", "limit": 15},
        headers=HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    found = []
    for feat in r.json().get("features", []):
        flon, flat = feat["geometry"]["coordinates"][:2]
        km = haversine_km(lat, lon, flat, flon)
        if km > SEARCH_RADIUS_M / 1000:
            continue
        props = feat.get("properties", {})
        found.append({
            "name": props.get("name", "Unnamed facility"),
            "lat": flat, "lon": flon, "phone": None, "km": km,
        })
    return found


def nearby(lat: float, lon: float, amenity: str, limit: int = 4) -> list:
    """Real OSM facilities: Overpass first, Photon as backup."""
    try:
        found = overpass_nearby(lat, lon, amenity)
    except Exception:
        found = photon_nearby(lat, lon, amenity)  # raises if this fails too
    return sorted(found, key=lambda f: f["km"])[:limit]


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
                f"~{u['km'] * ROAD_FACTOR:.1f} km by road, ETA ~{eta_minutes(u['km'])} min _[estimate]_ · "
                f"[Best route (live traffic in Google Maps)]({link})"
            ]
    if not case["units"].get(svc):
        return [f"- No {LABEL[svc].split(' ', 1)[1].lower()[:-1]} found within "
                f"{SEARCH_RADIUS_M // 1000} km in our data, so no {UNIT_NAME[svc].lower()} "
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
        lines += assign_next(case, svc)
    lines += ["", f"Incident location: {case['place']}"]
    return "\n".join(lines)


def build_plan(ctx: Context, inc: dict, geo) -> str:
    lat, lon, place_name = geo
    etype = inc.get("emergency_type", "unclear")
    severity = inc.get("severity", "unknown")
    services = [s for s in inc.get("services", []) if s in AMENITY] or ["medical", "police"]

    lines = [
        SAFETY,
        "",
        f"**Incident:** {etype} · severity: {severity}  _[analysis: {inc.get('source')}]_",
        f"**Summary:** {inc.get('summary', '')}",
        f"**Location matched:** {place_name}  _[OSM geocode — please confirm this is correct]_",
        "",
    ]

    case = {"lat": lat, "lon": lon, "place": place_name, "units": {}}

    for svc in services:
        lines.append(f"**{LABEL[svc]} nearby**")
        try:
            facilities = nearby(lat, lon, AMENITY[svc])
        except Exception:
            ctx.logger.exception(f"Facility lookup failed for {svc}")
            lines.append("- Lookup failed right now. Rely on 112 for this service.")
            lines.append("")
            continue
        if not facilities:
            lines.append(f"- None found within {SEARCH_RADIUS_M // 1000} km in OpenStreetMap data.")
        # simulated responder units, one based at each real facility (nearest first)
        case["units"][svc] = [
            {"id": f"{UNIT_PREFIX[svc]}-{n + 1}", "base": f["name"], "lat": f["lat"],
             "lon": f["lon"], "km": f["km"], "status": "available"}
            for n, f in enumerate(facilities)
        ]
        for f in facilities[:2]:
            phone = f" · 📞 {f['phone']} _[OSM, unverified]_" if f["phone"] else ""
            lines.append(
                f"- {f['name']} — ~{f['km']:.1f} km straight-line _[estimate]_ · "
                f"[Directions]({directions_link(lat, lon, f['lat'], f['lon'])}){phone}"
            )
        lines.append("")

    assignment_lines = []
    for svc in case["units"]:
        assignment_lines += assign_next(case, svc)
    if assignment_lines:
        lines += ["**🚦 Responder assignment [SIMULATED]**"] + assignment_lines + [
            "_If a unit is stuck or delayed, reply e.g. \"ambulance stuck\" and I will reassign the next nearest unit._",
            "",
        ]

    if etype in TIPS:
        lines += [f"**Immediate safety:** {TIPS[etype]}", ""]

    injured = {True: "yes", False: "no"}.get(inc.get("injured_people"), "not sure")
    loc_text = (inc.get("location") or "").strip()
    if not re.match(r"(?i)(near|at|in|on|opposite|behind|next to|beside)\b", loc_text):
        loc_text = f"at {loc_text}"
    inc["location"] = loc_text
    lines += [
        "**📞 Read this to the 112 operator:**",
        f"\"There is a {etype} emergency {inc.get('location')} ({place_name}). "
        f"It looks {severity} severity. Injured people: {injured}. "
        "My name is ____ and my phone number is ____.\"",
        "",
        "_Data notes: facilities come from OpenStreetMap. Opening status, beds, vehicles "
        "and live traffic are NOT verified. Road distance will be longer than straight-line. "
        "No service has been contacted. Responder units are simulated, based at real facilities._",
    ]
    return "\n".join(lines), case


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

    plan, case = build_plan(ctx, inc, geo)
    ctx.storage.set(case_key, json.dumps(case))
    await ctx.send(sender, make_reply(plan))


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
