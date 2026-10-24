# ResQNet AI — Emergency Response Coordinator

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Built for **Fetch-A-Thon 3.0** (ASI:One Agent Challenge — From Intent to Action).

## Agent details

| Field | Value |
|---|---|
| Agent name | ResQNet Emergency Coordinator |
| Handle | `@resqnet-emergency-co` |
| Agent address | `agent1qdwn3q3ejl3uct3whqdzhvcchagapkslrrc58l84rlqkmvlvlrquj0u27fk` |
| Agentverse profile | [https://agentverse.ai/agents/details/agent1qdwn3q3ejl3uct3whqdzhvcchagapkslrrc58l84rlqkmvlvlrquj0u27fk/profile](https://agentverse.ai/agents/details/agent1qdwn3q3ejl3uct3whqdzhvcchagapkslrrc58l84rlqkmvlvlrquj0u27fk/profile)|
| ASI:One shared chat | PASTE_SHARED_CHAT_URL |
| Demo video | PASTE_VIDEO_URL |

## Problem

In an emergency, people panic and lose critical minutes figuring out which services they need, where the nearest hospital, fire station or police station is, and what to tell the emergency operator. If a responder gets stuck in traffic, there is no simple way to see who else nearby could respond instead.

## Target users

Bystanders, students, drivers and residents who witness or face a fire, road accident, medical emergency or crime and need fast, clear next steps.

## What the agent does (outcome)

From one natural-language message inside ASI:One, ResQNet AI:

1. **Understands the emergency with the ASI1 LLM**: type, location, severity, injured people, and which services are needed (fire, medical, police).
2. **Asks for clarification** if the location is missing or cannot be found, and remembers the original report using Agent Storage.
3. **Geocodes the location** with OpenStreetMap (Nominatim, with Photon as a backup). Results outside the city the user named are rejected, and approximate matches are clearly marked.
4. **Finds the nearest real facilities** (hospitals, fire stations, police stations) within 10 km from OpenStreetMap data (Overpass API, with Photon as a backup).
5. **Assigns the nearest responder unit** for each needed service (simulated units based at the real facilities found), with an estimated ETA and a **one-tap Google Maps route** using live traffic.
6. **Reassigns on failure**: if the user replies "ambulance stuck" (or "fire tender delayed"), the stuck unit is released and the next nearest unit is assigned with a new route.
7. Gives **immediate safety tips** and a ready-to-read **script for the 112 operator**.

## Example conversation

```
User:   There is a fire in a building near Clock Tower, Roorkee
Agent:  Incident: fire · severity: high
        Hospitals nearby: Vardhman Hospital (~0.4 km), Mission Hospital (~0.4 km)
        Responder assignment [SIMULATED]: AMB-1 from Vardhman Hospital, ETA ~1 min, route link
        Safety tips + "Read this to the 112 operator" script

User:   ambulance stuck
Agent:  AMB-1 released → AMB-2 assigned from the next nearest hospital, new route link
```

## Architecture

```
ASI:One chat
   │  Agent Chat Protocol (ACP)
   ▼
ResQNet agent (uAgents, hosted on Agentverse)
   ├─ Intent & incident analysis ── ASI1 LLM (keyword fallback if the LLM fails)
   ├─ Clarification memory ───────── Agent Storage
   ├─ Geocoding ──────────────────── Nominatim → Photon (city-validated)
   ├─ Facility discovery ─────────── Overpass (2 servers) → Photon
   ├─ Responder assignment ───────── nearest available unit, ETA estimate, Google Maps route
   └─ Reassignment ───────────────── "stuck/delayed" → next nearest unit
```

## Reliability and safety

- If the ASI1 call fails, a keyword-based parser keeps the agent working.
- Every external lookup has a backup service, and failures are reported clearly instead of crashing.
- Every fact is labelled with its source. Distances and ETAs are marked as estimates. Responder units are clearly marked **[SIMULATED]**.
- The agent never claims to contact or dispatch any service, and always tells users to call 112 first.

## Tech stack

- **uAgents** + **Agent Chat Protocol (ACP)**, hosted on **Agentverse**
- **ASI1 LLM** for intent and incident understanding
- **OpenStreetMap**: Nominatim and Photon (geocoding), Overpass and Photon (facility discovery)
- Google Maps directions links for live-traffic routing
- Python: `requests`, `openai` client

## How to test

1. Open the ASI:One shared chat link above, or open the Agentverse profile and click **Chat with Agent**.
2. Send:
   - `There is a fire in a building near Clock Tower, Roorkee`
   - then `ambulance stuck`
3. Other examples:
   - `Bike and car accident near Roorkee bus stand, one person bleeding`
   - `Someone collapsed and is not responding` (the agent will ask for the location; reply with a place)

## Run it yourself

1. Sign in to https://agentverse.ai and launch a new Hosted Agent.
2. Paste `agent.py` into the **Build** tab and click **Run**. No API key is needed; Agentverse provides the ASI1 key to hosted agents.
3. Click **Chat with Agent** to open it in ASI:One.

## Limitations and future work

- Responder units are simulated. Real deployment would need authorised integration with 108/112 control rooms for live vehicle GPS and availability.
- Units are currently ranked by straight-line distance. Next step: build the OSM road graph and use **Dijkstra / A\*** to rank units by actual road travel time.
- Facility open status, bed availability and live traffic are not verified by the agent itself.
- Planned: ASI Interactive Cards for location confirmation and facility selection, and separate specialist agents (fire, medical, police) for multi-agent coordination.

> **Disclaimer:** ResQNet AI is a hackathon prototype and decision-support tool. In a real emergency, call 112 immediately.
