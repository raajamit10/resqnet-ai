[README (1).md](https://github.com/user-attachments/files/33911174/README.1.md)
# ResQNet AI — Multi-Agent Emergency Response Coordinator

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Built for **Fetch-A-Thon 3.0** (ASI:One Agent Challenge — From Intent to Action).

## Agents

| Agent | Role | Address | Agentverse profile |
|---|---|---|---|
| **ResQNet Emergency Coordinator** (`@resqnet-emergency-co`) | Orchestrator, talks to users in ASI:One | `agent1qdwn3q3ejl3uct3whqdzhvcchagapkslrrc58l84rlqkmvlvlrquj0u27fk` | https://agentverse.ai/agents/details/agent1qdwn3q3ejl3uct3whqdzhvcchagapkslrrc58l84rlqkmvlvlrquj0u27fk/profile |
| **ResQNet Fire Agent** | Fire-station search, fire tender count, fire advice | `agent1qf7572xcq62m5jwqu7wqa43znskqm3a8qayxre06ggej4vf3nma2k07x4kh` | https://agentverse.ai/agents/details/agent1qf7572xcq62m5jwqu7wqa43znskqm3a8qayxre06ggej4vf3nma2k07x4kh/profile |
| **ResQNet Medical Agent** | Hospital search, ambulance count, first-aid advice | `agent1qd7gf6fjqfzfhv2arsu4rnnymqhjd9qau58uz55jy3spw0x78a63vekdecz` | https://agentverse.ai/agents/details/agent1qd7gf6fjqfzfhv2arsu4rnnymqhjd9qau58uz55jy3spw0x78a63vekdecz/profile |
| **ResQNet Police Agent** | Police-station search, traffic/scene-safety advice | `agent1q0zthsypd7dct7jw0lg5ecyqfhdwj29q0h75tmj2sdwltqyvfxkeg3juqgc` | https://agentverse.ai/agents/details/agent1q0zthsypd7dct7jw0lg5ecyqfhdwj29q0h75tmj2sdwltqyvfxkeg3juqgc/profile |

**ASI:One shared chat:** [Chat URL](https://asi1.ai/chat/efd83439-5dd0-4993-bfeb-0dad50e18b96?from=profile&handle=resqnet-emergency-co)
**Demo video:** [VIDEO URL](https://youtu.be/aXAkuacDyPI)

## Problem

In an emergency, people panic and lose critical minutes figuring out which services they need, where the nearest hospital, fire station or police station is, and what to tell the emergency operator. If a responder gets stuck in traffic, there is no simple way to see who else nearby could respond instead.

## Target users

Bystanders, students, drivers and residents who witness or face a fire, road accident, medical emergency or crime and need fast, clear next steps.

## What the system does (outcome)

From one natural-language message inside ASI:One:

1. The **Orchestrator** uses the **ASI1 LLM** to understand the emergency: type, location, severity, injured people and which services are needed.
2. If the location is missing or cannot be found, it **asks for clarification** and remembers the original report (Agent Storage).
3. It **geocodes** the location with OpenStreetMap (Nominatim, with Photon as backup; results outside the named city are rejected).
4. It **dispatches tasks to specialist agents** (Fire, Medical, Police) over a custom uAgents protocol.
5. Each **specialist agent** finds the nearest real facilities from OpenStreetMap, **ranks them by real road drive time** (OSRM shortest-path routing on the OSM road graph), decides **how many units** the incident needs (e.g. 2 ambulances for a high-severity incident with injured people) and adds service-specific advice.
6. The Orchestrator **combines all responses** into one plan: assigned units with ETAs, one-tap **Google Maps routes with live traffic**, safety tips and a ready-to-read **112 operator script**.
7. **Reassignment**: if the user replies "ambulance stuck", the stuck unit is released and the next fastest unit is assigned with a new route.

## Architecture

```
ASI:One chat
   │  Agent Chat Protocol (ACP)
   ▼
ResQNet Orchestrator (hosted on Agentverse)
   ├─ ASI1 LLM: incident & intent analysis (keyword fallback)
   ├─ Geocoding: Nominatim → Photon (city-validated)
   │
   ├─ FacilityRequest ──► 🚒 Fire Agent    ┐  each: OSM facility search (Photon / Overpass),
   ├─ FacilityRequest ──► 🏥 Medical Agent ├─ OSRM road-time ranking, units needed, advice
   ├─ FacilityRequest ──► 🚓 Police Agent  ┘
   │◄── FacilityResponse (all, or timeout after 20 s → orchestrator does the lookup itself)
   │
   └─ Combined plan → assignment → "stuck" → reassignment
```

### Agent-to-agent protocol (`resqnet-dispatch` v0.1.0)

```python
class FacilityRequest(Model):
    request_id: str; service: str; lat: float; lon: float
    emergency_type: str; severity: str; injured: str

class FacilityResponse(Model):
    request_id: str; service: str; ok: bool; facilities_json: str
    units_needed: int; advice: str; error: str
```

## Reliability and safety

- If the ASI1 call fails, a keyword-based parser keeps the system working.
- If a specialist agent does not answer within 20 seconds, the Orchestrator performs that lookup itself, so the user always gets a plan.
- Every external call has a backup source and fails gracefully with a clear message.
- Every fact is labelled with its source. ETAs are marked as estimates. Responder units are clearly marked **[SIMULATED]**.
- The system never claims to contact or dispatch any service, and always tells users to call 112 first.

## Files

| File | Deployed as |
|---|---|
| `agent.py` | ResQNet Emergency Coordinator (Orchestrator) |
| `fire_agent.py` | ResQNet Fire Agent |
| `medical_agent.py` | ResQNet Medical Agent |
| `police_agent.py` | ResQNet Police Agent |

The three specialist files share the same code; only the `SERVICE` setting differs.

## How to test

1. Open the ASI:One shared chat link above, or open the Coordinator's Agentverse profile and click **Chat with Agent**.
2. Send:
   - `Car accident near Roorkee railway station, two people badly injured`
   - then `ambulance stuck`
3. More examples:
   - `Fire in a shop near Roorkee railway station`
   - `Someone collapsed and is not responding` (the agent will ask for the location)

## Run it yourself

1. On https://agentverse.ai create **four** Hosted Agents.
2. Paste `fire_agent.py`, `medical_agent.py` and `police_agent.py` into three of them. Run them and copy their addresses.
3. Paste `agent.py` into the fourth, put the three addresses into `SPECIALISTS`, and run it. No API key is needed; Agentverse provides the ASI1 key to hosted agents.
4. Click **Chat with Agent** on the Coordinator to open it in ASI:One.

## Tech stack

uAgents · Agent Chat Protocol · Agentverse Hosted Agents · ASI1 LLM · OpenStreetMap (Nominatim, Photon, Overpass) · OSRM routing · Google Maps directions · Python `requests`, `openai`

## Limitations and future work

- Responder units are simulated, based at real facilities. Real deployment needs authorised integration with 108/112 control rooms for live vehicle GPS and availability.
- OSM coverage varies: some cities have no fire or police stations mapped, and the agent says so clearly.
- OSRM drive times do not include live traffic (Google Maps links do).
- Planned: ASI Interactive Cards (location confirmation, one-tap "unit stuck" reassignment), live traffic in ranking, and a Payment Protocol model for private ambulance networks.

> **Disclaimer:** ResQNet AI is a hackathon prototype and decision-support tool. In a real emergency, call 112 immediately.
