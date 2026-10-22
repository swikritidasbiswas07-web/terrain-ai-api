import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from math import atan2, cos, radians, sin, sqrt
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# TERRAIN CONFIGURATION
# ============================================================

app = FastAPI(
    title="Terrain Outdoor Activity API",
    version="2.0.0",
    description="Evidence-grounded outdoor recommendations using live geospatial and weather data with local Qwen AI.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

USER_AGENT = "TerrainOutdoorAssistant/2.0 (outdoor activity research project)"

OLLAMA_URL = os.getenv(
    "TERRAIN_OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate",
)
OLLAMA_MODEL = os.getenv(
    "TERRAIN_OLLAMA_MODEL",
    "qwen2.5:1.5b",
)
OLLAMA_ENABLED = os.getenv(
    "TERRAIN_OLLAMA_ENABLED",
    "true",
).lower() not in {"0", "false", "no", "off"}


class RecommendationQuery(BaseModel):
    query: str = "Find a good place for an outdoor activity"
    activity: str | None = "Running"
    location: str | None = None
    coords: dict[str, float] | None = None
    radius_km: float = Field(default=10, ge=0.5, le=30)
    start: str | None = None
    end: str | None = None
    considerations: str | None = None


# ============================================================
# UTILITIES
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance between two coordinates."""
    earth_radius_km = 6371.0

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))

    return 2 * earth_radius_km * atan2(
        sqrt(a),
        sqrt(1 - a),
    )


def at(data: dict, key: str, index: int) -> Any:
    values = data.get(key) or []
    return values[index] if index < len(values) else None


def source_record(source_id, name, url, retrieved_at):
    return {
        "id": source_id,
        "name": name,
        "url": url,
        "retrieved_at": retrieved_at,
        "observed_at": None,
    }


def weather_description(code):
    """Convert Open-Meteo weather codes to readable descriptions."""
    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    try:
        return descriptions.get(int(code), f"Weather code {code}")
    except (TypeError, ValueError):
        return None


# ============================================================
# API STATUS
# ============================================================

@app.get("/")
def root():
    return {
        "app": "Terrain",
        "status": "online",
        "version": app.version,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
    }


@app.get("/model-status")
async def model_status():
    """Check whether the local Ollama model is available."""
    if not OLLAMA_ENABLED:
        return {
            "enabled": False,
            "available": False,
            "model": OLLAMA_MODEL,
        }

    tags_url = OLLAMA_URL.rsplit("/api/generate", 1)[0] + "/api/tags"

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(tags_url)
            response.raise_for_status()

        names = [
            item.get("name", "")
            for item in response.json().get("models", [])
        ]

        available = any(
            name == OLLAMA_MODEL
            or name.startswith(OLLAMA_MODEL + ":")
            for name in names
        )

        return {
            "enabled": True,
            "available": available,
            "model": OLLAMA_MODEL,
            "installed_models": names,
        }

    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "model": OLLAMA_MODEL,
            "detail": str(exc)[:200],
        }


# ============================================================
# TIME WINDOW INTERPRETATION
# ============================================================

def parse_time_window(request, timezone_name):
    """Interpret explicit times or common phrases such as tomorrow morning."""
    try:
        local_tz = ZoneInfo(timezone_name)
    except Exception:
        local_tz = timezone.utc

    now = datetime.now(local_tz)

    text = " ".join(
        part
        for part in [request.query, request.considerations]
        if part
    ).lower()

    if bool(request.start) != bool(request.end):
        return None, "Provide both start and end times, or leave both blank."

    if request.start and request.end:
        try:
            start_dt = datetime.fromisoformat(request.start)
            end_dt = datetime.fromisoformat(request.end)

            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=local_tz)
            else:
                start_dt = start_dt.astimezone(local_tz)

            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=local_tz)
            else:
                end_dt = end_dt.astimezone(local_tz)

        except ValueError:
            return None, "Could not interpret the supplied start/end dates."

    else:
        target_day = now.date()

        if "day after tomorrow" in text:
            target_day = (now + timedelta(days=2)).date()
        elif "tomorrow" in text:
            target_day = (now + timedelta(days=1)).date()
        elif "weekend" in text:
            days_until_saturday = (5 - now.weekday()) % 7
            target_day = (
                now + timedelta(days=days_until_saturday)
            ).date()

        if any(
            phrase in text
            for phrase in ["morning", "before 8 am", "before 8am"]
        ):
            start_hour = 6
            end_hour = 8 if "before 8" in text else 11
        elif "afternoon" in text:
            start_hour, end_hour = 12, 16
        elif "evening" in text:
            start_hour, end_hour = 16, 20
        elif "night" in text:
            start_hour, end_hour = 19, 22
        else:
            return None, None

        start_dt = datetime.combine(
            target_day,
            datetime.min.time().replace(hour=start_hour),
            tzinfo=local_tz,
        )

        end_dt = datetime.combine(
            target_day,
            datetime.min.time().replace(hour=end_hour),
            tzinfo=local_tz,
        )

    if start_dt >= end_dt:
        return None, "The end time must be later than the start time."

    return {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
    }, None


# ============================================================
# LOCATION GEOCODING
# ============================================================

async def geocode_location(client, request):
    if request.coords:
        lat = request.coords.get("lat")
        lon = request.coords.get("lon")

        if lat is not None and lon is not None:
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise HTTPException(
                    status_code=422,
                    detail="Coordinates are outside valid ranges.",
                )

            return lat, lon, f"Coordinates: {lat}, {lon}"

    if not request.location or not request.location.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a city, place, or valid coordinates.",
        )

    response = await client.get(
        NOMINATIM_URL,
        params={
            "q": request.location.strip(),
            "format": "jsonv2",
            "limit": 1,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )

    response.raise_for_status()
    results = response.json()

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Location not found. Try a city or nearby landmark.",
        )

    result = results[0]

    return (
        float(result["lat"]),
        float(result["lon"]),
        result.get("display_name", request.location.strip()),
    )


# ============================================================
# NEARBY OUTDOOR PLACE DISCOVERY
# ============================================================

async def find_nearby_places(client, lat, lon, radius_km):
    """Use OpenStreetMap Overpass, with Nominatim fallback."""
    radius_m = int(radius_km * 1000)
    last_error = None

    def normalize_overpass(elements):
        found = {}

        for element in elements:
            tags = element.get("tags", {})
            name = tags.get("name")

            if not name:
                continue

            center = element.get("center", {})
            p_lat = element.get("lat", center.get("lat"))
            p_lon = element.get("lon", center.get("lon"))

            if p_lat is None or p_lon is None:
                continue

            p_lat = float(p_lat)
            p_lon = float(p_lon)

            distance = haversine_km(lat, lon, p_lat, p_lon)

            if distance > radius_km:
                continue

            place_id = (
                f"osm-{element.get('type', 'feature')}-"
                f"{element.get('id', '')}"
            )

            found[place_id] = {
                "id": place_id,
                "name": name,
                "type": (
                    tags.get("leisure")
                    or tags.get("sport")
                    or tags.get("natural")
                    or tags.get("landuse")
                    or tags.get("tourism")
                ),
                "lat": p_lat,
                "lon": p_lon,
                "distance_km": round(distance, 2),
                "travel_time_min": None,
            }

        return found

    query = f"""
    [out:json][timeout:25];
    (
      nwr(around:{radius_m},{lat},{lon})["leisure"~"park|garden|pitch|golf_course|playground"];
      nwr(around:{radius_m},{lat},{lon})["landuse"="recreation_ground"];
      nwr(around:{radius_m},{lat},{lon})["natural"~"wood|heath|scrub"];
      nwr(around:{radius_m},{lat},{lon})["tourism"="picnic_site"];
      nwr(around:{radius_m},{lat},{lon})["sport"];
    );
    out center tags;
    """

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = await client.post(
                endpoint,
                content=query,
                headers={"Content-Type": "text/plain"},
                timeout=35,
            )

            response.raise_for_status()

            places_by_id = normalize_overpass(
                response.json().get("elements", [])
            )

            places = sorted(
                places_by_id.values(),
                key=lambda item: item["distance_km"],
            )

            print(f"Overpass succeeded: {len(places)} places found.")
            return places[:30], None

        except (httpx.HTTPError, ValueError) as exc:
            last_error = str(exc)
            print(f"Overpass failed: {last_error}")

    # Fallback searches, bounded to a geographic box around the location.
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / max(
        111.0 * abs(cos(radians(lat))),
        0.01,
    )

    viewbox = (
        f"{max(-180, lon - lon_delta)},"
        f"{min(90, lat + lat_delta)},"
        f"{min(180, lon + lon_delta)},"
        f"{max(-90, lat - lat_delta)}"
    )

    search_terms = [
        "park",
        "garden",
        "golf course",
        "sports field",
        "picnic site",
        "nature reserve",
    ]

    allowed_categories = {
        "leisure", "natural", "tourism", "landuse", "boundary"
    }

    allowed_types = {
        "park", "garden", "golf_course", "pitch", "sports_centre",
        "recreation_ground", "picnic_site", "nature_reserve", "wood",
        "heath", "forest", "national_park", "playground", "track",
        "common", "protected_area",
    }

    fallback_places = {}

    for index, term in enumerate(search_terms):
        if index:
            await asyncio.sleep(1.1)

        try:
            response = await client.get(
                NOMINATIM_URL,
                params={
                    "q": term,
                    "format": "jsonv2",
                    "limit": 20,
                    "viewbox": viewbox,
                    "bounded": 1,
                },
                headers={"User-Agent": USER_AGENT},
                timeout=20,
            )

            response.raise_for_status()

            for result in response.json():
                try:
                    p_lat = float(result["lat"])
                    p_lon = float(result["lon"])
                except (KeyError, TypeError, ValueError):
                    continue

                distance = haversine_km(lat, lon, p_lat, p_lon)

                if distance > radius_km:
                    continue

                category = result.get("category", "")
                feature_type = result.get("type", "")

                if (
                    category not in allowed_categories
                    and feature_type not in allowed_types
                ):
                    continue

                name = result.get("name") or result.get(
                    "display_name", ""
                ).split(",")[0].strip()

                if not name:
                    continue

                osm_type = result.get("osm_type")
                osm_id = result.get("osm_id")

                if osm_type and osm_id:
                    place_id = f"osm-{osm_type}-{osm_id}"
                else:
                    place_id = f"nominatim-{result.get('place_id')}"

                fallback_places[place_id] = {
                    "id": place_id,
                    "name": name,
                    "type": (
                        feature_type
                        or category
                        or term.replace(" ", "_")
                    ),
                    "lat": p_lat,
                    "lon": p_lon,
                    "distance_km": round(distance, 2),
                    "travel_time_min": None,
                }

        except (httpx.HTTPError, ValueError) as exc:
            last_error = str(exc)
            print(f"Nominatim fallback failed for {term}: {last_error}")

    places = sorted(
        fallback_places.values(),
        key=lambda item: item["distance_km"],
    )

    print(f"Nominatim fallback: {len(places)} places found.")

    if places:
        return places[:30], None

    return [], last_error or "No nearby mapped outdoor places were found."


# ============================================================
# WEATHER DATA
# ============================================================

async def get_weather(client, lat, lon):
    """Fetch actual hourly forecast values from Open-Meteo."""
    response = await client.get(
        OPEN_METEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": (
                "temperature_2m,precipitation_probability,"
                "wind_speed_10m,wind_direction_10m,weather_code,is_day"
            ),
            "forecast_days": 3,
            "timezone": "auto",
        },
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    points = []

    for i, timestamp in enumerate(hourly.get("time", [])):
        code = at(hourly, "weather_code", i)
        points.append({
            "time": timestamp,
            "weather": weather_description(code),
            "weather_code": code,
            "temp_c": at(hourly, "temperature_2m", i),
            "precip_prob": at(
                hourly, "precipitation_probability", i
            ),
            "wind_speed_kmh": at(hourly, "wind_speed_10m", i),
            "wind_dir_deg": at(hourly, "wind_direction_10m", i),
            "aqi": None,
            "is_daylight": (
                bool(at(hourly, "is_day", i))
                if at(hourly, "is_day", i) is not None
                else None
            ),
        })

    return points, payload.get("timezone", "UTC")


# ============================================================
# WEATHER WINDOW SELECTION
# ============================================================

def choose_forecast_window(hourly_points, time_window, timezone_name):
    """Choose a lower-risk forecast hour without claiming guaranteed safety."""
    try:
        local_tz = ZoneInfo(timezone_name)
    except Exception:
        local_tz = timezone.utc

    now_local = datetime.now(local_tz).replace(tzinfo=None)
    requested_start = None
    requested_end = None

    if time_window:
        try:
            requested_start = datetime.fromisoformat(
                time_window["start"]
            ).replace(tzinfo=None)

            requested_end = datetime.fromisoformat(
                time_window["end"]
            ).replace(tzinfo=None)
        except (ValueError, KeyError, TypeError):
            pass

    candidates = []

    for point in hourly_points:
        try:
            point_dt = datetime.fromisoformat(
                point["time"]
            ).replace(tzinfo=None)
        except (ValueError, TypeError):
            continue

        if point_dt < now_local:
            continue

        if requested_start and requested_end:
            if not requested_start <= point_dt < requested_end:
                continue
        elif point_dt > now_local + timedelta(hours=48):
            continue

        rain = point.get("precip_prob")
        wind = point.get("wind_speed_kmh")
        temp = point.get("temp_c")
        daylight = point.get("is_daylight")

        risk = 0.0
        risk += min(60.0, float(rain) * 0.6) if rain is not None else 20.0
        risk += (
            min(30.0, max(0.0, float(wind) - 12.0))
            if wind is not None else 8.0
        )

        if temp is None:
            risk += 8.0
        elif temp >= 35:
            risk += 25.0
        elif temp >= 32:
            risk += 12.0
        elif temp <= 5:
            risk += 15.0

        if daylight is False:
            risk += 8.0

        candidates.append((risk, point_dt, point))

    if not candidates:
        return (
            None,
            ["No matching future forecast hour was available."],
            None,
        )

    _, _, best = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )

    cautions = []
    rain = best.get("precip_prob")
    wind = best.get("wind_speed_kmh")
    temp = best.get("temp_c")

    if rain is not None and rain >= 50:
        cautions.append("Forecast precipitation probability is elevated.")

    if wind is not None and wind >= 30:
        cautions.append("Forecast wind speeds are elevated.")

    if temp is not None and temp >= 32:
        cautions.append(
            "Forecast heat may make strenuous outdoor activity uncomfortable."
        )

    if best.get("is_daylight") is False:
        cautions.append("This forecast hour is outside daylight hours.")

    if rain is None or wind is None or temp is None:
        cautions.append("Some weather variables are unavailable.")

    best_dt = datetime.fromisoformat(best["time"])

    return (
        {
            "start": best["time"],
            "end": (best_dt + timedelta(hours=1)).isoformat(),
        },
        cautions,
        best,
    )


# ============================================================
# DETERMINISTIC RANKING
# ============================================================

PREFERRED_TYPES = {
    "running": {"park", "garden", "recreation_ground", "wood", "forest", "track"},
    "jogging": {"park", "garden", "recreation_ground", "wood", "track"},
    "walking": {"park", "garden", "recreation_ground", "wood", "forest", "playground"},
    "cycling": {"park", "wood", "recreation_ground", "track", "cycleway"},
    "golf": {"golf_course"},
    "cricket": {"pitch", "recreation_ground"},
    "football": {"pitch", "recreation_ground"},
    "soccer": {"pitch", "recreation_ground"},
    "tennis": {"pitch", "sports_centre"},
    "hiking": {"wood", "heath", "park", "forest", "nature_reserve", "national_park"},
    "picnic": {"picnic_site", "park", "garden"},
    "outdoor events": {"park", "garden", "recreation_ground"},
    "photography": {"park", "garden", "wood", "heath", "nature_reserve"},
}


def make_rankings(
    places,
    activity,
    hourly_points,
    time_window,
    timezone_name,
    radius_km,
):
    preferred = PREFERRED_TYPES.get(
        (activity or "").lower().strip(),
        set(),
    )

    best_window, weather_cautions, best_weather = choose_forecast_window(
        hourly_points,
        time_window,
        timezone_name,
    )

    scored_places = []

    for place in places:
        feature_type = (place.get("type") or "").lower().strip()
        type_match = feature_type in preferred

        distance = float(place.get("distance_km") or 0)

        proximity_score = max(
            0.0,
            100.0 * (1.0 - distance / max(radius_km, 0.5)),
        )

        activity_score = (
            100.0 if type_match
            else 55.0 if not preferred
            else 35.0
        )

        weather_score = 55.0

        if best_weather:
            rain = best_weather.get("precip_prob")
            wind = best_weather.get("wind_speed_kmh")
            temp = best_weather.get("temp_c")

            penalties = 0.0

            penalties += (
                min(55.0, float(rain) * 0.55)
                if rain is not None else 15.0
            )

            penalties += (
                min(25.0, max(0.0, float(wind) - 10.0))
                if wind is not None else 8.0
            )

            if temp is not None and temp >= 32:
                penalties += min(20.0, (float(temp) - 30.0) * 2.0)

            if best_weather.get("is_daylight") is False:
                penalties += 8.0

            weather_score = max(0.0, 100.0 - penalties)

        # This is a heuristic ranking score, not a probability or safety rating.
        score = round(
            0.35 * proximity_score
            + 0.35 * activity_score
            + 0.30 * weather_score,
            1,
        )

        scored_places.append((
            score,
            place,
            type_match,
        ))

    scored_places.sort(
        key=lambda item: (
            -item[0],
            item[1].get("distance_km", 999),
        )
    )

    rankings = []

    for rank, (score, place, type_match) in enumerate(
        scored_places[:10],
        start=1,
    ):
        distance = place.get("distance_km", 0)

        factors = [
            {
                "label": "Distance",
                "value": f"{distance:.1f} km straight-line",
                "status": (
                    "good" if distance <= 2
                    else "fair" if distance <= 5
                    else "poor"
                ),
            },
            {
                "label": "Activity match",
                "value": (
                    "Mapped feature type matches the activity preference"
                    if type_match
                    else "Specific suitability is not established by mapped feature type"
                ),
                "status": "good" if type_match else "unknown",
            },
        ]

        if best_weather:
            rain = best_weather.get("precip_prob")
            wind = best_weather.get("wind_speed_kmh")

            if rain is None or wind is None:
                weather_status = "unknown"
            elif rain < 25 and wind < 20:
                weather_status = "good"
            elif rain < 50 and wind < 30:
                weather_status = "fair"
            else:
                weather_status = "poor"

            factors.append({
                "label": "Weather",
                "value": (
                    f"{best_weather.get('temp_c')}°C; "
                    f"precipitation chance {rain}%"
                    if rain is not None
                    else "Forecast details are incomplete"
                ),
                "status": weather_status,
            })

        carry = []
        watch_for = list(weather_cautions)

        activity_lower = (activity or "").lower()

        if activity_lower in {
            "running", "jogging", "walking", "hiking", "cycling"
        }:
            carry.extend([
                "Water",
                "Comfortable activity-appropriate footwear",
            ])

        if best_weather:
            if (
                best_weather.get("temp_c") is not None
                and best_weather["temp_c"] >= 30
            ):
                carry.append("Water and sun protection")

            if (
                best_weather.get("precip_prob") is not None
                and best_weather["precip_prob"] >= 40
            ):
                carry.append("Light rain protection")

        if not best_weather:
            watch_for.append(
                "No suitable forecast hour was available for the requested period."
            )

        rankings.append({
            "place_id": place["id"],
            "rank": rank,
            "score": score,
            "best_window": best_window,
            "factors": factors,
            "reasoning": (
                f"{place['name']} is approximately {distance:.1f} km away "
                "in straight-line distance. The score combines proximity, "
                "mapped activity-type match, and the available location-level "
                "forecast. It is a heuristic, not a safety rating or proof "
                "of venue access."
            ),
            "carry": list(dict.fromkeys(carry)),
            "watch_for": list(dict.fromkeys(watch_for)),
            "avoid_when": [],
            "activity_cautions": [
                "Confirm opening hours, access rules, and on-site conditions before visiting."
            ],
            "source_ids": ["osm"] + (["weather"] if hourly_points else []),
        })

    return rankings


# ============================================================
# LOCAL QWEN MODEL INTEGRATION
# ============================================================

async def enrich_rankings_with_model(
    rankings,
    places,
    hourly_points,
    request,
):
    """
    Use Qwen to improve explanations and practical advice.
    Scores and ranks remain calculated by Python.
    """
    if not OLLAMA_ENABLED or not rankings:
        return rankings, False

    place_by_id = {
        place["id"]: place
        for place in places
    }

    weather_by_time = {
        point["time"]: point
        for point in hourly_points
    }

    evidence = []

    for ranking in rankings[:10]:
        place = place_by_id.get(ranking["place_id"], {})
        window = ranking.get("best_window")

        weather = (
            weather_by_time.get(window.get("start"))
            if window else None
        )

        evidence.append({
            "place_id": ranking["place_id"],
            "name": place.get("name"),
            "mapped_type": place.get("type"),
            "distance_km_straight_line": place.get("distance_km"),
            "deterministic_score": ranking.get("score"),
            "activity": request.activity,
            "forecast_window": window,
            "forecast": {
                "weather": weather.get("weather"),
                "temperature_c": weather.get("temp_c"),
                "precipitation_probability_percent": weather.get("precip_prob"),
                "wind_speed_kmh": weather.get("wind_speed_kmh"),
                "daylight": weather.get("is_daylight"),
            } if weather else None,
        })

    prompt = f"""
You are Terrain, an evidence-grounded outdoor activity assistant.

Use ONLY the supplied JSON evidence. Do not invent opening hours, facilities,
surface conditions, route times, air quality, access rules, safety status,
or local events. A mapped feature type does not prove a venue is open or
suitable. The weather forecast is location-level, not a measurement at each
individual place. Scores are deterministic heuristics, not probabilities
or safety ratings.

Return valid JSON only with this structure:
{{
  "recommendations": [
    {{
      "place_id": "exact supplied ID",
      "reasoning": "two short evidence-grounded sentences",
      "carry": ["practical items"],
      "watch_for": ["evidence-grounded considerations"],
      "avoid_when": ["evidence-supported conditions"],
      "activity_cautions": ["limitations or checks"]
    }}
  ]
}}

Include each supplied place ID exactly once. Do not change ranks, scores,
names, distances, or forecast values. Use empty arrays when no specific
evidence supports a suggestion. Do not output Markdown. Keep advice concise.

User request: {request.query}
Additional considerations: {request.considerations or "None"}
Evidence JSON: {json.dumps(evidence, ensure_ascii=False)}
"""

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(45.0, connect=3.0)
        ) as model_client:
            response = await model_client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1600,
                    },
                },
            )

            response.raise_for_status()

            raw_output = response.json().get("response", "")
            generated = json.loads(raw_output)

        generated_by_id = {
            item.get("place_id"): item
            for item in generated.get("recommendations", [])
            if (
                isinstance(item, dict)
                and item.get("place_id") in place_by_id
            )
        }

        for ranking in rankings:
            item = generated_by_id.get(ranking["place_id"])

            if not item:
                continue

            reasoning = item.get("reasoning")

            if isinstance(reasoning, str) and reasoning.strip():
                ranking["reasoning"] = reasoning.strip()[:700]

            for field in (
                "carry",
                "watch_for",
                "avoid_when",
                "activity_cautions",
            ):
                value = item.get(field)

                if isinstance(value, list):
                    ranking[field] = [
                        str(entry).strip()[:180]
                        for entry in value[:5]
                        if isinstance(entry, str) and entry.strip()
                    ]

        print(f"Terrain: local Qwen enrichment completed ({OLLAMA_MODEL}).")
        return rankings, True

    except Exception as exc:
        print(
            "Terrain: Qwen unavailable; using deterministic fallback. "
            f"{str(exc)[:250]}"
        )
        return rankings, False


# ============================================================
# MAIN RECOMMENDATION ENDPOINT
# ============================================================

@app.post("/recommend")
async def recommend(request: RecommendationQuery):
    generated_at = datetime.now(timezone.utc).isoformat()

    try:
        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
        ) as client:

            # 1. Resolve the user's location.
            lat, lon, place_name = await geocode_location(
                client,
                request,
            )

            # 2. Fetch current forecast data.
            weather_error = None

            try:
                hourly_points, timezone_name = await get_weather(
                    client,
                    lat,
                    lon,
                )
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                hourly_points = []
                timezone_name = "UTC"
                weather_error = str(exc)
                print(f"Weather lookup failed: {weather_error}")

            # 3. Interpret the requested time window.
            time_window, time_error = parse_time_window(
                request,
                timezone_name,
            )

            # 4. Discover mapped outdoor places.
            places, places_error = await find_nearby_places(
                client,
                lat,
                lon,
                request.radius_km,
            )

            # 5. Calculate scores deterministically.
            rankings = make_rankings(
                places=places,
                activity=request.activity,
                hourly_points=hourly_points,
                time_window=time_window,
                timezone_name=timezone_name,
                radius_km=request.radius_km,
            )

            # 6. Let local Qwen improve explanations if available.
            rankings, model_used = await enrich_rankings_with_model(
                rankings=rankings,
                places=places,
                hourly_points=hourly_points,
                request=request,
            )

            # Keep the existing frontend's expected data shape.
            # These are location-level forecasts, not park-specific measurements.
            hourly_by_place = {
                "location": hourly_points,
                **{
                    place["id"]: hourly_points
                    for place in places
                },
            }

            sources = [
                source_record(
                    "osm",
                    "OpenStreetMap / Nominatim / Overpass",
                    "https://www.openstreetmap.org/",
                    generated_at,
                ),
            ]

            if hourly_points:
                sources.append(
                    source_record(
                        "weather",
                        "Open-Meteo",
                        "https://open-meteo.com/",
                        generated_at,
                    )
                )

            if model_used:
                sources.append(
                    source_record(
                        "qwen-local",
                        f"{OLLAMA_MODEL} via local Ollama",
                        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
                        generated_at,
                    )
                )

            result = {
                "interpreted_request": {
                    "activity": request.activity,
                    "location": {
                        "name": place_name,
                        "lat": lat,
                        "lon": lon,
                    },
                    "radius_km": request.radius_km,
                    "time_window": time_window,
                    "considerations": (
                        [request.considerations]
                        if request.considerations
                        else []
                    ),
                },
                "places": places,
                "hourly": hourly_by_place,
                "rankings": rankings,
                "sources": sources,
                "generated_at": generated_at,
                "simulated": False,
                "model_used": model_used,
                "scoring_note": (
                    "Scores are heuristic ranking scores based on distance, "
                    "mapped feature-type match, and available location-level "
                    "forecast data. They are not probabilities, verified venue "
                    "suitability, or safety ratings."
                ),
            }

            if places_error:
                result["places_error"] = places_error

            if weather_error:
                result["weather_error"] = weather_error

            if time_error:
                result["time_error"] = time_error

            return result

    except HTTPException:
        raise

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"An external data service failed: {str(exc)[:300]}",
        ) from exc

    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not process an external data response: {str(exc)[:300]}",
        ) from exc

    except Exception as exc:
        print(
            f"Unexpected recommendation error: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Terrain could not complete this recommendation. "
                "Check the backend terminal logs."
            ),
        ) from exc