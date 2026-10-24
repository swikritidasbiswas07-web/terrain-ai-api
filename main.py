import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from math import atan2, cos, isfinite, radians, sin, sqrt
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# TERRAIN CONFIGURATION
# ============================================================

APP_VERSION = "3.5.0"

app = FastAPI(
    title="Terrain Outdoor Activity API",
    version=APP_VERSION,
    description=(
        "Evidence-grounded outdoor recommendations using live "
        "geospatial and weather data with optional Qwen inference via Hugging Face."
    ),
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
USER_AGENT = "TerrainOutdoorAssistant/3.2 (outdoor activity research project)"
INDIA_TIMEZONE = "Asia/Kolkata"

# Hosted open-weight model inference. Keep the token in the deployment
# environment; never hard-code it or return it from an API response.
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_MODEL = os.getenv(
    "TERRAIN_HF_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct",
).strip()
HF_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"
HF_ENABLED = os.getenv(
    "TERRAIN_HF_ENABLED",
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
# FEATURE CLASSIFICATION
# ============================================================

# Exact OpenStreetMap tag/value pairs only.
# A generic "leisure", "sport", or "boundary" tag is not enough.

OVERPASS_FEATURE_TAGS = {
    "leisure": {
        "park",
        "garden",
        "pitch",
        "golf_course",
        "playground",
        "track",
        "sports_centre",
        "nature_reserve",
        "common",
    },
    "landuse": {
        "recreation_ground",
    },
    "natural": {
        "wood",
        "forest",
        "heath",
        "scrub",
    },
    "tourism": {
        "picnic_site",
    },
    "boundary": {
        "national_park",
        "protected_area",
    },
}

NOMINATIM_FEATURE_TYPES = {
    ("leisure", "park"): "park",
    ("leisure", "garden"): "garden",
    ("leisure", "pitch"): "pitch",
    ("leisure", "golf_course"): "golf_course",
    ("leisure", "playground"): "playground",
    ("leisure", "track"): "track",
    ("leisure", "sports_centre"): "sports_centre",
    ("leisure", "nature_reserve"): "nature_reserve",
    ("leisure", "common"): "common",
    ("landuse", "recreation_ground"): "recreation_ground",
    ("natural", "wood"): "wood",
    ("natural", "forest"): "forest",
    ("natural", "heath"): "heath",
    ("natural", "scrub"): "scrub",
    ("tourism", "picnic_site"): "picnic_site",
    ("boundary", "national_park"): "national_park",
    ("boundary", "protected_area"): "protected_area",
}

PREFERRED_TYPES = {
    "running": {"park", "recreation_ground", "track"},
    "jogging": {"park", "recreation_ground", "track"},
    "walking": {
        "park", "garden", "recreation_ground", "wood",
        "forest", "heath", "nature_reserve", "national_park",
        "common",
    },
    "cycling": {
        "park", "recreation_ground", "track", "wood",
        "forest", "cycleway",
    },
    "golf": {"golf_course"},
    "cricket": {"pitch", "recreation_ground"},
    "football": {"pitch", "recreation_ground"},
    "soccer": {"pitch", "recreation_ground"},
    "tennis": {"pitch", "sports_centre"},
    "hiking": {
        "wood", "forest", "heath", "park", "nature_reserve",
        "national_park", "protected_area",
    },
    "picnic": {
        "picnic_site", "park", "garden", "recreation_ground",
    },
    "outdoor events": {
        "park", "garden", "recreation_ground",
    },
    "photography": {
        "park", "garden", "wood", "forest", "heath",
        "nature_reserve", "national_park",
    },
}


ACTIVITY_ALIASES = {
    "run": "running",
    "running": "running",
    "jog": "jogging",
    "jogging": "jogging",
    "walk": "walking",
    "walking": "walking",
    "bike": "cycling",
    "biking": "cycling",
    "cycle": "cycling",
    "cycling": "cycling",
    "golfing": "golf",
    "golf": "golf",
    "cricket": "cricket",
    "football": "football",
    "soccer": "soccer",
    "tennis": "tennis",
    "hike": "hiking",
    "hiking": "hiking",
    "picnicking": "picnic",
    "picnic": "picnic",
    "outdoor event": "outdoor events",
    "outdoor events": "outdoor events",
    "photography": "photography",
}


def normalize_activity(activity):
    """Normalize common activity names and aliases."""
    text = (activity or "").strip().lower()
    return ACTIVITY_ALIASES.get(text, text)


def filter_places_for_activity(places, activity):
    """Filter known activities by mapped place type, without substitutions."""
    activity_key = normalize_activity(activity)
    preferred_types = PREFERRED_TYPES.get(activity_key)

    if not preferred_types:
        return places, None

    filtered_places = [
        place
        for place in places
        if (place.get("type") or "").strip().lower() in preferred_types
    ]
    excluded_count = len(places) - len(filtered_places)

    if not filtered_places:
        note = (
            f"No nearby places with mapped feature types matching "
            f"{activity_key!r} were found within the requested radius. "
            "Try increasing the radius or choosing another activity. "
            "Unrelated places have not been substituted."
        )
    elif excluded_count:
        note = (
            f"Excluded {excluded_count} place(s) whose mapped feature "
            f"types do not match {activity_key!r}. A mapped feature type "
            "does not prove that a venue permits or is suitable for this "
            "activity."
        )
    else:
        note = None

    return filtered_places, note


# ============================================================
# GENERAL UTILITIES
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two geographic coordinates."""
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


def valid_coordinates(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False

    return (
        isfinite(lat)
        and isfinite(lon)
        and -90 <= lat <= 90
        and -180 <= lon <= 180
    )


def source_record(source_id, name, url, retrieved_at):
    return {
        "id": source_id,
        "name": name,
        "url": url,
        "retrieved_at": retrieved_at,
        "observed_at": None,
    }


def weather_description(code):
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


def is_india_location(lat, lon):
    return 6.0 <= lat <= 37.5 and 68.0 <= lon <= 97.5


def timezone_for_location(lat, lon, api_timezone=None):
    """Use India Standard Time in India, otherwise the API timezone."""
    if is_india_location(lat, lon):
        return INDIA_TIMEZONE

    if api_timezone:
        try:
            ZoneInfo(api_timezone)
            return api_timezone
        except (ValueError, KeyError, ZoneInfoNotFoundError):
            pass

    return "UTC"


def parse_datetime(value, local_tz):
    parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=local_tz)

    return parsed.astimezone(local_tz)


def normalize_feature_type(tags):
    """
    Return a recognized feature type from exact OSM tag/value pairs.

    Never use a generic sport/leisure/boundary category as a type.
    """
    priority = [
        "leisure",
        "landuse",
        "tourism",
        "natural",
        "boundary",
    ]

    for key in priority:
        value = str(tags.get(key, "")).strip().lower()

        if value in OVERPASS_FEATURE_TAGS.get(key, set()):
            return value

    return None


def valid_place_name(value):
    if not isinstance(value, str):
        return None

    name = " ".join(value.split()).strip()

    if not name or len(name) > 180:
        return None

    if name.lower() in {
        "unknown", "unnamed", "null", "none", "park", "garden",
    }:
        return None

    return name


def normalize_osm_place(element, origin_lat, origin_lon, radius_km):
    tags = element.get("tags") or {}
    feature_type = normalize_feature_type(tags)

    if not feature_type:
        return None

    name = valid_place_name(tags.get("name"))

    if not name:
        return None

    center = element.get("center") or {}
    p_lat = element.get("lat", center.get("lat"))
    p_lon = element.get("lon", center.get("lon"))

    if not valid_coordinates(p_lat, p_lon):
        return None

    p_lat = float(p_lat)
    p_lon = float(p_lon)
    distance = haversine_km(origin_lat, origin_lon, p_lat, p_lon)

    if distance > radius_km:
        return None

    osm_type = element.get("type")
    osm_id = element.get("id")

    if not osm_type or osm_id is None:
        return None

    place_id = f"osm-{osm_type}-{osm_id}"

    return {
        "id": place_id,
        "name": name,
        "type": feature_type,
        "lat": p_lat,
        "lon": p_lon,
        "distance_km": round(distance, 2),
        "travel_time_min": None,
    }


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
    """Actively verify that the configured Hugging Face model can respond."""
    if not HF_ENABLED:
        return {
            "enabled": False,
            "available": False,
            "model": HF_MODEL,
            "provider": "Hugging Face Inference Providers",
            "detail": "Model inference is disabled by configuration.",
        }

    if not HF_TOKEN:
        return {
            "enabled": True,
            "available": False,
            "model": HF_MODEL,
            "provider": "Hugging Face Inference Providers",
            "detail": "HF_TOKEN is not configured in the deployment environment.",
        }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=8.0)) as client:
            response = await client.post(
                HF_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": HF_MODEL,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "max_tokens": 2,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            payload = response.json()
            content = payload.get("choices", [{}])[0].get("message", {}).get("content")
            available = isinstance(content, str) and bool(content.strip())

        return {
            "enabled": True,
            "available": available,
            "model": HF_MODEL,
            "provider": "Hugging Face Inference Providers",
            "detail": "Inference test succeeded." if available else "Provider returned no text content.",
        }

    except httpx.HTTPStatusError as exc:
        # Avoid returning request headers or the token; provider error bodies can
        # contain useful model-routing details but are kept short.
        detail = exc.response.text[:250] if exc.response is not None else str(exc)[:200]
        return {
            "enabled": True,
            "available": False,
            "model": HF_MODEL,
            "provider": "Hugging Face Inference Providers",
            "detail": f"Inference provider returned HTTP {exc.response.status_code}: {detail}",
        }
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "model": HF_MODEL,
            "provider": "Hugging Face Inference Providers",
            "detail": f"Inference check failed: {type(exc).__name__}: {str(exc)[:180]}",
        }


# ============================================================
# TIME WINDOW INTERPRETATION
# ============================================================

def parse_time_window(request, timezone_name):
    try:
        local_tz = ZoneInfo(timezone_name)
    except (ValueError, KeyError, ZoneInfoNotFoundError):
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
            start_dt = parse_datetime(request.start, local_tz)
            end_dt = parse_datetime(request.end, local_tz)
        except (ValueError, TypeError, OverflowError):
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
            if not valid_coordinates(lat, lon):
                raise HTTPException(
                    status_code=422,
                    detail="Coordinates are outside valid ranges.",
                )

            return (
                float(lat),
                float(lon),
                f"Coordinates: {lat}, {lon}",
            )

    if not request.location or not request.location.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a city, place, or valid coordinates.",
        )

    try:
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

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="The location lookup service is temporarily unavailable.",
        ) from exc

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Location not found. Try a city or nearby landmark.",
        )

    result = results[0]

    if not valid_coordinates(result.get("lat"), result.get("lon")):
        raise HTTPException(
            status_code=502,
            detail="The location service returned invalid coordinates.",
        )

    return (
        float(result["lat"]),
        float(result["lon"]),
        result.get("display_name", request.location.strip()),
    )


# ============================================================
# NEARBY OUTDOOR PLACE DISCOVERY
# ============================================================

async def find_nearby_places(client, lat, lon, radius_km):
    """
    Search Overpass first, then use Nominatim if Overpass fails
    or produces no validated outdoor places.
    """
    radius_m = int(radius_km * 1000)
    last_error = None

    query = f"""
    [out:json][timeout:25];
    (
      nwr(around:{radius_m},{lat},{lon})["leisure"~"^(park|garden|pitch|golf_course|playground|track|sports_centre|nature_reserve|common)$"];
      nwr(around:{radius_m},{lat},{lon})["landuse"="recreation_ground"];
      nwr(around:{radius_m},{lat},{lon})["natural"~"^(wood|forest|heath|scrub)$"];
      nwr(around:{radius_m},{lat},{lon})["tourism"="picnic_site"];
      nwr(around:{radius_m},{lat},{lon})["boundary"~"^(national_park|protected_area)$"];
    );
    out center tags;
    """

    # Try both Overpass endpoints. Do not stop on an empty/invalid result.
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = await client.post(
                endpoint,
                content=query,
                headers={"Content-Type": "text/plain"},
                timeout=35,
            )
            response.raise_for_status()
            payload = response.json()

            elements = payload.get("elements", [])
            places_by_id = {}

            for element in elements:
                place = normalize_osm_place(
                    element, lat, lon, radius_km
                )
                if place:
                    places_by_id[place["id"]] = place

            places = sorted(
                places_by_id.values(),
                key=lambda item: item["distance_km"],
            )

            if places:
                print(
                    f"Overpass succeeded: {len(places)} validated places."
                )
                return places[:30], None

            last_error = (
                "Overpass returned no places matching the strict "
                "outdoor-feature validation rules."
            )
            print(last_error)

        except (httpx.HTTPError, ValueError, TypeError) as exc:
            last_error = str(exc)
            print(f"Overpass failed: {last_error}")

    # Nominatim fallback. Accept exact category/type combinations only.
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / max(
        111.0 * abs(cos(radians(lat))),
        0.01,
    )

    left = max(-180, lon - lon_delta)
    right = min(180, lon + lon_delta)
    top = min(90, lat + lat_delta)
    bottom = max(-90, lat - lat_delta)

    viewbox = f"{left},{top},{right},{bottom}"

    search_terms = [
        "park",
        "garden",
        "golf course",
        "sports field",
        "picnic site",
        "nature reserve",
    ]

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
                category = str(result.get("category", "")).lower()
                feature_type = str(result.get("type", "")).lower()

                normalized_type = NOMINATIM_FEATURE_TYPES.get(
                    (category, feature_type)
                )

                if not normalized_type:
                    continue

                name = valid_place_name(
                    result.get("name")
                    or result.get("display_name", "").split(",")[0]
                )

                if not name:
                    continue

                if not valid_coordinates(
                    result.get("lat"), result.get("lon")
                ):
                    continue

                p_lat = float(result["lat"])
                p_lon = float(result["lon"])
                distance = haversine_km(lat, lon, p_lat, p_lon)

                if distance > radius_km:
                    continue

                osm_type = result.get("osm_type")
                osm_id = result.get("osm_id")

                if osm_type and osm_id is not None:
                    place_id = f"osm-{osm_type}-{osm_id}"
                else:
                    place_id = f"nominatim-{result.get('place_id')}"

                fallback_places[place_id] = {
                    "id": place_id,
                    "name": name,
                    "type": normalized_type,
                    "lat": p_lat,
                    "lon": p_lon,
                    "distance_km": round(distance, 2),
                    "travel_time_min": None,
                }

        except (httpx.HTTPError, ValueError, TypeError) as exc:
            last_error = str(exc)
            print(f"Nominatim fallback failed for {term}: {last_error}")

    places = sorted(
        fallback_places.values(),
        key=lambda item: item["distance_km"],
    )

    print(f"Nominatim fallback: {len(places)} validated places found.")

    if places:
        return places[:30], None

    return (
        [],
        last_error or "No validated nearby outdoor places were found.",
    )


# ============================================================
# WEATHER DATA
# ============================================================

async def get_weather(client, lat, lon):
    requested_timezone = (
        INDIA_TIMEZONE if is_india_location(lat, lon) else "auto"
    )

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
            "timezone": requested_timezone,
        },
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    points = []

    for i, timestamp in enumerate(hourly.get("time", [])):
        code = at(hourly, "weather_code", i)
        is_day = at(hourly, "is_day", i)

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
                bool(is_day) if is_day is not None else None
            ),
        })

    return points, payload.get("timezone")


# ============================================================
# WEATHER WINDOW SELECTION
# ============================================================

def choose_forecast_window(hourly_points, time_window, timezone_name):
    try:
        local_tz = ZoneInfo(timezone_name)
    except (ValueError, KeyError, ZoneInfoNotFoundError):
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

        if requested_start is not None and requested_end is not None:
            if not requested_start <= point_dt < requested_end:
                continue
        elif point_dt > now_local + timedelta(hours=48):
            continue

        rain = point.get("precip_prob")
        wind = point.get("wind_speed_kmh")
        temp = point.get("temp_c")
        daylight = point.get("is_daylight")

        risk = 0.0
        risk += (
            min(60.0, float(rain) * 0.6)
            if rain is not None else 20.0
        )
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

def make_rankings(
    places,
    activity,
    hourly_points,
    time_window,
    timezone_name,
    radius_km,
):
    activity_key = normalize_activity(activity)
    preferred = PREFERRED_TYPES.get(activity_key, set())

    # Defensive filtering keeps direct callers consistent with the endpoint.
    if preferred:
        places = [
            place for place in places
            if (place.get("type") or "").strip().lower() in preferred
        ]

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

        if type_match:
            activity_score = 100.0
        elif not preferred:
            activity_score = 55.0
        else:
            activity_score = 35.0

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
                penalties += min(
                    20.0,
                    (float(temp) - 30.0) * 2.0,
                )

            if best_weather.get("is_daylight") is False:
                penalties += 8.0

            weather_score = max(0.0, 100.0 - penalties)

        score = round(
            0.35 * proximity_score
            + 0.35 * activity_score
            + 0.30 * weather_score,
            1,
        )

        scored_places.append((score, place, type_match))

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
                    else (
                        "Specific suitability is not established "
                        "by mapped feature type"
                    )
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
        activity_lower = normalize_activity(activity)

        if activity_lower in {
            "running", "jogging", "walking", "hiking", "cycling",
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
                "Confirm opening hours, access rules, and on-site conditions "
                "before visiting."
            ],
            "source_ids": ["osm"] + (
                ["weather"] if hourly_points else []
            ),
        })

    return rankings


# ============================================================
# OPTIONAL QWEN INFERENCE VIA HUGGING FACE
# ============================================================

async def enrich_rankings_with_model(
    rankings,
    places,
    hourly_points,
    request,
):
    """Use Qwen for evidence-grounded explanations; Python owns all scores/ranks."""
    if not HF_ENABLED or not HF_TOKEN or not rankings:
        return rankings, False

    place_by_id = {place["id"]: place for place in places}
    weather_by_time = {point["time"]: point for point in hourly_points}
    evidence = []

    for ranking in rankings[:10]:
        place = place_by_id.get(ranking["place_id"], {})
        window = ranking.get("best_window")
        weather = weather_by_time.get(window.get("start")) if window else None
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

    prompt = f"""You are Terrain, an evidence-grounded outdoor activity assistant.
Use ONLY the supplied JSON evidence. Do not invent opening hours, facilities,
surface conditions, route times, air quality, access rules, safety status, or
local events. A mapped feature type does not prove a venue is open or suitable.
Weather is location-level forecast data. Scores are deterministic heuristics,
not probabilities or safety ratings.

Return valid JSON only in this schema:
{{"recommendations":[{{"place_id":"exact supplied ID","reasoning":"two short evidence-grounded sentences","carry":["practical items"],"watch_for":["evidence-grounded considerations"],"avoid_when":["evidence-supported conditions"],"activity_cautions":["limitations or checks"]}}]}}
Include only supplied place IDs, at most once each. Do not change ranks, scores,
names, distances, or forecast values. Use empty arrays when no specific evidence
supports a suggestion. Do not output Markdown. Keep advice concise.

User request: {request.query}
Additional considerations: {request.considerations or "None"}
Evidence JSON: {json.dumps(evidence, ensure_ascii=False)}
"""

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(75.0, connect=8.0),
            follow_redirects=True,
        ) as model_client:
            response = await model_client.post(
                HF_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": HF_MODEL,
                    "messages": [
                        {"role": "system", "content": "Return only valid JSON matching the requested schema."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1800,
                },
            )
            response.raise_for_status()
            payload = response.json()
            raw_output = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(raw_output, str) or not raw_output.strip():
                return rankings, False

        # Parse direct JSON first; tolerate a provider/model adding code fences
        # or a short preamble, without accepting invented place IDs.
        try:
            generated = json.loads(raw_output.strip())
        except json.JSONDecodeError:
            first = raw_output.find("{")
            last = raw_output.rfind("}")
            if first < 0 or last <= first:
                return rankings, False
            generated = json.loads(raw_output[first:last + 1])

        recommendations = generated.get("recommendations", []) if isinstance(generated, dict) else []
        if not isinstance(recommendations, list):
            return rankings, False

        valid_ids = set(place_by_id)
        generated_by_id = {}
        for item in recommendations:
            if not isinstance(item, dict):
                continue
            place_id = item.get("place_id")
            if isinstance(place_id, str) and place_id in valid_ids and place_id not in generated_by_id:
                generated_by_id[place_id] = item

        for ranking in rankings:
            item = generated_by_id.get(ranking["place_id"])
            if not item:
                continue
            reasoning = item.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                ranking["reasoning"] = reasoning.strip()[:700]
            for field in ("carry", "watch_for", "avoid_when", "activity_cautions"):
                value = item.get(field)
                if isinstance(value, list):
                    ranking[field] = [
                        entry.strip()[:180]
                        for entry in value[:5]
                        if isinstance(entry, str) and entry.strip()
                    ]

        model_used = bool(generated_by_id)
        print(f"Terrain: Hugging Face Qwen enrichment {'completed' if model_used else 'returned no usable recommendations'} ({HF_MODEL}).")
        return rankings, model_used

    except Exception as exc:
        print(
            "Terrain: Hugging Face Qwen unavailable; using deterministic fallback. "
            f"{type(exc).__name__}: {str(exc)[:250]}"
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

            # 1. Resolve location.
            lat, lon, place_name = await geocode_location(
                client, request
            )

            # 2. Retrieve hourly weather.
            weather_error = None
            api_timezone = None

            try:
                hourly_points, api_timezone = await get_weather(
                    client, lat, lon
                )
            except (
                httpx.HTTPError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                hourly_points = []
                weather_error = str(exc)
                print(f"Weather lookup failed: {weather_error}")

            timezone_name = timezone_for_location(
                lat, lon, api_timezone
            )

            # 3. Interpret the user's requested time.
            time_window, time_error = parse_time_window(
                request, timezone_name
            )

            # 4. Find and validate nearby outdoor features.
            places, places_error = await find_nearby_places(
                client, lat, lon, request.radius_km
            )

            # 5. Filter places for the requested activity before ranking.
            places, activity_filter_note = filter_places_for_activity(
                places, request.activity
            )

            # 6. Rank the filtered places deterministically.
            rankings = make_rankings(
                places=places,
                activity=request.activity,
                hourly_points=hourly_points,
                time_window=time_window,
                timezone_name=timezone_name,
                radius_km=request.radius_km,
            )

            # 7. Optionally enrich explanations.
            rankings, model_used = await enrich_rankings_with_model(
                rankings=rankings,
                places=places,
                hourly_points=hourly_points,
                request=request,
            )

            # Preserve the existing frontend's expected data shape.
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
                        "qwen-huggingface",
                        f"{HF_MODEL} via Hugging Face Inference Providers",
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
                        "timezone": timezone_name,
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
                "model": HF_MODEL if model_used else None,
                "model_provider": "Hugging Face Inference Providers" if model_used else None,
                "scoring_note": (
                    "Scores are heuristic ranking scores based on distance, "
                    "mapped feature-type match, and available location-level "
                    "forecast data. They are not probabilities, verified "
                    "venue suitability, or safety ratings."
                ),
            }

            if places_error:
                result["places_error"] = places_error

            if activity_filter_note:
                result["activity_filter_note"] = activity_filter_note

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
            detail=(
                "An external data service failed: "
                f"{str(exc)[:300]}"
            ),
        ) from exc

    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not process an external data response: "
                f"{str(exc)[:300]}"
            ),
        ) from exc

    except Exception as exc:
        print(
            "Unexpected recommendation error: "
            f"{type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Terrain could not complete this recommendation. "
                "Check the backend terminal logs."
            ),
        ) from exc