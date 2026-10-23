# Terrain — GeoAI Outdoor Recommendation System

Terrain is a geospatial outdoor recommendation prototype that helps users discover nearby outdoor locations based on their selected activity, geographic proximity, and available weather information.

## Features

- Location-based discovery: Retrieves nearby places using OpenStreetMap-based geospatial data.
- Activity-aware recommendations: Filters and ranks locations according to the selected activity.
- Heuristic ranking: Combines proximity, mapped activity-type compatibility, and available weather information.
- Weather-aware guidance: Uses available forecast data to inform recommendations and suggested activity windows.
- Recommendation explanations: Provides a brief rationale, practical cautions, and information about data freshness.
- API backend: A FastAPI service that provides recommendation and health-check endpoints.

## Architecture

1. The frontend collects the user's location, activity, search radius, and any available time preferences.
2. The backend retrieves and processes nearby geospatial features.
3. Activity-based filtering and heuristic scoring are applied to candidate locations.
4. Available weather information is used to support recommendations and time-window guidance.
5. The API returns ranked locations and explanatory information to the frontend.

## Technology Stack

- Backend: Python, FastAPI, Uvicorn
- Geospatial data: OpenStreetMap, Nominatim, and Overpass
- Weather: Forecast data available through the configured weather integration
- Frontend: Separate web application
- Deployment: Render

## Live API

Base URL: https://terrain-ai-api.onrender.com

Health endpoint: https://terrain-ai-api.onrender.com/health

The health endpoint can be used to check whether the deployed backend is responding.

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/swikritidasbiswas07-web/terrain-ai-api.git
cd terrain-ai-api
```

### 2. Create and activate a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

The API will be available at http://127.0.0.1:8001.

Interactive API documentation is available at /docs while the local server is running.

## Recommendation Method

Terrain currently uses a heuristic recommendation approach, rather than claiming a newly trained machine-learning model. Candidate locations are filtered by activity type and ranked using factors such as proximity, mapped feature compatibility, and available weather information.

The scores are heuristic indicators. They are not calibrated probabilities, safety ratings, or guarantees that a venue is accessible or suitable for a particular activity.

## Data Sources

- OpenStreetMap: Geographic features and mapped place information.
- Nominatim: Geocoding and location search, where configured.
- Overpass API: Retrieval of relevant OpenStreetMap features.
- Weather integration: Forecast information available through the configured provider.

Data completeness, place classifications, forecast availability, and access information can vary.

## Limitations

- A mapped feature type does not prove that a location is open to the public or suitable for an activity.
- Weather forecasts may be unavailable for some locations or requested time windows.
- Recommendation scores are heuristic and should not be interpreted as safety assessments.
- Geographic distances may represent straight-line distance rather than actual travel distance.
- OpenStreetMap coverage and place names can be incomplete or inconsistent.

Users should verify venue access, opening hours, local conditions, and applicable safety guidance before visiting.

## Project Status

Terrain is a working prototype developed to demonstrate location-aware, activity-based outdoor recommendations using geospatial data and available weather information.
