# VESDI-maps

Scripts to process and visualize vehicle delivery routes in Dutch cities using postcode data, based on the CBS deliveries as part of VESDI (https://www.cbs.nl/nl-nl/dossier/vesdi).

## Prerequisites

- Python 3.x
- Required Python packages:
  - geopy
  - pandas
  - folium
  - numpy
  - requests

## Required Services & API Keys

1. OpenCage Geocoding API key
   - Sign up at https://opencagedata.com/
   - Required for converting postal codes to GPS coordinates

2. OpenRouteService Local Server
   - Must be running on http://localhost:8080
   - Used for route calculations

## Input Requirements

The scripts require the following user inputs:

### PCtoGPS_lookup.py
- `api_key`: OpenCage API key
- `stad`: City name (used for file organization)
- `deelritten_bestand`: Input CSV file with delivery route data
- Requires CSV with columns: losPC4, laadPC4, losPC6, laadPC6

### MakeMap_gh.py
- `stad`: City name (same as used in PCtoGPS_lookup.py)
- `deelritten_bestand`: CSV file containing route data with best available postcodes
- `postcode_lu`: Lookup file linking PC6 to GPS coordinates
- `url`: OpenRouteService API endpoint (default: http://localhost:8080/ors/v2/directions/driving-hgv/gpx)

## Usage

1. Run PCtoGPS_lookup.py first to:
   - Generate GPS coordinates for postal codes
   - Create lookup tables
   - Process partial postal codes

2. Run MakeMap_gh.py to:
   - Calculate delivery routes
   - Generate an interactive map visualization
   - Show route frequencies and logistics data

## Output

The scripts generate:
- Postal code to GPS coordinate lookup tables
- Processed route data
- Interactive HTML map with:
  - Color-coded route frequencies
  - Vehicle environmental classifications
  - Logistics class information
  - Popup details for each route segment

## File Structure

```
VESDI-maps/
├── PCtoGPS_lookup.py
├── MakeMap_gh.py
└── {city}/
    ├── results/
    │   └── {timestamp}/
    │       ├── routes_*.html
    │       └── routes_*.csv
    └── Postcode lookup/
        ├── PC6_GPS_*.csv
        ├── PC4_lu_*.csv
        └── {city}_bestPC.csv
```
