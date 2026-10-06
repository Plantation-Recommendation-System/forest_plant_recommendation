from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

USER_AGENT = "AI-Based-Plantation-Recommendation-System/1.0"


class LocationService:

    def _request_json(self, url: str) -> Any:
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=15) as response:
                return json.loads(
                    response.read().decode("utf-8")
                )
        except Exception as error:
            raise RuntimeError(
                "Could not connect to the geocoding service."
            ) from error


    def _extract_location(
        self,
        *,
        result: dict[str, Any],
        latitude: float,
        longitude: float,
        query: str,
    ) -> dict[str, Any]:

        address = result.get("address", {})

        country = address.get("country") or ""

        province = (
            address.get("state")
            or address.get("region")
            or address.get("state_district")
            or address.get("county")
            or ""
        )

        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("county")
            or ""
        )

        if not country:
            raise ValueError(
                "Location was found but country could not be determined."
            )

        if not province:
            province = country

        return {
            "query": query,
            "display_name": result.get(
                "display_name",
                query,
            ),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "country": country,
            "province": province,
            "city": city,
        }


    def geocode(
        self,
        location: str,
    ) -> dict[str, Any]:

        location = str(location).strip()

        if not location:
            raise ValueError("Location cannot be empty.")

        params = {
            "q": location,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 1,
        }

        url = (
            f"{NOMINATIM_SEARCH_URL}?"
            f"{urlencode(params)}"
        )

        payload = self._request_json(url)

        if not payload:
            raise ValueError(
                f"Location could not be found: {location}"
            )

        result = payload[0]

        return self._extract_location(
            result=result,
            latitude=float(result["lat"]),
            longitude=float(result["lon"]),
            query=location,
        )


    def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:

        latitude = float(latitude)
        longitude = float(longitude)

        if not -90 <= latitude <= 90:
            raise ValueError(
                "Latitude must be between -90 and 90."
            )

        if not -180 <= longitude <= 180:
            raise ValueError(
                "Longitude must be between -180 and 180."
            )

        params = {
            "lat": latitude,
            "lon": longitude,
            "format": "jsonv2",
            "addressdetails": 1,
            "zoom": 10,
        }

        url = (
            f"{NOMINATIM_REVERSE_URL}?"
            f"{urlencode(params)}"
        )

        result = self._request_json(url)

        if not result or result.get("error"):
            raise ValueError(
                "The detected coordinates could not be resolved to a location."
            )

        query = f"{latitude:.6f}, {longitude:.6f}"

        return self._extract_location(
            result=result,
            latitude=latitude,
            longitude=longitude,
            query=query,
        )


location_service = LocationService()
