from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Region:
    region_id: str
    name: str
    lat: float
    lon: float
    defaults: Dict[str, float]


_REGIONS: List[Region] = [
    Region(
        region_id="la",
        name="Los Angeles, CA",
        lat=34.05,
        lon=-118.25,
        defaults={
            "MedInc": 6.2,
            "HouseAge": 30.0,
            "AveRooms": 5.2,
            "AveBedrms": 1.1,
            "Population": 1500.0,
            "AveOccup": 2.7,
            "Latitude": 34.05,
            "Longitude": -118.25,
        },
    ),
    Region(
        region_id="sf",
        name="San Francisco, CA",
        lat=37.77,
        lon=-122.42,
        defaults={
            "MedInc": 8.4,
            "HouseAge": 35.0,
            "AveRooms": 5.0,
            "AveBedrms": 1.0,
            "Population": 1100.0,
            "AveOccup": 2.2,
            "Latitude": 37.77,
            "Longitude": -122.42,
        },
    ),
    Region(
        region_id="sd",
        name="San Diego, CA",
        lat=32.72,
        lon=-117.16,
        defaults={
            "MedInc": 7.1,
            "HouseAge": 28.0,
            "AveRooms": 5.4,
            "AveBedrms": 1.0,
            "Population": 1300.0,
            "AveOccup": 2.6,
            "Latitude": 32.72,
            "Longitude": -117.16,
        },
    ),
    Region(
        region_id="sac",
        name="Sacramento, CA",
        lat=38.58,
        lon=-121.49,
        defaults={
            "MedInc": 5.6,
            "HouseAge": 25.0,
            "AveRooms": 5.7,
            "AveBedrms": 1.1,
            "Population": 1400.0,
            "AveOccup": 2.8,
            "Latitude": 38.58,
            "Longitude": -121.49,
        },
    ),
    Region(
        region_id="fre",
        name="Fresno, CA",
        lat=36.74,
        lon=-119.78,
        defaults={
            "MedInc": 4.8,
            "HouseAge": 22.0,
            "AveRooms": 5.3,
            "AveBedrms": 1.1,
            "Population": 1600.0,
            "AveOccup": 3.0,
            "Latitude": 36.74,
            "Longitude": -119.78,
        },
    ),
]


def get_regions() -> List[Region]:
    return list(_REGIONS)


def get_region(region_id: str) -> Region | None:
    for region in _REGIONS:
        if region.region_id == region_id:
            return region
    return None


def default_payload_for_region(region_id: str) -> Dict[str, float]:
    region = get_region(region_id)
    if region is None:
        raise KeyError(f"Unknown region_id: {region_id}")
    return dict(region.defaults)
