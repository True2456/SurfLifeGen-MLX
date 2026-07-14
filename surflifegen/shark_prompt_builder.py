"""
Modular Aerial Nadir Prompt Builder for Submerged Marine Predators (Sharks)
Generates varied, high-standard maritime Search and Rescue prompts for training
aerial patrol models to detect submerged shark silhouettes from 40m-100m altitude.
Strictly omits all aircraft, drone, and camera hardware terminology.
"""

import random
from typing import Dict, Any

ALTITUDES_M = [40, 50, 60, 70, 75, 80, 85, 90, 100]

SHARK_SPECIES_AND_SILHOUETTES = [
    "a large Great White Shark submerged two meters below the surface, showing a broad dark fusiform torpedo silhouette with extended pectoral fins and caudal tail outline visible beneath clear turquoise seawater",
    "a lone Bull Shark cruising just beneath surface swell ripples over a sandy coastal drop-off, appearing as a dark streamlined predatory silhouette contrasting sharply against light sandy seabed beneath the water",
    "a Tiger Shark swimming submerged beneath gentle ocean swell outside the surf break, its dark elongated silhouette and pectoral fin span clearly refracted through transparent coastal water",
    "a Bronze Whaler shark gliding submerged at shallow depth along a coastal reef channel, visible from directly above as a distinct dark marine predator silhouette under clear sunlit seawater",
    "a large predatory shark submerged beneath gentle turquoise ocean swell, showing an unmistakable dark streamlined silhouette with extended pectoral fins visible through crystal clear subsurface water"
]

WATER_AND_SEABED_CONTEXTS = [
    "over clear shallow turquoise ocean water with a bright white sand seabed visible below, providing strong visual contrast for the dark submerged silhouette",
    "in deep azure coastal ocean swell outside breaking sandbar waves, with gentle swell ripples refracting sunlight across the water surface",
    "along a coastal rip current sandbar channel where turquoise shallow water meets deeper blue water, with subtle seafoam lace drifting on the surface",
    "over a sun-dappled coastal reef pass with crystal clear turquoise seawater showing the sandy ocean floor beneath",
    "in calm open coastal bay water under high optical clarity conditions, allowing deep subsurface penetration"
]

LIGHTING_CONDITIONS = [
    "under direct midday high sun overhead providing maximum vertical water transparency and minimal surface glare",
    "under clear early afternoon sunlight with soft caustic light patterns dancing across the sandy seafloor",
    "under bright diffused coastal daylight revealing crisp subsurface details and underwater contrast",
    "during late morning maritime sun illuminating the turquoise water column with natural optical depth"
]

CAMERA_PERSPECTIVES = [
    "Direct overhead nadir photograph looking straight down from {alt} meters altitude above sea level",
    "Straight-down vertical photograph captured from {alt} meters height directly above the ocean surface",
    "Overhead nadir aerial photograph from {alt} meters altitude looking straight down into clear coastal water"
]

QUALITY_SUFFIXES = [
    "High optical resolution aerial photography, realistic water depth refraction, natural subsurface light absorption, crisp anatomical silhouette",
    "Realistic marine photography, clean optical refraction through seawater, distinct submerged biological shape, sharp contrast against ocean background",
    "Authentic oceanographic aerial view, natural water transparency, true-to-life submerged marine silhouette"
]

def generate_shark_prompt(altitude_m: int = None, count: int = 1) -> Dict[str, Any]:
    """
    Generates a modular, randomized prompt for submerged shark aerial detection.
    """
    alt = altitude_m if altitude_m is not None else random.choice(ALTITUDES_M)
    perspective = random.choice(CAMERA_PERSPECTIVES).format(alt=alt)
    species_sil = random.choice(SHARK_SPECIES_AND_SILHOUETTES)
    context = random.choice(WATER_AND_SEABED_CONTEXTS)
    lighting = random.choice(LIGHTING_CONDITIONS)
    suffix = random.choice(QUALITY_SUFFIXES)

    if count == 2:
        species_sil = species_sil.replace("a large", "two large").replace("a lone", "two").replace("a Tiger", "two Tiger").replace("a Bronze", "two Bronze")

    prompt = (
        f"{perspective} showing {species_sil} {context} {lighting}. {suffix}"
    )

    return {
        "prompt": prompt,
        "altitude_m": alt,
        "shark_count": count,
        "target_class": "shark",
        "description": species_sil[:60] + "..."
    }
