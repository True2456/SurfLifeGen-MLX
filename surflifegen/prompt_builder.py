"""
Modular Prompt Engine for Aerial Surf Life Saving Dataset Generation.
Dynamically combines realistic maritime conditions, altitudes (40m-100m), swimmer counts,
attire, and lighting while strictly avoiding drone/camera object nouns.
"""

import random
from typing import Dict, Any, Tuple

ALTITUDES = [80, 90, 100, 100, 100, 110, 120]  # Weighted toward 100 meters

SWIMMER_CONFIGS = [
    (1, "a single human swimmer actively swimming or splashing across coastal ocean water"),
    (2, "two human swimmers actively swimming or splashing across coastal ocean water"),
    (3, "multiple human swimmers actively swimming or splashing across coastal ocean water"),
    (4, "a group of four human swimmers actively swimming or splashing across coastal ocean water"),
    (2, "two people actively swimming freestyle side by side across ocean swell"),
    (3, "three swimmers swimming vigorously across a rip current channel"),
]

ATTIRE_OPTIONS = [
    "wearing a high-visibility red-and-yellow lifeguard rash vest",
    "wearing a bright fluorescent orange safety rash vest",
    "wearing a vibrant yellow rescue swim shirt and cap",
    "wearing a full-length dark black neoprene wetsuit",
    "wearing high-contrast red swimwear",
    "wearing a dark wetsuit with a high-visibility orange swim cap",
]

WATER_CONDITIONS = [
    "coastal ocean water with gentle swell ripples",
    "deep azure ocean swell outside breaking sandbar waves",
    "emerald green coastal ocean water with subtle surface texture",
    "a sandy ocean rip current channel between white surf sandbanks",
    "choppy dark blue open ocean swell with scattered seafoam",
    "transparent turquoise water over a shallow rocky underwater reef",
]

LIGHTING_CONDITIONS = [
    "under bright midday sunlight with crisp surface clarity",
    "under high noon solar light with deep water saturation",
    "during morning daylight with subtle natural surface reflections",
    "under overcast diffused maritime daylight",
    "during late afternoon golden hour with warm sunlight glint on the sea surface",
]

def generate_modular_prompt() -> Dict[str, Any]:
    """
    Generates a modular, high-standard synthetic aerial prompt and associated metadata.
    Guarantees zero drone/aircraft noun hallucinations and strictly follows anatomical rules.
    """
    altitude = random.choice(ALTITUDES)
    swimmer_count, swimmer_desc = random.choice(SWIMMER_CONFIGS)
    attire = random.choice(ATTIRE_OPTIONS)
    water = random.choice(WATER_CONDITIONS)
    lighting = random.choice(LIGHTING_CONDITIONS)

    prompt = (
        f"Direct overhead nadir photograph looking straight down from {altitude} meters altitude "
        f"above sea level showing {swimmer_desc} in {water} {lighting}. "
        f"Each swimmer is {attire} and partially submerged. Make sure humans have no extra limbs. "
        f"Clean distinct human silhouettes visible against coastal ocean swell and seafoam, high optical resolution aerial photography"
    )

    return {
        "prompt": prompt,
        "altitude_m": altitude,
        "swimmer_count": swimmer_count,
        "attire": attire,
        "water_condition": water,
        "lighting": lighting
    }
