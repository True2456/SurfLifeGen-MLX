"""
Modular Prompt Engine for Aerial Surf Life Saving Dataset Generation.
Dynamically combines realistic maritime conditions, altitudes (40m-100m), swimmer counts,
attire, and lighting while strictly avoiding drone/camera object nouns.
"""

import random
from typing import Dict, Any, Tuple

ALTITUDES = [40, 50, 60, 75, 85, 90, 100]

SWIMMER_CONFIGS = [
    (1, "a single human swimmer treading water"),
    (1, "a lone swimmer floating on their back"),
    (1, "one person swimming freestyle"),
    (2, "two swimmers floating side by side"),
    (2, "two swimmers treading water spaced slightly apart"),
    (3, "three swimmers spaced apart in the water"),
    (4, "a group of four swimmers floating together"),
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
    "clear turquoise open coastal ocean water with gentle swell ripples",
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
    Guarantees zero drone/aircraft noun hallucinations.
    """
    altitude = random.choice(ALTITUDES)
    swimmer_count, swimmer_desc = random.choice(SWIMMER_CONFIGS)
    attire = random.choice(ATTIRE_OPTIONS)
    water = random.choice(WATER_CONDITIONS)
    lighting = random.choice(LIGHTING_CONDITIONS)

    prompt = (
        f"Direct overhead nadir photograph looking straight down from {altitude} meters altitude "
        f"above sea level showing {swimmer_desc} in {water} {lighting}. "
        f"Each swimmer is {attire}. Clean anatomically realistic human silhouette separated "
        f"from white seafoam, high optical resolution aerial photography"
    )

    return {
        "prompt": prompt,
        "altitude_m": altitude,
        "swimmer_count": swimmer_count,
        "attire": attire,
        "water_condition": water,
        "lighting": lighting
    }
