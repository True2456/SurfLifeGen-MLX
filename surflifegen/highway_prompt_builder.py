# surflifegen/highway_prompt_builder.py
"""
Modular Prompt Builder for Synthetic Highway Defect & Surface Wear Dataset Generation.
Synthesizes photorealistic pavement inspection prompts (cracks, potholes, alligatoring, rutting, lane wear)
tailored for Apple Silicon MLX quantized Cosmos 3 Omni and zero-shot Grounding DINO localization.
"""

import random
from typing import Dict, Any, Optional

DEFECT_TYPES = {
    "alligator_crack": {
        "description": "interconnecting web of fatigue cracks forming an alligator skin or chicken wire pattern across the wheel path",
        "dino_query": "alligator cracking . crack network . pavement fracture . road crack .",
        "count_range": (1, 3)
    },
    "longitudinal_crack": {
        "description": "long continuous linear crack running parallel to the highway centerline and traffic flow direction",
        "dino_query": "longitudinal crack . road crack . linear fracture in asphalt .",
        "count_range": (1, 2)
    },
    "transverse_crack": {
        "description": "sharp thermal crack running perpendicular across the asphalt traffic lanes from shoulder to shoulder",
        "dino_query": "transverse crack . road crack . fracture across lane .",
        "count_range": (1, 3)
    },
    "pothole": {
        "description": "deep structural pothole with jagged broken asphalt edges and exposed underlying base gravel layer",
        "dino_query": "pothole . road hole . broken asphalt cavity . pavement crater .",
        "count_range": (1, 4)
    },
    "rutting": {
        "description": "longitudinal surface depression and asphalt deformation along the wheel ruts caused by heavy truck traffic",
        "dino_query": "pavement rutting . asphalt depression . wheel rut deformation .",
        "count_range": (1, 2)
    },
    "faded_lane_marking": {
        "description": "severely degraded, chipped, and weather-worn white and yellow highway paint lane markings with missing reflective glass beads",
        "dino_query": "faded road marking . worn lane line . degraded highway paint .",
        "count_range": (1, 3)
    },
    "edge_break": {
        "description": "crumbling, unraveled outer asphalt pavement edge breaking off where the travel lane meets the gravel shoulder",
        "dino_query": "pavement edge break . broken shoulder edge . crumbling asphalt .",
        "count_range": (1, 2)
    },
    "mixed": {
        "description": "combination of severe interconnecting alligator cracking surrounding a deep jagged pothole along a weathered travel lane",
        "dino_query": "pothole . alligator cracking . road crack . pavement defect .",
        "count_range": (2, 5)
    }
}

ASPHALT_SURFACES = [
    "weathered grey oxidized asphalt highway surface with exposed fine aggregates",
    "dark newly paved or seal-coated asphalt surface with distinct high-contrast surface cracks",
    "heavy-duty interstate concrete pavement slabs with aging expansion joints and edge spalling",
    "rough chip-seal country highway surface with localized bitumen bleeding and aggregate loss",
    "damp road surface after a light rain showing high-contrast dark moisture penetration inside cracks"
]

PERSPECTIVES = [
    {
        "name": "nadir_drone",
        "text": "Direct overhead nadir drone inspection photograph looking straight down from 15 meters altitude above the highway surface",
        "detail": "Orthophoto pavement inspection view providing flat geometric accuracy across both travel lanes and shoulder"
    },
    {
        "name": "low_nadir",
        "text": "Close-up overhead nadir inspection photograph looking straight down from 5 meters altitude directly over the right wheel path",
        "detail": "Macro pavement inspection resolution revealing micro-fractures, aggregate pop-outs, and crack opening widths"
    },
    {
        "name": "vehicle_surface",
        "text": "Vehicle-mounted road inspection camera view looking slightly forward and down from 2 meters height above the asphalt travel lane",
        "detail": "Automated road condition survey perspective capturing realistic oblique surface texture and shadow depth"
    }
]

LIGHTING_CONDITIONS = [
    "under clear midday overhead sunlight casting sharp micro-shadows that define crack depth and edges",
    "under diffuse overcast sky providing uniform, shadow-free illumination optimal for automated crack segmentation",
    "under low-angle golden hour morning sunlight casting long directional shadows across pavement irregularities and rutting",
    "under bright dry afternoon conditions emphasizing natural color contrast between grey asphalt and dark cracks"
]

def generate_highway_prompt(
    defect_type: str = "random",
    asphalt_type: str = "random",
    perspective: str = "random",
    custom_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a structured prompt dictionary for synthesizing highway pavement defects and surface wear.
    """
    if custom_prompt:
        return {
            "prompt": custom_prompt,
            "defect_type": defect_type if defect_type != "random" else "custom",
            "asphalt": asphalt_type,
            "perspective": perspective,
            "dino_query": "crack in asphalt . road crack . pothole . pavement defect . worn lane marking .",
            "expected_count": 2
        }

    # Select defect
    if defect_type == "random" or defect_type not in DEFECT_TYPES:
        selected_defect_key = random.choice(list(DEFECT_TYPES.keys()))
    else:
        selected_defect_key = defect_type
    defect_info = DEFECT_TYPES[selected_defect_key]

    # Select asphalt
    if asphalt_type == "random" or not asphalt_type:
        selected_asphalt = random.choice(ASPHALT_SURFACES)
    else:
        selected_asphalt = asphalt_type

    # Select perspective
    if perspective == "random":
        selected_perspective = random.choice(PERSPECTIVES)
    else:
        matching = [p for p in PERSPECTIVES if p["name"] == perspective]
        selected_perspective = matching[0] if matching else random.choice(PERSPECTIVES)

    selected_lighting = random.choice(LIGHTING_CONDITIONS)
    expected_count = random.randint(*defect_info["count_range"])

    prompt = (
        f"{selected_perspective['text']} showing {defect_info['description']} across a {selected_asphalt}. "
        f"{selected_perspective['detail']} {selected_lighting}. "
        f"Clean distinct structural road distress boundaries visible without motion blur, high optical resolution engineering inspection photograph"
    )

    return {
        "prompt": prompt,
        "defect_type": selected_defect_key,
        "asphalt": selected_asphalt,
        "perspective": selected_perspective["name"],
        "dino_query": defect_info["dino_query"],
        "expected_count": expected_count
    }
