#!/usr/bin/env python3
"""
Example standalone script demonstrating how to use SurfLifeGen-MLX programmatically.
"""

from surflifegen import SurfLifeGenPipeline, PrecisionSwimmerAnnotator

def run_example():
    # 1. Initialize Pipeline (points to local path OR automatically downloads weights)
    pipeline = SurfLifeGenPipeline(
        model_path="/Users/true/Documents/Mati_Train/models/Cosmos3-Nano-MLX-8bit",
        auto_download=True
    )

    # 2. Initialize Annotator
    annotator = PrecisionSwimmerAnnotator(output_dir="./example_dataset")

    # 3. Define prompt for high-altitude nadir aerial rescue
    prompt = (
        "Direct overhead nadir photograph looking straight down from 90 meters altitude "
        "over open coastal ocean water at a single human swimmer treading water. "
        "Swimmer wears a high-visibility red-and-yellow lifeguard rash vest. Clean distinct "
        "swimmer silhouette visible against turquoise ocean water, realistic seafoam and swell ripples."
    )

    print("Generating image...")
    image = pipeline.generate(prompt=prompt, width=1024, height=768, steps=25)
    image_path = "./example_dataset/example_swimmer_90m.png"
    image.save(image_path)

    print("Annotating swimmer bounding boxes...")
    annotation = annotator.annotate_image(image_path, target_count=1)

    print("Result YOLO Label:", annotation["yolo_label_file"])
    annotator.export_html_gallery([annotation])

if __name__ == "__main__":
    run_example()
