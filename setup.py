from setuptools import setup, find_packages

setup(
    name="surflifegen-mlx",
    version="1.2.0",
    description="Apple Silicon Native MLX 8-Bit Synthetic Dataset Generator, Annotator & Qwen2.5-VL Tag Verifier for Aerial Surf Life Saving AI Models",
    author="True2456",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "huggingface_hub>=0.24.0",
    ],
    extras_require={
        "vlm": ["mlx-vlm>=0.6.0"],
    },
    entry_points={
        "console_scripts": [
            "surflifegen=surflifegen.cli:main",
            "surflifegen-verify=surflifegen.verify_cli:main",
            "surflifegen-dino=surflifegen.dino_cli:main",
            "surflifegen-highalt=surflifegen.highalt_cli:main",
            "surflifegen-highway=surflifegen.highway_cli:main",
            "surflifegen-segment=surflifegen.segment_cli:main",
            "surflifegen-urban=surflifegen.urban_cli:main",
            "surflifegen-urban-live=surflifegen.urban_live_cli:main",
            "surflifegen-train-yolo=surflifegen.train_urban_yolo:main",
            "surflifegen-yolo-export=surflifegen.yolo_cli:main",
        ],
    },
    python_requires=">=3.10",
)
