from setuptools import setup, find_packages

setup(
    name="surflifegen-mlx",
    version="1.0.0",
    description="Apple Silicon Native MLX 8-Bit Synthetic Swimmer Dataset Generator & YOLO Precision Annotator",
    author="True2456",
    url="https://github.com/True2456/SurfLifeGen-MLX",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.22.0",
        "torch>=2.0.0",
        "diffusers>=0.32.0",
        "transformers>=4.45.0",
        "accelerate>=0.28.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "huggingface_hub>=0.24.0",
    ],
    entry_points={
        "console_scripts": [
            "surflifegen=surflifegen.cli:main",
        ],
    },
    python_requires=">=3.10",
)
