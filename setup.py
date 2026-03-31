"""Package setup for concert-video-editor."""

from setuptools import find_packages, setup

setup(
    name="concert-video-editor",
    version="0.1.0",
    author="Stefan",
    description="ML-based concert video editor: auto-cut between wide and close-up cameras.",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-learn",
        "torch",
        "torchvision",
        "PyYAML",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "jupyter", "ipykernel", "matplotlib"],
    },
)
