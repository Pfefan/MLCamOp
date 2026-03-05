"""Package setup for concert-video-editor."""

from setuptools import find_packages, setup

setup(
    name="concert-video-editor",
    version="0.1.0",
    author="Stefan",
    description="ML project for editing concert videos by combining total and close-up views.",
    packages=find_packages(),  # finds 'src' and all sub-packages: src.data, src.models etc.
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-learn",
        "torch",
        "torchvision",
        "moviepy",
        "pandas",
        "matplotlib",
        "PyYAML",
    ],
)
