#!/usr/bin/env python3
"""
Setup script for Autonomous Driving System
"""
from setuptools import setup, find_packages
import os

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="autonomous-driving-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Autonomous Driving System using Deep Learning and Computer Vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Automatic_Driving_System",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["tensorflow-gpu>=2.8.0"],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "ads-train=train_cnn:main",
            "ads-test=test_model:main",
            "ads-drive=drive_realtime:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="autonomous driving, deep learning, computer vision, CNN, lane detection",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/Automatic_Driving_System/issues",
        "Source": "https://github.com/yourusername/Automatic_Driving_System",
        "Documentation": "https://github.com/yourusername/Automatic_Driving_System/wiki",
    },
) 