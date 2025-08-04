from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="liquid-vision-sim-kit",
    version="0.1.0",
    author="Terragon Labs",
    author_email="contact@terragonlabs.com",
    description="Neuromorphic Dataset Generator & Training Loop for Liquid Neural Networks on Edge Devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/liquid-vision-sim-kit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
        ],
    },
    entry_points={
        "console_scripts": [
            "liquid-vision=liquid_vision.cli:main",
        ],
    },
)