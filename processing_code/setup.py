"""
Setup configuration for processing_code package.

Allows installation via: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="processing_code",
    version="0.1.0",
    description="Convert P4 XML files (EEBO, ECCO, EVAN) to structured page-level CSV data",
    author="Text-Machine",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.0.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
