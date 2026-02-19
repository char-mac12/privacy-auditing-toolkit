from setuptools import setup, find_packages

setup(
    name="privacy-auditing-toolkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "reportlab>=4.0.0",
    ],
)