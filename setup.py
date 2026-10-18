from setuptools import setup, find_packages

setup(
    name="docuflow",
    version="0.1.0",
    author="Mohd Faizan",
    description="Modular RAG service: document Q&A, comparison and metadata analysis",
    packages=find_packages(exclude=("tests", "notebook")),
    python_requires=">=3.10",
)
