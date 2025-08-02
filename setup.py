import pathlib
from pathlib import Path
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

long_description = (HERE / "README.md").read_text(encoding="utf-8")

def parse_requirements(filename: Path):
    with open(HERE / filename) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return lines


setup(
    name="rwm",
    version="0.1.0",
    description="Variational version for Reduced World Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Malcom Montenegro",
    author_email="malcommntngr@gmail.com",
    # url="https://github.com/MalcomDM/scrape-panatax-full",
    license="MIT",

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    python_requires=">=3.11,<4",
    install_requires=parse_requirements(Path(".devcontainer/requirements.txt")),

    entry_points={
        "console_scripts": [
            "rwm = rwm.cli:app"
        ],
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)