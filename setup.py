from setuptools import setup, find_packages

setup(
    name="shiva-cognitive-os",
    version="1.0.0",
    description="Shiva.ai: B2B Cognitive Swarm Engine",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "transformers",
        "psutil",
        "websockets",
        "soundfile",
        "chromadb",
        "pydantic",
    ],
    entry_points={
        "console_scripts": [
            "shiva=src.input.cli:main",
        ],
    },
)
