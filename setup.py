# setup.py
#pip install .
# setup.py
from setuptools import setup, find_packages

setup(
    name="kratiox",
    version="0.1.0",
    description="Sprach- und Übersetzungstool",
    author="klamenzui",
    python_requires=">=3.10,<3.11",        # Nur Python 3.10.x
    install_requires=[
        "numpy==1.23.5",
        "scipy",
        "sounddevice",
        "webrtcvad",
        "TTS==0.22.0",
        "transformers",
        "sentencepiece",
        "torch==2.7.1",
        "openai-whisper @ git+https://github.com/openai/whisper.git@main",
    ],
    packages=find_packages(),
)
