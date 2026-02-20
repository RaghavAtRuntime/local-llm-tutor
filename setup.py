from setuptools import setup, find_packages

setup(
    name="local-llm-tutor",
    version="0.1.0",
    description="Locally hosted AI tutor with voice interaction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pyttsx3>=2.90",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0",
        "sentence-transformers>=2.2.2",
    ],
    extras_require={
        "audio": [
            "sounddevice>=0.4.6",
            "faster-whisper>=1.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "local-llm-tutor=local_llm_tutor.tutor:main",
        ],
    },
)
