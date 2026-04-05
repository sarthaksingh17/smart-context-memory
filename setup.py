from setuptools import setup, find_packages

setup(
    name             = "smart-context-memory",
    version          = "0.1.0",
    author           = "Sarthak",
    description      = "Smart context/memory manager for LLM conversations",
    packages         = find_packages(),
    python_requires  = ">=3.9",
    install_requires = [
        "tiktoken>=0.5.0",
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0",
    ],
    extras_require = {
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
)
