# setup.py

from setuptools import setup, find_packages

setup(
    name="2cat-selfmod",    # The name of your package.
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # List any dependencies here, for example:
        "torch",  # If you need PyTorch; add others as needed.
    ],
)
