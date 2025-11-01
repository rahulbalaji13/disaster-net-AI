from setuptools import setup, find_packages

setup(
    name="disasterNet-swarmAI",
    version="0.1.0",
    description="Swarm-Enabled Edge Intelligence for Disaster Response IoT Systems",
    author="Rahul Balaji",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
)
