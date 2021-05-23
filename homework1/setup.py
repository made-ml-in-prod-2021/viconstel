from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Heart Disease UCI Classification",
    author="viconstel",
    install_reqs=requirements
)