from setuptools import setup, find_namespace_packages, find_packages

setup(
    name="seedoo",
    version="1.0",
    packages=find_namespace_packages(include=['seedoo*']) + find_packages()
)
