from distutils.core import setup
import setuptools


setup(
    name="active_critic",
    author="Hendrik Elvers",
    version="0.1",
    description="Implementation of ActiveCritic algorithm.",
    packages=setuptools.find_packages(),
    package_dir={"": "src"},
)