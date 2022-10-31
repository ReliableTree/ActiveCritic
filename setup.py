from distutils.core import setup
import setuptools


setup(
    name="active_critic",
    author="Hendrik Elvers",
    version="0.1",
    description="Implementation of ActiveCritic algorithm.",
    packages=setuptools.find_packages(),
    package_dir={"": "src"},
    install_requires=[
    "gym[classic_control]==0.21.0",
    "matplotlib",
    "numpy>=1.15",
    "torch>=1.4.0",
    "mujoco-py<2.2,>=2.1",
    "tensorflow-cpu",
    "hashids",
    "stable-baselines3",
    "prettytable",
    "imitation"]
)