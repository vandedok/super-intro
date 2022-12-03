from setuptools import setup, find_packages

setup( 
    name="superintro",
    version="0.1dev",
    packages=find_packages(include=["superintro"]),
    install_requires=[
       'numpy',
       'opencv-python',
       'matplotlib'
    ]
)
