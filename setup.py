import os
from setuptools import setup, find_packages

setup(
    name='ackeras',
    version='0.0.3',
    packages=find_packages("ackeras", exclude=["test.py", "images"]),
    author='Andrea Titton',
    author_email='andrea.titton@accurat.it',
    keywords='autoML',
)
