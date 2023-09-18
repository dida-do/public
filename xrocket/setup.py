from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="xrocket",
    version="0.1",
    description="explainable rocket implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="dida Datenschmiede GmbH",
    author_email="info@dida.do",
    packages=find_packages(),
    install_requires=[
        "pytorch"
    ],
    python_requires=">=3.6",
)
