from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dida-labelingtool",
    version="0.1",
    description="tool for image class labeling in a notebook",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="dida Datenschmiede GmbH",
    author_email="info@dida.do",
    packages=find_packages(),
    install_requires=[
        "ipython",
        "ipywidgets",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
