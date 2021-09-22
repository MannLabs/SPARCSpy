import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="viper-core",
    version="0.0.1",
    author="Georg Wallmann, Sophia Mädler, Niklas Schmacke",
    author_email="g.wallmann@campus.lmu.de",
    description="Viper core processing pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeorgWa/viper-core",
    project_urls={
        "Bug Tracker": "https://github.com/GeorgWa/viper-core",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

