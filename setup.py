import setuptools
import platform
import stat
import os
import sys

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

if platform.system() == "Linux":
    target_folder = "/usr/local/bin"
    commands = ["viper-stat", "viper-split", "viper-merge"]
    src_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),"src","vipercmd")
    bin_directory = os.path.dirname(os.path.abspath(sys.executable))
    
    for cmd in commands:
        src_module = os.path.join(src_directory,cmd+".py")
        symlink_origin = os.path.join(bin_directory,cmd)
        
        # make script executebale
        st = os.stat(src_module)
        os.chmod(src_module, st.st_mode | 0o111)
        
        if not os.path.isfile(symlink_origin):
            os.symlink(symlink_origin, src_module)