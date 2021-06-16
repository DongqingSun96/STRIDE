# -*- coding: utf-8 -*-
# @Author: Dongqing Sun
# @E-mail: Dongqingsun96@gmail.com
# @Date:   2021-06-10 15:25:08
# @Last Modified by:   Dongqing Sun
# @Last Modified time: 2021-06-17 02:10:19


import sys,os

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Could not load setuptools. Please install the setuptools package.")

with open("version.txt", "r") as infile:
    version = infile.readline().strip()

def main():
    setup(
        name = "STRIDE",
        package_dir = {'':'src'},
        version = version,
        packages = find_packages(where="src"),
        scripts = ['bin/STRIDE'],
        include_package_data = True,
        author = "Dongqing Sun",
        author_email = "Dongqingsun96@gmail.com",
        description = "STRIDE (Spatial TRanscrIptomics DEconvolution by topic modelling) is a cell-type deconvolution tool for spatial transcriptomics. ",
        license = "GPL-3.0",
        url = "https://github.com/dongqingsun96/STRIDE",
        
        # entry_points = {"console_scripts": ["strap = strap:main"]},
        classifiers = [
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GPL License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Bio-Informatics"
        ],
        python_requires=">=3.7",
    )

if __name__ == "__main__":
    main()
