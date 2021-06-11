# -*- coding: utf-8 -*-
# @Author: Dongqing Sun
# @E-mail: Dongqingsun96@gmail.com
# @Date:   2021-06-10 15:25:08
# @Last Modified by:   Dongqing Sun
# @Last Modified time: 2021-06-10 22:28:33


import sys,os

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Could not load setuptools. Please install the setuptools package.")


def main():
    setup(
        name = "STRIDE",
        package_dir = {'STRIDE':'STRIDE'},
        version = open("STRIDE/version.txt", "r").readline().strip(),
        packages = ['STRIDE'],
        scripts = ['STRIDE/STRIDE'],
        include_package_data = True,
        
        author = "Dongqing Sun",
        author_email = "Dongqingsun96@gmail.com",
        description = "STRIDE (Spatial TRanscRIptomics DEconvolution by topic modelling) is a cell-type deconvolution tool for spatial transcriptomics. ",
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
        ]   
    )

if __name__ == "__main__":
    main()
