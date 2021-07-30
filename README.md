# STRIDE
STRIDE (Spatial TRanscrIptomics DEconvolution by topic modelling) is a cell-type deconvolution tool for spatial transcriptomics by using single-cell transcriptomics data. 

## Change Log
### v0.0.1
* Build STRIDE.

## Install STRIDE
```bash
git clone https://github.com/DongqingSun96/STRIDE.git
cd STRIDE
pip install -r requirements.txt
python setup.py install
```

## Usage
```bash
STRIDE --help
usage: STRIDE [-h] [-v] {deconvolve,plot,cluster,integrate} ...

STRIDE (Spatial TRanscrIptomics DEconvolution by topic modelling) is a cell-
type deconvolution tool for spatial transcriptomics by using single-cell
transcriptomics data.

positional arguments:
  {deconvolve,plot,cluster,integrate}
    deconvolve          Decompose celltype proportion for spatial
                        transcriptomics.
    plot                Visualize the deconvolution result.
    cluster             Neighbourhood analysis based on cell-type composition
                        and local cell population
    integrate           Integrate multiple samples from the same tissue.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         Print version info.
```
