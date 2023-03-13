## ACEHAL: Hyperactive Learning (HAL) Python interface for building Atomic Cluster Expansion potentials (ACE1.jl/ACE1x.jl) 

This package builds ACE interatomic potentials using Hyperactive Learning (HAL).

### Installation instructions:

1. install julia 1.8.5 and python 3.9.x (with python ase, scikit-learn, matplotlib and numpy installed)
2. run julia command (add ```Pkg.activate(".")``` to use local project and set environment variable ```JULIA_PROJECT``` accordingly) 

```using Pkg; pkg"registry add https://github.com/JuliaRegistries/General"; pkg"registry add https://github.com/ACEsuit/ACEregistry"; pkg"add ACE1, ACE1x, ASE, JuLIP"```

   make sure you have at least ACE1 version = 0.11.4 and ACE1x = 0.0.4. 
   
3. install `julia` python package to set up Python -> Julia connection 

```python -m pip install julia==0.6.1```

```python -c "import julia; julia.install()"```

4. Install this package by ```pip install . ``` or ```python setup.py install``` after cloning this repo

### References:

If using this code please reference

```
@misc{van2022hyperactive,
  doi = {10.48550/ARXIV.2210.04225},
  url = {https://arxiv.org/abs/2210.04225},
  author = {van der Oord, Cas and Sachs, Matthias and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  title = {Hyperactive Learning (HAL) for Data-Driven Interatomic Potentials},
  publisher = {arXiv},
  year = {2022},
}

@article{DUSSON2022110946,
title = {Atomic cluster expansion: Completeness, efficiency and stability},
journal = {Journal of Computational Physics},
volume = {454},
pages = {110946},
year = {2022},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2022.110946},
url = {https://www.sciencedirect.com/science/article/pii/S0021999122000080},
}
```
