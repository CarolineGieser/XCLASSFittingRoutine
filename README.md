# XCLASS Fitting Routine
--- this readme is under construction ---
## Description
This routine performs XCLASS fits on spectra from a given sample of regions and molecules.

Information about the XCLASS Software can be found on the following website:
https://xclass.astro.uni-koeln.de/
T. Möller, C. Endres, and P. Schilke, "eXtended CASA Line Analysis Software Suite (XCLASS)",A&A 598, A7 (2017), arXiv:1508.04114


**execute routine with "python3 run.py" in terminal**
## Required Packages
*The routine has been tested successfully using the following packages:*

- casa 5.4.0-68 (with IPython 5.1.0)
- python 3.6.5
- XCLASS 1.2.5
- numpy 1.16.3
- astropy (python3: 3.2.3)
- spectralcube 0.4.4
- matplotlib 3.0.3
- os


## Required Input
**regions.dat**:  
*Table of regions*  

flux unit of the fits data cubes: Jy/beam
spectral uniz of the fits data cubes: Hz

- column 1: name of the region (avoid special characters)
- column 2: filename of the .fits datacube

**cores.dat**:
*Table of selected positions within the regions*

- column 1: region (avoid special characters)
- column 2: number
- column 3: right ascension (pixel)
- column 4: declination (pixel)

**molecules.dat**:
*Table of molecules to be fitted with XCLASS*

- column 1: XCLASS name of the molecule (may contain special characters)
- column 2: standard name of the molecule (avoid special characters)

**molecule_lines.dat**:
*Table of rest frequencies for the fitted molecules*

Each molecule needs to have at least 1 frequency range!

- column 1: XCLASS name of the molecule (may contain special characters)
- column 2: rest frequency in MHz
