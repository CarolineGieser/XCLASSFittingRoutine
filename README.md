# XCLASS Fitting Routine

## Description
- automatically performs XCLASS Fit on spectra from a given sample and given molecules


## Required Packages
*The routine has been tested successfully using the following packages:*

- casa 5.4.0 (with python2 2.7.14)
- python 3.6.5
- XCLASS 1.2.5
- numpy 1.16.3
- astropy (python3: 3.1.2; python2: 2.0.12)
- spectralcube 0.4.4
- matplotlib 3.0.3


## Required Input
**input.dat**:
*Some input variables*
- data_directory: 
- do_error_estimation:
- channel_start:
- channel_stop:

**regions.dat**:
*Table of regions*
- first column: region name which will be used for file names
- second column: latex format of region name which will be used in plots
- third column: right ascension (J2000)
- fourth column: declination (J2000)
- fifth column: distance in kpc
- sixth column: .fits filename of the region
    **flux unit of the fits data cubes: Kelvin!**

**cores.dat**:
*Table of selected positions within the regions*
- first column: region (then also input in regions.dat required)
- second column: give each position in the region a number
- third column: right ascension in pixel units
- fourth column: declination in pixel units
- fifth column: give positions a classification (e.g., C:"core" or E:"envelope")

**molecules.dat**:
*Table of molecules to be fitted with XCLASS*
- first column: XCLASS label of the molecule
- second column: molecule label which will be used for file names
- third column: MUSCLE label which will be used for computing the model input
- fourth column: molecule label which will be used for plotting

**molecule_ranges.dat**:
*Table of frequency ranges for the fitted molecules*

Each molecule needs to have at least 1 frequency range!
Multiple frequency ranges for a molecule are also possible!

- first column: XCLASS label of the molecule
- second column: lower frequency in MHz
- third column: upper frequency in MHz

## Functions

### cont_functs.py

### functs.py

### XCLASS_fit_VLSR_determination.py

### XCLASS_fit.py

### XCLASS_fit_Results_all.py

### run.py
