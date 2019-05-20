# XCLASS Fitting Routine

## Description
automatically performs XCLASS Fit on spectra from a given sample and given molecules  

**execute routine with "python3 run.py" in terminal**  
- for each given position spectra are extracted and the noise is computed in a give line-free channel range

- the correct systemic velocity is determined by fitting the C18O line first

- XCLASS is executed for all given molecules

- a fit is considered as "good" if the modeled flux has a peak flux > 5*noise

- error estimation: spectra are scaled by +- 20% and fitted with the same method; for each parameter the error is computed by the mean standard deviation from the best-fit value

- the FITS/ folder contains all extracted spectra and XCLASS input and output files

- the PLOTS/ folder contains plots of the continuum in CONTINUUM/, systemic velocity determination in VLSR/, good XCLASS fits in GOODFIT/, bad XCLASS fits in BADFIT/

- results (XCLASS fitting parameters, barchart and histogram plots) are stored in RESULTS/


## Required Packages
*The routine has been tested successfully using the following packages:*

- casa 5.4.0 (with python 2.7.14)
- python 3.6.5
- XCLASS 1.2.5
- numpy 1.16.3
- astropy (python3: 3.1.2; python2: 2.0.12)
- spectralcube 0.4.4
- matplotlib 3.0.3


## Required Input
**input.dat**:
*Some input variables*
- first column: directory in which fits datacubes are stored (var: **data_directory**)
- second column: perform error estimation (yes/no)? (var: **do_error_estimation**)
- third column: channel where noise computation should start (var: **channel1**)
- fourth column: channel where noise computation should stop (var: **channel2**)

**regions.dat**:  
*Table of regions*    
flux unit of the fits data cubes: Kelvin!  
- first column: region name which will be used for file names (var: **regions**)
- second column: latex format of region name which will be used in plots (var: **regions_plot**)
- third column: right ascension (J2000) 
- fourth column: declination (J2000)
- fifth column: distance in kpc (var: **distances**)
- sixth column: .fits filename of the spectral line data of the region (var: **filenames**)
- seventh column: .fits filename of the continuum data of the region (var: **filenames_continuum**)

**cores.dat**:
*Table of selected positions within the regions*
- first column: region (var: **cores**)
- second column: give each position in the region a number (var: **number**)
- third column: right ascension in pixel units (var: **x_pix**)
- fourth column: declination in pixel units (var: **y_pix**)
- fifth column: give positions a classification (e.g., C:"core" or E:"envelope") (var: **core_label**)

**molecules.dat**:
*Table of molecules to be fitted with XCLASS*
- first column: XCLASS label of the molecule (var: **mol_name**)
- second column: molecule label which will be used for file names (var: **mol_name_file**)
- third column: MUSCLE label which will be used for computing the model input
- fourth column: molecule label which will be used for plotting

**molecule_ranges.dat**:
*Table of frequency ranges for the fitted molecules*

Each molecule needs to have at least 1 frequency range!
Multiple frequency ranges for a molecule are also possible!

- first column: XCLASS label of the molecule (var: **mol_ranges_name**)
- second column: lower frequency in MHz (var: **mol_ranges_low**)
- third column: upper frequency in MHz (var: **mol_ranges_upp**)
