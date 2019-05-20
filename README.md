# XCLASS Fitting Routine

## Description
- automatically performs XCLASS Fit on spectra from a given sample and given molecules


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
- sixth column: .fits filename of the region (**flux unit of the fits data cubes: Kelvin!**)

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

determine_noise_continuum(): noise computation of the continuum data

plot_continuum(): plot continuum and input positions

### functs.py

make_directory(dirName): 
create a subdirectory called 'dirName'

workingdir(): 
determine current directory, 
return **working_dir**

rm_previous_fitting_results(): 
remove previous fitting results

setup_directory(delete_previous_results=False):
creates all required directories
if delete_previous_results=True: **all** previous fitting results will be removed!

load_input_table():
loads input from input.dat table
return data_directory, do_error_estimation, channel1 , channel2

load_regions_table():
load input from regions.dat table
return regions, regions_plot, distances, filenames

load_cores_table():
load input from cores.dat table
return cores, number, x_pix, y_pix, core_label

load_molecules_table():
load input from molecules.dat table
return mol_name, mol_name_file

load_molecule_ranges_table():
load input from molecule_ranges.dat table
return mol_ranges_name, mol_ranges_low, mol_ranges_upp

check_error_estimation(do_error_estimation):
check from input if error estimation should be performed or not

determine_noise(data_directory, regions, filenames, cores, number, x_pix, y_pix,channel1,channel2):
compute noise in a spectrum within a given channel range
return std_line

extract_spectrum_init(data_directory, regions, filenames, cores, number, x_pix, y_pix)

create_XCLASS_molfits_file(cores, number,mol_name,mol_name_file,method)

create_XCLASS_obsxml_file(data_directory,working_directory,regions,filenames,cores, number,mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp, do_error_estimation,method)

carbon_12_13_ratio(d)

nitrogen_14_15_ratio(d)

oxygen_16_18_ratio(d)

sulfur_32_34_ratio()

create_XCLASS_isoratio_file(regions,data_directory,filenames,distances)

setup_XCLASS_files(data_directory, working_directory, regions, filenames, distances, cores, number, x_pix, y_pix, mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp,do_error_estimation):
	
get_velocity_offset(cores, number)

extract_spectrum(data_directory, regions, filenames, cores, number, x_pix, y_pix, v_off,do_error_estimation)

extract_properties(cores, number, x_pix, y_pix, v_off, std_line, data_directory)

rm_casa_files()

run_XCLASS_fit(data_directory, regions, filenames,cores, number, x_pix, y_pix,std_line,do_error_estimation,C18O_vlsr=True)

extract_results(cores, number, mol_name_file,std_line,do_error_estimation)

plot_results_cores(cores, number,do_error_estimation)

plot_results_molecule(mol_name_file)

create_plots(cores, number, mol_name_file,std_line,do_error_estimation)

plot_fit_residuals_optical_depth(cores, number,std_line)

run_XCLASS_fit_all_fixed(data_directory,working_directory,regions, filenames,cores, number,mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp,std_line,do_error_estimation):
	
### XCLASS_fit_VLSR_determination.py

### XCLASS_fit.py

### XCLASS_fit_Results_all.py

### run.py
