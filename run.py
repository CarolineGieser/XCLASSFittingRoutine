import cont_functs as contfc
import functs as fc
import os
import numpy as np

### INPUT ###
#directory where fits datacubes are stored
#absolute or relative path
data_directory = '../CORESample/DATA/'

#error estimation for XCLASS fit parameters? 
#'yes' or 'no'
do_error_estimation = 'yes'

#line-free channel range in which noise is determined in each spectrum [start:stop]
freq_low=219000.0
freq_upp=219130.0

#specific line for vlsr determination
#(otherwise set vlsr_corr=False, vlsr_molec='',vlsr_freq=np.nan to use general region value from datacube)
#True or False
vlsr_corr=True
#XCLASS molecule name
vlsr_molec='CO-18;v=0;'
#rest frequency of specific line
vlsr_freq=219560.0
###REQUIRED INPUT ###


#fit isotopologues simultaneously:
#has to be added in create_XCLASS_isoratio_file() function in functs.py

### LOADING INPUT AND SETTING UP DIRECTORIES ###
#setup working directory
working_directory = fc.setup_directory(delete_previous_results=False)
fc.create_input_table(data_directory,do_error_estimation,freq_low,freq_upp)

### CONTINUUM DATA ###

#continuum plots
contfc.plot_continuum(data_directory)

### SETTING UP XCLASS FILES ###

#determine noise in spectra
#fc.determine_noise(data_directory,freq_low,freq_upp)

#extract spectra
#fc.extract_spectrum_init(data_directory)
	
#create XCLASS input files
#fc.setup_XCLASS_files(data_directory, working_directory,do_error_estimation,vlsr_molec,vlsr_freq)

### RUN XCLASS AND EXTRACT FIT PARAMETERS ###

#run XCLASS Fit
#fc.run_XCLASS_fit(data_directory,do_error_estimation,vlsr_corr,vlsr_molec,vlsr_freq)

#extract all best fit parameters and save tables
#fc.extract_results(do_error_estimation,plotting=True,flagging=False)

### CREATE PLOTS AND EXTRACT RESULTS ###
#plot results (histograms and barcharts)
#fc.create_plots(do_error_estimation)

#compute total fit spectrum
#fc.run_XCLASS_fit_all_fixed(data_directory,working_directory,do_error_estimation,vlsr_molec,vlsr_freq)

#compute H2 column density and mass from continuum
#fc.determine_H2_col_dens(data_directory,do_error_estimation)

##create MUSCLE input files
#fc.create_MUSCLE_input(data_directory)
	
#create plot with observed + fitted spectrum
#fc.plot_fit()

#fc.abundance_analysis(do_error_estimation)



















