import numpy as np
from astropy.io import fits
import os

#get working directory 
working_directory = os.getcwd() + "/"

### input data
input_tab=np.loadtxt('input.dat', dtype='U', comments='#')
data_directory = input_tab[0,1].astype(np.str) #directory in which fits datacubes are stored

#### input table of regions
regions_tab=np.loadtxt('regions.dat', dtype='U', comments='#')
regions = regions_tab[:,0].astype(np.str) #CORE region name
filenames = regions_tab[:,3].astype(np.str) #fits filenames of spectral line datacubes

#### input table of cores
cores_tab=np.loadtxt('cores.dat', dtype='U', comments='#')
cores = cores_tab[:,0].astype(np.str)
number = cores_tab[:,1].astype(np.int)


#loop over all cores
for j in range(cores.size):
	
	# get fits filename of datacube
	mask = np.where(regions == cores[j])
	filename = filenames[mask]
	#load datacube
	datafile = data_directory + filename[0]
	hdu = fits.open(datafile)[0]
	
	#get beam major and minor axis and frequency resolution
	bmin = hdu.header['BMIN'] * 3600.0 #arcsec
	bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
	delta_nu = hdu.header['CDELT3'] #MHz
	
	#create average beam FWHM
	beam_avg = (bmin + bmaj ) / 2.0

	#XCLASS Input parameters
	FreqMin = 217000.0
	FreqMax = 221000.0
	FreqStep = delta_nu
	TelescopeSize = beam_avg
	Inter_Flag = True
	t_back_flag = False
	tBack = 0.0
	tslope = 0.0
	nH_flag = False
	N_H = 0.0
	beta_dust = 0.0
	kappa_1300 = 0.0
	MolfitsFileName = 'FITS/molecules_' + str(cores[j]) + '_' + str(number[j]) + '_compl.molfit'
	iso_flag = True
	IsoTableFileName = str(working_directory) + 'FITS/isotopologues_' + str(cores[j]) + '.dat'
	RestFreq = 0.0
	vLSR = 0.0
	
	#run myXCLASS
	modeldata, log, TransEnergies, IntOptical, jobDir = myXCLASS()
	
	####copy results from XCLASS default directory to working directory:
	
	#transition energies
	os.system('scp -rp ' + jobDir + 'transition_energies.dat Results/' + str(cores[j]) + '_' + str(number[j]) + '_transition_energies.dat')
	
	
	#get list of all files 
	listOfFiles = os.listdir(jobDir)  
	pattern = 'optical_depth__*.dat'
	
	#loop over all files
	for tau_file in listOfFiles:  
	
		#extract only optical depth files
		if fnmatch.fnmatch(tau_file, pattern):
			
				#copy optical depth file to working directory
				os.system('scp -rp ' + jobDir + tau_file + ' Results/' + str(cores[j]) + '_' + str(number[j]) + '_' + tau_file)
				
				
				
				
				
				
				
				
