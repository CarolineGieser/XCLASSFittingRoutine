import os
import numpy as np
from astropy import units as u
from astropy.io import fits
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.ticker import MultipleLocator
from matplotlib import ticker

###---plotting parameters---###
params = {'font.family' : 'serif',
			 'font.size' : 10,
			 'errorbar.capsize' : 3,
			 'lines.linewidth'   : 1.0,
			 'xtick.top' : True,
			 'ytick.right' : True,
			 'legend.fancybox' : False,
			 'xtick.major.size' : 4.0 ,
          'xtick.minor.size' : 2.0,    
          'ytick.major.size' : 4.0 ,
          'ytick.minor.size' : 2.0, 
			 'xtick.direction' : 'out',
			 'ytick.direction' : 'out',
			 'xtick.color' : 'black',
			 'ytick.color' : 'black',
			 'mathtext.rm' : 'serif',
          'mathtext.default': 'regular', 
			}
plt.rcParams.update(params)

###---constants---###
#speed of light
c = 299792.458 #km/s


def make_directory(dirName):
	### create a directory called 'dirName'
	
	try:
		#only create directory if it does not already exist
		
	    os.makedirs(dirName)   
	     
	    print('Directory ' + dirName +  ' created. ')
	    
	except FileExistsError:
		#do not create directories if they already exist so nothing will be overwritten
		
		print('Directory ' + dirName +  ' already exists.')	
		
		
def workingdir():
	### get working directory path
	
	working_dir = os.getcwd() + '/'
	
	print('Working Directory: ' + working_dir)
	
	return working_dir
	
	
def rm_previous_fitting_results():
	### !!!WARNING!!! ALL previous results will be lost!
	
	#remove directories in which results are stored
	os.system('rm -r FITS')
	os.system('rm -r PLOTS')
	os.system('rm -r RESULTS')
	
	print('Removed previous results!')
	
	
def setup_directory(delete_previous_results=False):
	### setup all directories in which fit results are stored
	
	if delete_previous_results==True:
		# remove previous results		
		rm_previous_fitting_results()
		
	# get working directory
	working_directory = workingdir()
	
	#XCLASS setup files and fit results
	make_directory(dirName='FITS')
	
	#plots directory
	make_directory(dirName='PLOTS')
	#bad XCLASS fits
	make_directory(dirName='PLOTS/BADFITS')
	#good XCLASS fits
	make_directory(dirName='PLOTS/GOODFITS')
	#vlsr detemination fit
	make_directory(dirName='PLOTS/VLSR_DETERMINATION')
	#results directory
	make_directory(dirName='RESULTS')
	#XCLASS spectra
	make_directory(dirName='RESULTS/SPECTRA')
	#barcharts for each core
	make_directory(dirName='RESULTS/BARCHARTS')
	#histograms for each molecule
	make_directory(dirName='RESULTS/HISTOGRAMS')
	#tables (e.g. noise,...)
	make_directory(dirName='RESULTS/TABLES')
	
	print('Working directory: ' + working_directory)
	
	return working_directory

	
def load_init_table():
	###create init.dat file which contains some input parameters from "run.py"
	
	#load table
	input_tab=np.loadtxt('Input/init.dat', dtype='U', comments='#')
	
	data_directory = input_tab[0,1].astype(np.str)
	do_error_estimation = input_tab[1,1].astype(np.str)
	freq_low = input_tab[2,1].astype(np.float)
	freq_upp = input_tab[3,1].astype(np.float)
	vlsr_corr = input_tab[4,1].astype(np.str)
	vlsr_molec = input_tab[5,1].astype(np.str)
	vlsr_freq = input_tab[6,1].astype(np.float)
	sourcesize_low = input_tab[7,1].astype(np.str)
	sourcesize_upp = input_tab[8,1].astype(np.str)
	temperature_low = input_tab[9,1].astype(np.str)
	temperature_upp = input_tab[10,1].astype(np.str)
	columndensity_low = input_tab[11,1].astype(np.str)
	columndensity_upp = input_tab[12,1].astype(np.str)
	linewidth_low = input_tab[13,1].astype(np.str)
	linewidth_upp = input_tab[14,1].astype(np.str)
	velocityoffset_low = input_tab[15,1].astype(np.str)
	velocityoffset_upp = input_tab[16,1].astype(np.str)
	
	return data_directory, do_error_estimation, freq_low, freq_upp, vlsr_corr, vlsr_molec, vlsr_freq, sourcesize_low, sourcesize_upp, temperature_low, temperature_upp, columndensity_low, columndensity_upp, linewidth_low, linewidth_upp, velocityoffset_low, velocityoffset_upp
		
	
def load_regions_table():
	#### input table of regions
	
	#load table
	regions_tab=np.loadtxt('Input/regions.dat', dtype='U', comments='#',ndmin=2)
	
	#region name
	regions = regions_tab[:,0].astype(np.str) 
	#fits filenames of spectral line datacubes
	filenames_line = regions_tab[:,1].astype(np.str) 
	
	return regions, filenames_line


def load_cores_table():
	#### input table of cores
	
	#load table
	cores_tab=np.loadtxt('Input/cores.dat', dtype='U', comments='#',ndmin=2)
	
	#region name
	cores = cores_tab[:,0].astype(np.str)
	#core numbers
	numbers = cores_tab[:,1].astype(np.int)
	#position in RA (pixel)
	x_pix = cores_tab[:,2].astype(np.int)
	#position in DEC (pixel)
	y_pix = cores_tab[:,3].astype(np.int)
		
	return cores, numbers, x_pix, y_pix
	
	
def load_molecules_table():
	### input table of molecules
	
	#load table
	mol_tab=np.loadtxt('Input/molecules.dat', dtype='U', comments='%',ndmin=2)
	
	#XCLASS molecule label
	mol_names_XCLASS = mol_tab[:,0].astype(np.str)
	#filename molecule label
	mol_names_file=mol_tab[:,1].astype(np.str)
	
	return mol_names_XCLASS, mol_names_file
	
	
def load_molecule_lines_table():
	### input table of molecule lines to fit
	
	#load table
	mol_lines_tab=np.loadtxt('Input/molecule_lines.dat', dtype='U', comments='%',ndmin=2)
	
	#XCLASS molecule label
	mol_lines_name=mol_lines_tab[:,0]
	#rest frequency of transition
	mol_lines_freq=mol_lines_tab[:,1].astype(np.float)
	
	return mol_lines_name, mol_lines_freq
	

#load all input tables
data_directory, do_error_estimation, freq_low, freq_upp, vlsr_corr, vlsr_molec, vlsr_freq, sourcesize_low, sourcesize_upp, temperature_low, temperature_upp, columndensity_low, columndensity_upp, linewidth_low, linewidth_upp, velocityoffset_low, velocityoffset_upp = load_init_table()
regions, filenames_line = load_regions_table()
cores, numbers, x_pix, y_pix = load_cores_table()
mol_names_XCLASS, mol_names_file = load_molecules_table()
mol_lines_name, mol_lines_freq = load_molecule_lines_table()


def check_error_estimation():
	###check if error estimation should be performed
	
	#create 3 tags for normal spectrum, and lower, and upper error estimation
	if do_error_estimation == 'yes':
		
		tag = np.array(['','_lowErr','_uppErr'])
		
	#create only 1 tag for normal spectrum
	elif do_error_estimation == 'no':
		
		tag = np.array([''])
		
	#raise an error if do_error_estimation is neither 'yes' nor 'no'
	else:
		print('Only yes and no are allowed for do_error_estimation parameter in init.dat!')
		
	return tag


def convert_to_brightness_temperature(X,Y,bmaj,bmin):
	###convert flux density (Jy/beam) to brightness temperature (K)
	#X in frequency (input with unit,e.g. array*u.Hz)
	#Y in Jy/beam (without units)
	#bmaj, bmin in arcsec (without units)

	#conversion factor from fwhm to standard deviation
	fwhm_to_sigma = 1.0/(8.0*np.log(2.0))**0.5
	#compute beam area
	beam_area = 2.0*np.pi*((bmaj*u.arcsec)*(bmin*u.arcsec)*fwhm_to_sigma**2)
	
	#empty array for converted values
	Y_K = np.zeros(Y.size)*u.K
	
	#loop over all channels
	for i in range(Y.size):
	
		#convert Jy/beamto K
		Y_K[i] = (Y[i] *u.Jy/beam_area).to(u.K, equivalencies=u.brightness_temperature(X[i]))
		
	return Y_K


def determine_noise():
	### compute noise in each spectrum in a line-free channel range
	
	#array in which noise values are stored
	std_line = np.zeros(cores.size)
	
	#loop over all cores
	for j in range(cores.size):
		
		# get fits filename of datacube
		mask = np.where(regions == cores[j])
		filename = filenames_line[mask]
		
		#load datacube
		datafile = data_directory + filename[0] 
		hdu = fits.open(datafile)[0]
		#get rest-frequency of observations
		restfreq = hdu.header['RESTFREQ'] #Hz
		#get systemic velocity of observations
		vlsr = hdu.header['VELO-LSR'] / 1000.0 #km/s
		#get beam major and minor axis
		bmin = hdu.header['BMIN'] * 3600.0 #arcsec 
		bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
		
		#convert spectral axis to MHz
		cube = SpectralCube.read(datafile).with_spectral_unit(u.MHz, velocity_convention='radio',rest_value=restfreq * u.Hz)  
		cube = cube.spectral_slab(freq_low*(1.0 - (vlsr/c))*u.MHz, freq_upp*(1.0 - (vlsr/c))*u.MHz) 

		#extract spectrum
		subcube = cube[:,y_pix[j],x_pix[j]]
		#extract line-free part of spectrum
		subcube = convert_to_brightness_temperature(subcube.spectral_axis,subcube.value,bmaj,bmin)
		
		#compute standard deviation of the spectrum
		std_line[j] = np.around(np.std(subcube.value), decimals=2)
		
		#print results
		print('Determined Noise in ' + str(cores[j]) + ' ' + str(numbers[j]) + ' Spectrum: ' + str(std_line[j])+ ' K')
		
	#save noise values in file
	np.savetxt('RESULTS/TABLES/noise.dat', np.c_[cores,numbers,std_line], delimiter=' ', fmt='%s')
	
	return std_line
	

def extract_spectrum_init():
	### extract and save spectrum of each core which will be fitted with XCLASS (only specific line)
	
	#loop over all cores
	for j in range(cores.size):
		
		# get fits filename of datacube
		mask = np.where(regions == cores[j])
		filename = filenames_line[mask]
		#load datacube
		datafile = data_directory + filename[0]
		hdu = fits.open(datafile)[0]
		
		#get systemic velocity of observations
		vlsr = hdu.header['VELO-LSR'] / 1000.0 #km/s
		#get rest-frequency of observations
		restfreq = hdu.header['RESTFREQ'] #Hz
		#get beam major and minor axis
		bmin = hdu.header['BMIN'] * 3600.0 #arcsec 
		bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
		#convert spectral axis to MHz
		cube = SpectralCube.read(datafile).with_spectral_unit(u.MHz, velocity_convention='radio',rest_value=restfreq * u.Hz)  
	
		#extract spectrum
		sub_cube = cube[:,y_pix[j],x_pix[j]]
		#extract spectral axis
		X = sub_cube.spectral_axis/(1.0 - (vlsr/c))
		#convert to brightness temperature
		Y = convert_to_brightness_temperature(X,sub_cube.value,bmaj,bmin)
	
		#save spectra
		np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '.dat', np.c_[X.value,Y.value])
		
		#print results
		print('Extracted ' + str(cores[j]) + ' ' + str(numbers[j]) + ' Spectrum!')
		
		
def create_XCLASS_molfits_file(method):
	### create input molfits for XCLASS fitting
	
	#molfits file for initial C18O fit
	if method == 'vLSR':
		
		#open and write molfits file
		file = open('FITS/molecules_vLSR.molfit','w') 
		file.write(
vlsr_molec + """   1 \n"""
"""y   """ + sourcesize_low + """   """ + sourcesize_upp + """   0.4   y   """ + temperature_low + """   """ + temperature_upp + """   100.0   y   """ + columndensity_low + """   """ + columndensity_upp + """   1.0e+15   y   """ + linewidth_low + """   """ + linewidth_upp + """   5.0   y   -15.0   15.0   0.0   c"""
		)
		file.close()

	#molfits file for molecule fits
	elif method == 'single':
		
		#loop over all molecules
		for k in range(mol_names_XCLASS.size): 
			
			#open and write molfits file
			file = open('FITS/molecules_' + str(mol_names_file[k]) + '.molfit','w') 
			file.write(
mol_names_XCLASS[k] +"""   1 \n"""
"""y   """ + sourcesize_low + """   """ + sourcesize_upp + """   0.4   y   """ + temperature_low + """   """ + temperature_upp + """   100.0   y   """ + columndensity_low + """   """ + columndensity_upp + """   1.0e+15   y   """ + linewidth_low + """   """ + linewidth_upp + """   5.0   y   """ + velocityoffset_low + """   """ + velocityoffset_upp + """   0.0   c"""
			)
			file.close()
					
	#molfits file for final fit at fixed parameters		
	elif method == 'all':
		
		#loop over all cores
		for j in range(cores.size):
			
			#open and write molfits file
			file = open('FITS/molecules_' + str(cores[j]) + '_' + str(numbers[j]) + '_compl.molfit','w') 
			
			#loop over all fitted molecules
			for k in range(mol_names_XCLASS.size):
				
				#load fit results
				fitdata = np.loadtxt('RESULTS/TABLES/results_' + str(mol_names_file[k]) +'.dat', dtype='U', comments='#',ndmin=2)
				
				#source size
				Theta_source = fitdata[:,2]
				#temperature
				T = fitdata[:,3]
				#column density
				N = fitdata[:,4]
				#line width
				Delta_v = fitdata[:,5]
				#velocity offset
				v_off = fitdata[:,6]
				
				#only add molecules with good fits in molfits file
				if N[j] != 'nan':
					file.write(
mol_names_XCLASS[k] +"""   1 \n"""
"""n   0.1   1.0   """+ str(Theta_source[j]) +"""   n   5.0   300.0   """+ str(T[j]) +"""   n   1.0E+10   1.0E+21   """+ str(N[j]) +"""   n   0.5   20.0   """+ str(Delta_v[j]) +"""   n   -6.0   6.0    """+ str(v_off[j]) +"""   c\n"""
					)
			file.close()
			
	# raise error if "method" keyword is wrong	
	else:
		
		raise Exception('Only method = vLSR, single or all possible!')
		
	print('Created molfits files!') 
	
	
def create_XCLASS_obsxml_file(working_directory,method):
	### create input observations xml files for XCLASS fitting
	
	#observational xml file for vlsr detemination fit
	if method == 'vLSR':

		
		#loop over all cores
		for j in range(cores.size):
			
			# get fits filename of datacube
			mask = np.where(regions == cores[j])
			filename = filenames_line[mask]
			#load datacube
			datafile = data_directory + filename[0]
			cube = SpectralCube.read(datafile)  
			hdu = fits.open(datafile)[0]
			
			#get beam major and minor axis and frequency resolution
			bmin = hdu.header['BMIN'] * 3600.0 #arcsec
			bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
			#get rest-frequency of observations
			restfreq = hdu.header['RESTFREQ'] #Hz
			#calculate frequency spectral resolution in MHz
			delta_nu = restfreq * np.abs(hdu.header['CDELT3']) / (c * 10.0**9.0) #MHz
			
			#create average beam FWHM
			beam_avg = (bmin + bmaj ) / 2.0
		
			#open and write observation.xml file (optimized for C18O line!)
			file = open('FITS/observation_' + str(cores[j]) + '_' + str(numbers[j]) + '_vLSR.xml','w') 
			file.write(
"""<?xml version="1.0" encoding="UTF-8"?>
<ExpFiles>
 <NumberExpFiles>1</NumberExpFiles>
 <file>
     <FileNamesExpFiles>""" + str(working_directory) + """FITS/spectrum_""" + str(cores[j]) + """_""" + str(numbers[j]) + """.dat</FileNamesExpFiles>
     <ImportFilter>xclassASCII</ImportFilter>
     <NumberExpRanges>1</NumberExpRanges>
     <FrequencyRange>
         <MinExpRange>""" + np.str(vlsr_freq - 20.0) + """</MinExpRange>
         <MaxExpRange>"""+ np.str(vlsr_freq + 20.0) + """</MaxExpRange>
         <StepFrequency>""" + str(delta_nu) + """</StepFrequency>
         <t_back_flag>False</t_back_flag>
         <BackgroundTemperature>0.0</BackgroundTemperature>
         <TemperatureSlope>0.0</TemperatureSlope>
         <HydrogenColumnDensity>0.e+0</HydrogenColumnDensity>
         <DustBeta>0.0</DustBeta>
         <Kappa>0.0</Kappa>
     </FrequencyRange>
     <GlobalvLSR>0.0</GlobalvLSR>
     <TelescopeSize>""" + str(beam_avg) + """</TelescopeSize>
     <Inter_Flag>True</Inter_Flag>
     <ErrorY>no</ErrorY>
     <NumberHeaderLines>0</NumberHeaderLines>
     <SeparatorColumns> </SeparatorColumns>
     </file>
     <iso_flag>True</iso_flag>
     <IsoTableFileName>""" + str(working_directory) + """Input/isotopologues_""" + str(cores[j]) + """.dat</IsoTableFileName>
</ExpFiles>"""
			)
			file.close() 
	
	#observation xml file for molecule fits
	elif method == 'single':
		
		# check if error estimation or not
		tag = check_error_estimation()
		
		#loop over all molecules
		for k in range(mol_names_XCLASS.size):
			
			#load line frequencies for molecule
			mask = np.where(mol_lines_name == mol_names_XCLASS[k])
			mol_lines_name_mask = mol_lines_name[mask]
			mol_lines_freq_mask = mol_lines_freq[mask]
			
			#loop over all tags (for error estimation)
			for z in range(tag.size):
				
				#loop over all cores
				for j in range(cores.size):
					
					# get fits filename of datacube
					mask = np.where(regions == cores[j])
					filename = filenames_line[mask]
					#load datacube
					datafile = str(data_directory) + filename[0]
					cube = SpectralCube.read(datafile)  
					hdu = fits.open(datafile)[0]
					
					#get beam major and minor axis and frequency resolution
					bmin = hdu.header['BMIN'] * 3600.0 #arcsec 
					bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
					#get rest-frequency of observations
					restfreq = hdu.header['RESTFREQ'] #Hz
					#calculate frequency spectral resolution in MHz
					delta_nu = restfreq * np.abs(hdu.header['CDELT3']) / (c * 10.0**9.0) #MHz
					
					#create average beam FWHM
					beam_avg = (bmin + bmaj ) / 2.0
					
					#open and write observation xml file
					file = open('FITS/observation_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + str(tag[z]) + '.xml','w') 
					file.write(
"""<?xml version="1.0" encoding="UTF-8"?>
<ExpFiles>
 <NumberExpFiles>1</NumberExpFiles>
 <file>
     <FileNamesExpFiles>""" + str(working_directory) +"""FITS/spectrum_""" + str(cores[j]) + """_""" + str(numbers[j]) + str(tag[z]) + """_vLSRcorr.dat</FileNamesExpFiles>
     <ImportFilter>xclassASCII</ImportFilter>
     <NumberExpRanges>""" + str(mol_lines_name_mask.size) + """</NumberExpRanges>\n"""
			        )
			        
			      #loop over all lines
					for p in range(mol_lines_name_mask.size):
						file.write(
"""     <FrequencyRange>
         <MinExpRange>""" + str(mol_lines_freq_mask[p] - 20.0) + """</MinExpRange>
         <MaxExpRange>""" + str(mol_lines_freq_mask[p] + 20.0) + """</MaxExpRange>
         <StepFrequency>""" + str(delta_nu) + """</StepFrequency>
         <t_back_flag>False</t_back_flag>
         <BackgroundTemperature>0.0</BackgroundTemperature>
         <TemperatureSlope>0.0</TemperatureSlope>
         <HydrogenColumnDensity>0.e+0</HydrogenColumnDensity>
         <DustBeta>0.0</DustBeta>
         <Kappa>0.0</Kappa>
     </FrequencyRange>\n"""
						)
					        
					file.write(
"""     <GlobalvLSR>0.0</GlobalvLSR>
     <TelescopeSize>""" + str(beam_avg) + """</TelescopeSize>
	  <Inter_Flag>True</Inter_Flag>
	  <ErrorY>no</ErrorY>
	  <NumberHeaderLines>0</NumberHeaderLines>
	  <SeparatorColumns> </SeparatorColumns>
	  </file>
	  <iso_flag>True</iso_flag>
     <IsoTableFileName>""" + str(working_directory) + """Input/isotopologues_""" + str(cores[j]) + """.dat</IsoTableFileName>
</ExpFiles>"""
					)
					file.close() 
					
	#observation xml file for final fit at fixed parameters					
	elif method == 'all':
		
		#loop over all cores
		for j in range(cores.size):
			
			#load spectrum to get upper and lower frequency bounds
			data_tab = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_vLSRcorr.dat',ndmin=2)
			#get spectral axis
			X = data_tab[:,0]
			
			# get fits filename of datacube
			mask = np.where(regions == cores[j])
			filename = filenames_line[mask]
			#load datacube
			datafile = data_directory + filename[0]
			cube = SpectralCube.read(datafile)  
			hdu = fits.open(datafile)[0]
			
			#get beam major and minor axis and frequency resolution
			bmin = hdu.header['BMIN'] * 3600.0 #arcsec
			bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
			#get rest-frequency of observations
			restfreq = hdu.header['RESTFREQ'] #Hz
			#calculate frequency spectral resolution in MHz
			delta_nu = restfreq * np.abs(hdu.header['CDELT3']) / (c * 10.0**9.0) #MHz
			
			#create average beam FWHM
			beam_avg = (bmin + bmaj ) / 2.0
		
			#open and write observation.xml file
			file = open('FITS/observation_' + str(cores[j]) + '_' + str(numbers[j]) + '_all.xml','w') 
			file.write(
"""<?xml version="1.0" encoding="UTF-8"?>
<ExpFiles>
 <NumberExpFiles>1</NumberExpFiles>
 <file>
     <FileNamesExpFiles>""" + str(working_directory) +"""FITS/spectrum_""" + str(cores[j]) + """_""" + str(numbers[j]) + """_vLSRcorr.dat</FileNamesExpFiles>
     <ImportFilter>xclassASCII</ImportFilter>
     <NumberExpRanges>1</NumberExpRanges>
     <FrequencyRange>
         <MinExpRange>""" + np.str(np.nanmin(X)-10.0) + """</MinExpRange>
         <MaxExpRange>""" + np.str(np.nanmax(X)+10.0) + """</MaxExpRange>
         <StepFrequency>""" + str(delta_nu) + """</StepFrequency>
         <t_back_flag>False</t_back_flag>
         <BackgroundTemperature>0.0</BackgroundTemperature>
         <TemperatureSlope>0.0</TemperatureSlope>
         <HydrogenColumnDensity>0.e+0</HydrogenColumnDensity>
         <DustBeta>0.0</DustBeta>
         <Kappa>0.0</Kappa>
     </FrequencyRange>
     <GlobalvLSR>0.0</GlobalvLSR>
     <TelescopeSize>""" + str(beam_avg) + """</TelescopeSize>
     <Inter_Flag>True</Inter_Flag>
     <ErrorY>no</ErrorY>
     <NumberHeaderLines>0</NumberHeaderLines>
     <SeparatorColumns> </SeparatorColumns>
     </file>
     <iso_flag>True</iso_flag>
     <IsoTableFileName>""" + str(working_directory) + """Input/isotopologues_""" + str(cores[j]) + """.dat</IsoTableFileName>
</ExpFiles>"""
			)
			file.close() 
			
	# raise error if "method" keyword is wrong			
	else:
		
		raise Exception('Only method = vLSR, single or all possible!')
		
	print('Created observation xml files!') 
	
	
def setup_XCLASS_files(working_directory):
	### create isoratio, molfits and xml files
	
	#molfits files for vlsr determination
	create_XCLASS_molfits_file(method='vLSR')
	#xml files for vlsr determination
	create_XCLASS_obsxml_file(working_directory, method='vLSR')
	
	#molfits files for molecule fits
	create_XCLASS_molfits_file(method='single')
	#xml files for molecule fits
	create_XCLASS_obsxml_file(working_directory,method='single')
	

def get_velocity_offset():
	### extract bestfit systemic velocity and plot fitted vlsr determination line
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#load noise table
	noise = np.loadtxt('RESULTS/TABLES/noise.dat',usecols=2)
	
	#empty array in which v_off values are stored
	VelocityOffset = np.zeros(cores.size)
	
	#loop over all cores
	for j in range(cores.size):
		
		#load spectrum
		data = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '.dat')
		X = data[:,0] 
		Y = data[:,1]
		
		#load fit		
		fit = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_vLSR.out.dat')
		X_fit = fit[:,0] 
		Y_fit = fit[:,1]
		
		#plot spectrum and fit
		plt.ioff()
		plt.figure() 
		#create subplot
		ax = plt.subplot(1, 1, 1)
		
		#plot spectrum
		plt.step(X,Y,'k', label='Data', where='mid')
		#plot fit
		plt.step(X_fit,Y_fit,'tab:red', label='Fit',linestyle='--', where='mid')
		#plot 3 sigma threshold
		plt.axhline(y=3.0*noise[j], xmin=0, xmax=1,color='tab:green', linestyle=':', label=r'3$\sigma$')
		#plotting parameters
		plt.xlim(np.amin(X_fit)-0.001,np.amax(X_fit)+0.001)
		plt.xlabel('Frequency (GHz)')
		plt.ylabel('Brightness Temperature (K)')
		plt.ticklabel_format(useOffset=False)
		plt.annotate(str(cores[j]) + ' ' + str(numbers[j]), xy=(0.1, 0.9), xycoords='axes fraction')  
		plt.legend(loc='upper right')
		#apply ylimits depending on highest brightness temperature in fitted range
		if np.amax(Y_fit) > 5.0*noise[j]:
			plt.ylim(-5.0*noise[j],np.amax(Y_fit*1.5))
		else:
			plt.ylim(-5.0*noise[j],10.0*noise[j])
		#save plot
		plt.savefig('PLOTS/VLSR_DETERMINATION/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_vLSR.png', format='png', bbox_inches='tight')
		plt.close()
		
		#extract v_off
		#load fit results table
		fit_results = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(numbers[j]) + '_vLSR.molfit', dtype='U', skiprows=1) 
		#get best fit velocity offset
		VelocityOffset[j] = fit_results[19] #km/s
	
	#save velocity offset table
	np.savetxt('RESULTS/TABLES/velocity_offset.dat', np.c_[cores,numbers,VelocityOffset], delimiter=' ', fmt='%s')
	
	print('Extracted all velocity offsets!')
	
	return VelocityOffset
	
		
def extract_spectrum(v_off):
	### extract and save spectra of each core which will be fitted with XCLASS
	
	#loop over all cores
	for j in range(cores.size):
		
		# get fits filename of datacube
		mask = np.where(regions == cores[j])
		filename = filenames_line[mask]
		#load datacube
		datafile = data_directory + filename[0]
		hdu = fits.open(datafile)[0]
		
		#get systemic velocity of observations
		vlsr = (hdu.header['VELO-LSR'] / 1000.0) #km/s
		#get rest-frequency of observations
		restfreq = hdu.header['RESTFREQ'] #Hz
		#get beam major and minor axis
		bmin = hdu.header['BMIN'] * 3600.0 #arcsec 
		bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
		#convert spectral axis to MHz
		cube = SpectralCube.read(datafile).with_spectral_unit(u.MHz, velocity_convention='radio',rest_value=restfreq * u.Hz)
	
		#extract spectrum
		sub_cube = cube[:,y_pix[j],x_pix[j]]
		
		#compute core systemic velocity from vlsr detemination fit
		vlsr_core = vlsr+v_off[j]
		
		#extract spectral axis
		X = sub_cube.spectral_axis/(1.0 - (vlsr_core/c))
		
		#convert to brightness temperature
		Y = convert_to_brightness_temperature(X,sub_cube.value,bmaj,bmin)
	
		#save spectra
		np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_vLSRcorr.dat', np.c_[X.value,Y.value])
		
		#get tags if or if not error estimation should be performed
		tag = check_error_estimation()
		
		if do_error_estimation =='yes':
			
			#scale spectra by +-20 % to estimate how fit parameters vary
			np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + tag[1] + '_vLSRcorr.dat', np.c_[X.value,Y.value*0.8])
			np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + tag[2] +'_vLSRcorr.dat', np.c_[X.value,Y.value*1.2])
		
		print('Extracted ' + str(cores[j]) + ' ' + str(numbers[j]) + ' Spectrum (spectral axis corrected for systemic velocity)!')
	
	
def rm_casa_files():
	### remove unnecessary casa files
	
	os.system('rm casa*.log')
	os.system('rm *.last')
	
	print('Removed casa log files!')
	

def extract_results():
	###extract best fit parameters from molfit files
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#load noise table
	std_line = np.loadtxt('RESULTS/TABLES/noise.dat', delimiter=' ', usecols=2, ndmin=2)	
	
	#create empty best fit parameter arrays
	#source size
	Theta_source = np.zeros(shape=(mol_names_file.size,cores.size))
	#temperature
	T = np.zeros(shape=(mol_names_file.size,cores.size))
	#column density
	N = np.zeros(shape=(mol_names_file.size,cores.size))
	#line width
	Delta_v = np.zeros(shape=(mol_names_file.size,cores.size))
	#velocity offset
	v_off = np.zeros(shape=(mol_names_file.size,cores.size))
	
	#create empty error parameter array
	if do_error_estimation == 'yes':
		
		#source size err
		Theta_source_err = np.zeros(shape=(mol_names_file.size,cores.size))
		#temperature
		T_err = np.zeros(shape=(mol_names_file.size,cores.size))
		#column density
		N_err = np.zeros(shape=(mol_names_file.size,cores.size))
		#line width
		Delta_v_err = np.zeros(shape=(mol_names_file.size,cores.size))
		#velocity offset
		v_off_err = np.zeros(shape=(mol_names_file.size,cores.size))

	#loop over all fitted molecules
	for k in range(mol_names_file.size):
		
		#loop over all cores
		for j in range(cores.size):
			
			#load data
			data = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_vLSRcorr.dat')
			#frequency
			X = data[:,0]/1000.0 #GHz
			#brightness temperature
			Y = data[:,1] #K
			
			#load fit
			fit = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + '.out.dat')
			#fit frequency			
			X_fit = fit[:,0]/1000.0 #GHz
			#fit brightness temperature
			Y_fit = fit[:,1] #K
			
			#load scaled spectra if error estimation is performed
			if do_error_estimation == 'yes':
				
				#load data
				fit_lowErr = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + '_lowErr.out.dat')
				#fit frequency
				X_fit_lowErr = fit_lowErr[:,0]/1000.0 #GHz
				#fit brightness temperature
				Y_fit_lowErr = fit_lowErr[:,1] #K
				
				#load data
				fit_uppErr = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + '_uppErr.out.dat')
				#fit frequency
				X_fit_uppErr = fit_uppErr[:,0]/1000.0 #GHz
				#fit brightness temperature
				Y_fit_uppErr = fit_uppErr[:,1] #K
				
			#load line frequencies for molecule
			mask = np.where(mol_lines_name == mol_names_XCLASS[k])
			mol_lines_name_mask = mol_lines_name[mask]
			mol_lines_freq_mask = mol_lines_freq[mask]
			
			#compute numbers of rows for subplots
			if mol_lines_name_mask.size % 2 == 1:
				n_row = np.int(mol_lines_name_mask.size/2 + 0.5)
			if mol_lines_name_mask.size % 2 == 0:
				n_row = np.int(mol_lines_name_mask.size/2)
			if mol_lines_name_mask.size == 1:
				n_row = 1
			

			
			#plot spectrum and fit
			plt.ioff()
			plt.figure(figsize=(6.0,3.0*n_row)) 
			
			#loop over all molecule ranges
			for p in range(mol_lines_name_mask.size):
				
				#create subplot
				ax = plt.subplot(n_row, 2, p+1)
				#plot spectrum
				plt.step(X,Y,'k', label='Data', where='mid')
				#plot fit
				plt.step(X_fit,Y_fit,'tab:red', label='Fit',linestyle='dashed', where='mid')
				
				#load scaled fitted spectra if error estimation is performed
				if do_error_estimation == 'yes':
					
					#upper error estimation
					plt.step(X_fit_uppErr,Y_fit_uppErr,'tab:olive', label='upper error',linestyle='dashed', where='mid')
					#lower error estimation
					plt.step(X_fit_lowErr,Y_fit_lowErr,'tab:cyan', label='lower error',linestyle='dashed', where='mid')
				
				#plot 3 sigma line	
				plt.axhline(y=3.0*std_line[j], xmin=0, xmax=1,color='tab:green', linestyle=':', label=r'3$\sigma$')
				
				#limits in frequency
				plt.xlim((mol_lines_freq_mask[p]-20.0)/1000.0,(mol_lines_freq_mask[p]+20.0)/1000.0)
				
				#annotate y-axis to left panels
				if p % 2 == 0:
					plt.ylabel('Brightness Temperature (K)')
					
				#annotate x-axis to bottom panels
				if (p == mol_lines_name_mask.size-1) or (p == mol_lines_name_mask.size-2):
					plt.xlabel('Rest Frequency (GHz)')
					
				#no offset for ticklabels
				plt.ticklabel_format(useOffset=False)
				
				#annotate legend in first panel
				if p == 0:
					plt.annotate(str(cores[j]) + ' ' + str(numbers[j]) + ' ' + str(mol_names_XCLASS[k]), xy=(0.1, 1.1), xycoords='axes fraction')  
				if p == mol_lines_name_mask.size-1:
					plt.legend(bbox_to_anchor=(0.0, 1.0, 1.9, 0.0),loc='upper right')
				
				#mask to fitted frequency range
				mask, = np.where((X_fit > (mol_lines_freq_mask[p]-20.0)/1000.0) & (X_fit < (mol_lines_freq_mask[p]+20.0)/1000.0))

				#apply ylimits depending on highest brightness temperature in fitted range
				if mask.size > 0:
					
					if np.amax(Y_fit[mask]) > 5.0*std_line[j]:
						plt.ylim(-5.0*std_line[j],np.amax(Y_fit[mask]*2))
						
					else:							
						plt.ylim(-5.0*std_line[j],10.0*std_line[j])
		
			
			#empty lists where fitted data is stored in
			X_fit_mask = []
			Y_fit_mask = []
			
			#loop over all fitted transitions
			for p in range(mol_lines_name_mask.size):
				
				#find frequency range of transition
				mask, = np.where((X_fit > (mol_lines_freq_mask[p]-20.0)/1000.0) & (X_fit < (mol_lines_freq_mask[p]+20.0)/1000.0))
				X_fit_mask.append(X_fit[mask])
				Y_fit_mask.append(Y_fit[mask])
				
			#convert list to array
			X_fit_mask = np.asarray(X_fit_mask)
			Y_fit_mask = np.asarray(Y_fit_mask)
			
			#create 1D array
			X_fit_mask = np.concatenate(X_fit_mask).ravel()
			Y_fit_mask = np.concatenate(Y_fit_mask).ravel()
				
				
			#check that list contains values	
			if (Y_fit_mask.size > 0) and (np.amax(Y_fit_mask) > 3.0*std_line[j]):
				
				#save plot
				plt.savefig('PLOTS/GOODFITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + '.png', format='png', bbox_inches='tight')
				plt.close()
			
				#load best fit molfits file
				fit_table = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + '.molfit', dtype='U', skiprows=1) 
				
				#source size
				Theta_source[k,j] = fit_table[3]
				#temperature
				T[k,j] = fit_table[7]
				#column density
				N[k,j] = fit_table[11]
				#line width
				Delta_v[k,j] = fit_table[15]
				#velocity offset
				v_off[k,j] = fit_table[19] 
				
				#error computation
				if do_error_estimation == 'yes':
					
					#load best fit molfits file (upper error)
					fit_table_upp = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + '_uppErr.molfit', dtype='U', skiprows=1) 
				
					#load best fit molfits file (lower error)
					fit_table_low = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + '_lowErr.molfit', dtype='U', skiprows=1) 
					
					#for all parameters except N, compute mean standard deviation from best fit
					#source size error
					Theta_source_err[k,j] = (np.abs(np.float(fit_table_upp[3])-np.float(fit_table[3])) + np.abs(np.float(fit_table_low[3])-np.float(fit_table[3]))) * 0.5
					#temperature error
					T_err[k,j] = (np.abs(np.float(fit_table_upp[7])-np.float(fit_table[7])) + np.abs(np.float(fit_table_low[7])-np.float(fit_table[7]))) * 0.5
					#column density error
					N_err[k,j] = (np.abs(np.float(fit_table_upp[11])-np.float(fit_table[11])) + np.abs(np.float(fit_table_low[11])-np.float(fit_table[11]))) * 0.5
					#line width error
					Delta_v_err[k,j] = (np.abs(np.float(fit_table_upp[15])-np.float(fit_table[15])) + np.abs(np.float(fit_table_low[15])-np.float(fit_table[15]))) * 0.5
					#velocity offset error
					v_off_err[k,j] = (np.abs(np.float(fit_table_upp[19])-np.float(fit_table[19])) + np.abs(np.float(fit_table_low[19])-np.float(fit_table[19]))) * 0.5
					
	
			#no flux was recovered from XCLASS fit, set best fit parameters to nan
			else:
				

				#save plot
				plt.ylim(-5.0*std_line[j],5*std_line[j])
				plt.savefig('PLOTS/BADFITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + '.png', format='png', bbox_inches='tight')
				plt.close()
						
				#set best fit parameters to nan
				#source size
				Theta_source[k,j] = np.nan					
				#temperature
				T[k,j] = np.nan
				#column density
				N[k,j] = np.nan
				#line width
				Delta_v[k,j] = np.nan
				#velocity offset
				v_off[k,j] = np.nan
				
				if do_error_estimation == 'yes':
					#source size error
					Theta_source_err[k,j] = np.nan
					#temperature error	
					T_err[k,j] = np.nan
					#column density error
					N_err[k,j] = np.nan
					#line width error
					Delta_v_err[k,j] = np.nan
					#velocity offset error
					v_off_err[k,j] = np.nan
						
					
	#create output table for each molecule	
	for k in range(mol_names_file.size):
		
		if do_error_estimation == 'yes':
			np.savetxt('RESULTS/TABLES/results_' + str(mol_names_file[k]) + '.dat', np.c_[cores,numbers,Theta_source[k,:],T[k,:], N[k,:],Delta_v[k,:],v_off[k,:],Theta_source_err[k,:],T_err[k,:], N_err[k,:],Delta_v_err[k,:],v_off_err[k,:]], delimiter=' ', fmt='%s')
		
		elif do_error_estimation == 'no':
			np.savetxt('RESULTS/TABLES/results_' + str(mol_names_file[k]) + '.dat', np.c_[cores,numbers,Theta_source[k,:],T[k,:], N[k,:],Delta_v[k,:],v_off[k,:]], delimiter=' ', fmt='%s')
				
	print('Extracted all best fit parameters for each molecules!')
	
	#create output table for each core
	for j in range(cores.size):
		
		if do_error_estimation == 'yes':
			np.savetxt('RESULTS/TABLES/results_' + str(cores[j]) + '_' + str(numbers[j]) + '.dat', np.c_[mol_names_file, Theta_source[:,j], T[:,j], N[:,j],Delta_v[:,j],v_off[:,j], Theta_source_err[:,j],T_err[:,j], N_err[:,j],Delta_v_err[:,j],v_off_err[:,j]], delimiter=' ', fmt='%s')
		
		elif do_error_estimation == 'no':
			np.savetxt('RESULTS/TABLES/results_' + str(cores[j]) + '_' + str(numbers[j]) + '.dat', np.c_[mol_names_file, Theta_source[:,j], T[:,j], N[:,j],Delta_v[:,j],v_off[:,j]], delimiter=' ', fmt='%s')
			
	print('Extracted all best fit parameters for each core!')		
	
			
def plot_results_cores():
	### plot results for temperature, column density, linewidth and velocity offset for each core
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#loop over all cores
	for j in range(cores.size):
		
		#load table with best fit parameters
		fit_results = np.loadtxt('RESULTS/TABLES/results_' + str(cores[j]) + '_' + str(numbers[j]) + '.dat', dtype='U', comments='#',ndmin=2)

		#mask out bad fits
		T = fit_results[:,2].astype(np.str)
		mask = np.where(T != 'nan')
		
		#load best fit parameters
		molecule = fit_results[:,0].astype(np.str)
		molecule = molecule[mask]
		T = fit_results[:,2].astype(np.float)
		T = T[mask]
		N = fit_results[:,3].astype(np.float)
		N = N[mask]
		delta_v = fit_results[:,4].astype(np.float)
		delta_v = delta_v[mask]
		v_off = fit_results[:,5].astype(np.float)
		v_off = v_off[mask]

		if do_error_estimation == 'yes':
			T_err = fit_results[:,7].astype(np.float)
			T_err = T_err[mask]
			N_err = fit_results[:,8].astype(np.float)
			N_err = N_err[mask]
			delta_v_err = fit_results[:,9].astype(np.float)
			delta_v_err = delta_v_err[mask]
			v_off_err = fit_results[:,10].astype(np.float)
			v_off_err = v_off_err[mask]
			
		#numbers for each molecule
		x = np.arange(1,T.size+1,1)
		
		#make 2x2 barchart plot
		fig = plt.figure(1)
		
		ax = plt.subplot(2, 2, 1)
		#annotate core and numbers
		plt.annotate(str(cores[j]) + ' ' + str(numbers[j]), xy=(0.1, 1.1), xycoords='axes fraction') 
		#temperature bar chart 
		if do_error_estimation == 'yes':
			plt.bar(x, T, width=0.8, yerr=T_err,align='center',color='tab:orange',edgecolor='black',linewidth=0.5, log=False)
		elif do_error_estimation == 'no':
			plt.bar(x, T, width=0.8,align='center',color='tab:orange',edgecolor='black',linewidth=0.5, log=False)
		#plot parameters	
		plt.ylabel('Temperature (K)')
		plt.xticks(x, molecule, rotation='vertical')
		ax.xaxis.set_ticklabels([])
		
		ax = plt.subplot(2, 2, 2)
		#column density bar chart
		if do_error_estimation == 'yes':
			plt.bar(x, N, width=0.8, yerr=N_err, align='center',color='tab:blue',edgecolor='black',linewidth=0.5, log=True)
		elif do_error_estimation == 'no':
			plt.bar(x, N, width=0.8, align='center',color='tab:blue',edgecolor='black',linewidth=0.5, log=True)
		#plot parameters
		plt.ylabel('Column Density (cm$^{-2}$)')
		plt.xticks(x, molecule, rotation='vertical')
		ax.xaxis.set_ticklabels([])
		
		ax = plt.subplot(2, 2, 3)
		#line width bar chart
		if do_error_estimation == 'yes':
			plt.bar(x, delta_v, width=0.8, yerr=delta_v_err,align='center',color='tab:green',edgecolor='black',linewidth=0.5, log=False)
		elif do_error_estimation == 'no':
			plt.bar(x, delta_v, width=0.8,align='center',color='tab:green',edgecolor='black',linewidth=0.5, log=False)
		#plot parameters
		plt.xticks(x, molecule, rotation='vertical')
		plt.ylabel('Line Width (km s$^{-1}$)')
		
		ax = plt.subplot(2, 2, 4)
		#velocity offset bar chart
		if do_error_estimation == 'yes':
			plt.bar(x, v_off, width=0.8, yerr=v_off_err,align='center',color='tab:purple',edgecolor='black',linewidth=0.5, log=False)
		elif do_error_estimation == 'no':
				plt.bar(x, v_off, width=0.8,align='center',color='tab:purple',edgecolor='black',linewidth=0.5, log=False)
		#plot parameters
		plt.ylabel('Velocity Offset (km s$^{-1}$)')
		plt.xticks(x, molecule, rotation='vertical')
		
		#adjust space between subplots
		plt.subplots_adjust(wspace=0.5, hspace=0.1)
		
		#save plot
		plt.savefig('RESULTS/BARCHARTS/Barchart_' + str(cores[j]) + '_' + str(numbers[j]) + '.pdf', format='pdf', bbox_inches='tight')
		plt.close()
		
	print('Created barchart plot for each core!')
	
	
def plot_results_molecule():
	### plot histogram for temperature, column density, linewidth and velocity offset for each molecule
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#loop over all molecules
	for k in range(mol_names_file.size):
		
		#load table with best fit parameters
		fit_results = np.loadtxt('RESULTS/TABLES/results_' + str(mol_names_file[k]) + '.dat', dtype='U', comments='#',ndmin=2)
		
		#mask out bad fits
		T = fit_results[:,3].astype(np.str)
		mask = np.where(T != 'nan')
		
		#load best fit parameters
		T = fit_results[:,3].astype(np.float)
		T = T[mask]
		N = fit_results[:,4].astype(np.float)
		N = N[mask]
		delta_v = fit_results[:,5].astype(np.float)
		delta_v = delta_v[mask]
		v_off = fit_results[:,6].astype(np.float)
		v_off = v_off[mask]
		
		
		if do_error_estimation == 'yes':
			
			#load column density error
			N_err = fit_results[:,9].astype(np.float)
			N_err = N_err[mask]
			
		
		#make 2x2 histogram
		fig = plt.figure(1) 
		
		ax = plt.subplot(2, 2, 1)
		#temperature histogram
		plt.annotate(str(mol_names_XCLASS[k]), xy=(0.1, 1.1), xycoords='axes fraction') 
		plt.hist(T, range=(np.float(temperature_low),np.float(temperature_upp)), log=False, color='tab:orange', label=None)
		plt.xlabel('Temperature (K)')
		
		ax = plt.subplot(2, 2, 2)
		#column density histogram
		plt.hist(np.log10(N), range=(np.log10(np.float(columndensity_low)),np.log10(np.float(columndensity_upp))), log=False, color='tab:blue', label=None)
		plt.xlabel('log(Column Density) (log(cm$^{-2}$))')
		
		ax = plt.subplot(2, 2, 3)
		#line width histogram
		plt.hist(delta_v, range=(np.float(linewidth_low),np.float(linewidth_upp)), log=False, color='tab:green', label=None)
		plt.xlabel('Line Width (km s$^{-1}$)')
		
		ax = plt.subplot(2, 2, 4)
		#velocity offset histogram
		plt.hist(v_off, range=(np.float(velocityoffset_low), np.float(velocityoffset_upp)), log=False, color='tab:purple', label=None)
		plt.xlabel('Velocity Offset (km s$^{-1}$)')
		
		#adjust space between subplots
		plt.subplots_adjust(wspace=0.3, hspace=0.5)
		
		#save plot
		plt.savefig('RESULTS/HISTOGRAMS/Histogram_' + str(mol_names_file[k]) + '.pdf', format='pdf', bbox_inches='tight')
		plt.close()

	print('Created histogram plot for each molecule!')
	
	
def create_plots():
	###make bar chart and histogram plots
	
	#plot barchart for each core
	plot_results_cores()
	
	#plot histogram for each molecule
	plot_results_molecule()

		
		
def run_XCLASS_fit_all_fixed(working_directory):
	###create molfits and observation xml files to compute total XCLASS fit spectrum
	
	create_XCLASS_molfits_file(method='all')
	create_XCLASS_obsxml_file(working_directory, method='all')

	#run XCLASS with fixed parameters
	os.system('casa -c Functions/XCLASS_fit_RESULTS_all.py')
	rm_casa_files()
	
	
def plot_fit():
	###plot observed spectrum + fit
	
	#plotting parameters
	plt.rcParams.update(params)
	
	#load noise table
	std_line = np.loadtxt('RESULTS/TABLES/noise.dat', delimiter=' ', usecols=2, ndmin=2)	
		
	#loop over all cores
	for j in range(cores.size):

		#load observed spectrum
		data = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_vLSRcorr.dat')
		X = data[:,0] / 1000.0
		Y = data[:,1]		
		
		#load model spectrum
		fit = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_compl.out.dat')
		X_fit = fit[:,0] / 1000.0
		Y_fit = fit[:,1]
		
		#plot spectrum & fit
		plt.ioff()
		plt.figure(1) 
		ax = plt.subplot(111)
		#plot spectrum
		plt.step(X,Y,'k', label='Observed Flux Density',lw=0.5, where='mid')
		#plot fit
		plt.step(X_fit,Y_fit,'tab:red', label='XCLASS Fit',lw=0.5,alpha=0.8, where='mid')
		#plot parameters
		plt.xlim(float(min(X))-0.02, max(X)+0.02)
		xmajor = MultipleLocator(0.5)
		ax.xaxis.set_major_locator(xmajor)
		xminor = MultipleLocator(0.1)
		ax.xaxis.set_minor_locator(xminor)
		plt.ylabel('Brightness Temperature (K)')
		plt.xlabel('Frequency (GHz)')
		plt.ticklabel_format(useOffset=False)
		plt.legend(bbox_to_anchor=(0.0, 1.4, 0.0, 0.0),loc='upper left')

		#label fitted lines
		for i in range(mol_lines_name.size):
				plt.annotate(mol_lines_name[i], xy=(mol_lines_freq[i]/1000.0, 0.0), xytext=(mol_lines_freq[i]/1000.0, np.amax(Y)*1.25), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.1, color='tab:blue',linewidth=0.5), fontsize=5, rotation = 90, alpha=1.0, color='tab:blue')
	
		#save plot
		plt.savefig('RESULTS/SPECTRA/XCLASSFit_' + str(cores[j]) + '_' + str(numbers[j]) + '_single.pdf', format='pdf', bbox_inches='tight')
		plt.close() 
		
	print('Created plot for each observed and fitted spectrum!')
		
		
		
def setup_files(working_directory):
	
	determine_noise()

	extract_spectrum_init()
	
	setup_XCLASS_files(working_directory)
	
	
def run_fit():
	### perform XCLASS fit in casa
	
	# use specific line to determine precise systemic velocity
	if vlsr_corr== 'yes':
		
		#run inital XCLASS Fit on specific line to determine the vLSR of each core
		os.system('casa -c Functions/XCLASS_VLSR_determination.py')
		rm_casa_files()
		
		#get velocity offset from specific line fit
		v_off = get_velocity_offset()
	
	# use general region systemic velocity
	else:
		v_off = np.zeros(cores.size)
		
	#extract spectra & correct for systemic velocity
	extract_spectrum(v_off)
		
	#run XCLASS Fit
	os.system('casa -c Functions/XCLASS_fit.py')
	rm_casa_files()
	
	
def analysis(working_directory):
	
	extract_results()
	
	create_plots()
	
	run_XCLASS_fit_all_fixed(working_directory)
	
	plot_fit()
	
	
	
	
	
	
	