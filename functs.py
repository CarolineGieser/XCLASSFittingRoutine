import os
import fnmatch 
import numpy as np
from astropy import units as u
from astropy.io import fits
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from matplotlib.ticker import MultipleLocator
from numba import jit

###---plotting parameters---###
params = {'font.family' : 'serif',
			 'font.size' : 7,
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
#galactic center coordinates and distance
GC_coords = SkyCoord('17 45 40.04 -29 00 28.1', unit=(u.hourangle, u.deg),distance=8.122*u.kpc)


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
	###WARNING! ALL results will be lost!
	
	#remove directories in which results are stored
	os.system('rm -r FITS')
	os.system('rm -r PLOTS')
	os.system('rm -r RESULTS')
	
	print('Removed previous results!')
	
	
def setup_directory(delete_previous_results=False):
	### setup all directories in which XCLASS results are stored
	
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
	make_directory(dirName='PLOTS/BADFIT')
	#good XCLASS fits
	make_directory(dirName='PLOTS/GOODFIT')
	#vlsr detemination fit
	make_directory(dirName='PLOTS/VLSR')
	#continuum plots
	make_directory(dirName='PLOTS/CONTINUUM')
	#abundance correlation plots
	make_directory(dirName='PLOTS/ABUNDANCES')
	#spectrum plots
	make_directory(dirName='PLOTS/SPECTRA')
	#results directory
	make_directory(dirName='RESULTS')
	#MUSCLE input tables
	make_directory(dirName='RESULTS/MODELINPUT')
	#XCLASS spectra
	make_directory(dirName='RESULTS/XCLASS_SPECTRA')
	#barcharts for each core
	make_directory(dirName='RESULTS/BARCHART')
	#histograms for each molecule
	make_directory(dirName='RESULTS/HISTOGRAM')
	#tables (e.g. noise,...)
	make_directory(dirName='RESULTS/TABLES')
	
	print('Working directory: ' + working_directory)
	
	return working_directory

	
def create_input_table(data_directory,do_error_estimation,freq_low,freq_upp):
	###create input.dat file with parameters
	
	#open and write inputfile
	file = open('input.dat','w') 
	file.write(
	"""data_directory """ + np.str(data_directory) + """\n"""
	"""do_error_estimation """ + np.str(do_error_estimation) + """\n"""
	"""freq_low """ + np.str(freq_low) + """\n"""
	"""freq_upp """ + np.str(freq_upp) + """\n"""
	)
	file.close()	
		
	
def load_regions_table():
	#### input table of regions
	
	#load table
	regions_tab=np.loadtxt('regions.dat', dtype='U', comments='#')
	
	#CORE region name
	regions = regions_tab[:,0].astype(np.str) 
	#CORE region name for plotting
	regions_plot = regions_tab[:,1].astype(np.str)
	#distance (kpc) 
	distances = regions_tab[:,2].astype(np.float) 
	#fits filenames of spectral line datacubes
	filenames = regions_tab[:,3].astype(np.str) 
	#fits filenames of continuum data
	filenames_continuum = regions_tab[:,4].astype(np.str) 
	#luminosity (10^4 L_sun)
	region_luminosity = regions_tab[:,5].astype(np.float)
	#mass (M_sun) 
	region_mass = regions_tab[:,6].astype(np.float) 
	
	return regions, regions_plot, distances, filenames, filenames_continuum, region_luminosity, region_mass


def load_cores_table():
	#### input table of cores
	
	#load table
	cores_tab=np.loadtxt('cores.dat', dtype='U', comments='#')
	
	#region name
	cores = cores_tab[:,0].astype(np.str)
	#core number
	number = cores_tab[:,1].astype(np.int)
	#position in RA (pixel)
	x_pix = cores_tab[:,2].astype(np.int)
	#position in DEC (pixel)
	y_pix = cores_tab[:,3].astype(np.int)
	#'C' (core) or 'E' (envelope)
	core_label = cores_tab[:,4].astype(np.str)
		
	return cores, number, x_pix, y_pix, core_label
	
	
def load_molecules_table():
	### input table of molecules
	
	#load table
	mol_data=np.loadtxt('molecules.dat', dtype='U', comments='%')
	
	#XCLASS molecule label
	mol_name=mol_data[:,0]
	#filename molecule label
	mol_name_file=mol_data[:,1] 
	#MUSCLE molecule label
	mol_name_MUSCLE=mol_data[:,2]
	#plot molecule label 
	mol_name_plot=mol_data[:,3] 
	
	return mol_name, mol_name_file, mol_name_MUSCLE, mol_name_plot
	
	
def load_molecule_lines_table():
	### input table of molecule lines to fit
	
	#load table
	mol_lines_tab=np.loadtxt('molecule_lines.dat', dtype='U', comments='%')
	
	#XCLASS molecule label
	mol_line_name=mol_lines_tab[:,0]
	#rest frequency of transition
	mol_line_freq=mol_lines_tab[:,1].astype(np.float)
	
	return mol_line_name, mol_line_freq
	

regions, regions_plot, distances, filenames, filenames_continuum, region_luminosity, region_mass = load_regions_table()
cores, number, x_pix, y_pix, core_label = load_cores_table()
mol_name, mol_name_file, mol_name_MUSCLE, mol_name_plot = load_molecules_table()
mol_lines_name, mol_lines_freq = load_molecule_lines_table()

	

def check_error_estimation(do_error_estimation):
	###check if error estimation should be performed
	
	#create 3 tags for normal spectrum, and lower, and upper error estimation
	if do_error_estimation == 'yes':
		
		tag = np.array(['','_lowErr','_uppErr'])
		
	#create only 1 tag for normal spectrum
	elif do_error_estimation == 'no':
		
		tag = np.array([''])
		
	#raise an error if do_error_estimation is neither 'yes' nor 'no'
	else:
		print('Only yes and no are allowed for do_error_estimation parameter in input.dat!')
		
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
	
	
@jit
def determine_noise(data_directory,freq_low,freq_upp):
	### compute noise in each spectrum in a line-free channel range
	
	#array in which noise values are stored
	std_line = np.zeros(cores.size)
	
	#loop over all cores
	for j in range(cores.size):
		
		# get fits filename of datacube
		mask = np.where(regions == cores[j])
		filename = filenames[mask]
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
		print('Determined Noise in ' + str(cores[j]) + ' ' + str(number[j]) + ' Spectrum: ' + str(std_line[j])+ ' K')
		
	#save noise values in file
	np.savetxt('RESULTS/TABLES/Noise.dat', np.c_[cores,number,std_line], delimiter=' ', fmt='%s')
	
	return std_line
	

@jit	
def extract_spectrum_init(data_directory):
	### extract and save spectrum of each core which will be fitted with XCLASS (only specific line)
	
	#loop over all cores
	for j in range(cores.size):
		
		# get fits filename of datacube
		mask = np.where(regions == cores[j])
		filename = filenames[mask]
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
		np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '.dat', np.c_[X.value,Y.value])
		
		#print results
		print('Extracted ' + str(cores[j]) + ' ' + str(number[j]) + ' Spectrum!')
		
		
def create_XCLASS_molfits_file(vlsr_molec,method):
	### create input molfits for XCLASS fitting
	
	#molfits file for initial C18O fit
	if method == 'vLSR':
		
		#open and write molfits file
		file = open('FITS/molecules_vLSR.molfit','w') 
		file.write(
vlsr_molec + """   1 \n"""
"""y   0.1   1.0   0.4   y   5.0   300.0   100.0   y   1.0E+12   1.0E+19   1.0e+15   y   0.5   15.0   5.0   y   -15.0   15.0   0.0   c"""
		)
		file.close()

	#molfits file for molecule fits
	elif method == 'single':
		
		#loop over all molecules
		for k in range(mol_name.size): 
			
			#open and write molfits file
			file = open('FITS/molecules_' + str(mol_name_file[k]) + '.molfit','w') 
			file.write(
mol_name[k] +"""   1 \n"""
"""y   0.1   1.0   0.4   y   5.0   300.0   100.0   y   1.0E+12   1.0E+19   1.0e+15   y   0.5   10.0   5.0   y   -6.0   6.0   0.0   c"""
			)
			file.close()
					
	#molfits file for final fit at fixed parameters		
	elif method == 'all':
		
		#loop over all cores
		for j in range(cores.size):
			
			#open and write molfits file
			file = open('FITS/molecules_' + str(cores[j]) + '_' + str(number[j]) + '_compl.molfit','w') 
			
			#loop over all fitted molecules
			for k in range(mol_name.size):
				
				#load fit results
				fitdata = np.loadtxt('RESULTS/TABLES/Results_' + str(mol_name_file[k]) +'.dat', dtype='U', comments='#')
				
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
mol_name[k] +"""   1 \n"""
"""n   0.1   1.0   """+ str(Theta_source[j]) +"""   n   5.0   300.0   """+ str(T[j]) +"""   n   1.0E+10   1.0E+21   """+ str(N[j]) +"""   n   0.5   20.0   """+ str(Delta_v[j]) +"""   n   -6.0   6.0    """+ str(v_off[j]) +"""   c\n"""
					)
			file.close()
			
	# raise error if "method" keyword is wrong	
	else:
		
		raise Exception('Only method = vLSR, single or all possible!')
		
	print('Created molfits files!') 
	
	
def create_XCLASS_obsxml_file(data_directory,working_directory, do_error_estimation,vlsr_freq,method):
	### create input observations xml files for XCLASS fitting
	
	#observational xml file for vlsr detemination fit
	if method == 'vLSR':
		
		#loop over all cores
		for j in range(cores.size):
			
			# get fits filename of datacube
			mask = np.where(regions == cores[j])
			filename = filenames[mask]
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
			file = open('FITS/observation_' + str(cores[j]) + '_' + str(number[j]) + '_vLSR.xml','w') 
			file.write(
"""<?xml version="1.0" encoding="UTF-8"?>
<ExpFiles>
 <NumberExpFiles>1</NumberExpFiles>
 <file>
     <FileNamesExpFiles>""" + str(working_directory) + """FITS/spectrum_""" + str(cores[j]) + """_""" + str(number[j]) + """.dat</FileNamesExpFiles>
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
     <IsoTableFileName>""" + str(working_directory) + """FITS/isotopologues_""" + str(cores[j]) + """.dat</IsoTableFileName>
</ExpFiles>"""
			)
			file.close() 
	
	#observation xml file for molecule fits
	elif method == 'single':
		
		# check if error estimation or not
		tag = check_error_estimation(do_error_estimation)
		
		#loop over all molecules
		for k in range(mol_name.size):
			
			#load line frequencies for molecule
			mask = np.where(mol_lines_name == mol_name[k])
			mol_lines_name_mask = mol_lines_name[mask]
			mol_lines_freq_mask = mol_lines_freq[mask]
			
			#loop over all tags (for error estimation)
			for z in range(tag.size):
				
				#loop over all cores
				for j in range(cores.size):
					
					# get fits filename of datacube
					mask = np.where(regions == cores[j])
					filename = filenames[mask]
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
					file = open('FITS/observation_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + str(tag[z]) + '.xml','w') 
					file.write(
"""<?xml version="1.0" encoding="UTF-8"?>
<ExpFiles>
 <NumberExpFiles>1</NumberExpFiles>
 <file>
     <FileNamesExpFiles>""" + str(working_directory) +"""FITS/spectrum_""" + str(cores[j]) + """_""" + str(number[j]) + str(tag[z]) + """_vLSRcorr.dat</FileNamesExpFiles>
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
     <IsoTableFileName>""" + str(working_directory) + """FITS/isotopologues_""" + str(cores[j]) + """.dat</IsoTableFileName>
</ExpFiles>"""
					)
					file.close() 
					
	#observation xml file for final fit at fixed parameters					
	elif method == 'all':
		
		#loop over all cores
		for j in range(cores.size):
			
			#load spectrum to get upper and lower frequency bounds
			data_tab = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_vLSRcorr.dat')
			#get spectral axis
			X = data_tab[:,0]
			
			# get fits filename of datacube
			mask = np.where(regions == cores[j])
			filename = filenames[mask]
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
			file = open('FITS/observation_' + str(cores[j]) + '_' + str(number[j]) + '_all.xml','w') 
			file.write(
"""<?xml version="1.0" encoding="UTF-8"?>
<ExpFiles>
 <NumberExpFiles>1</NumberExpFiles>
 <file>
     <FileNamesExpFiles>""" + str(working_directory) +"""FITS/spectrum_""" + str(cores[j]) + """_""" + str(number[j]) + """_vLSRcorr.dat</FileNamesExpFiles>
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
     <IsoTableFileName>""" + str(working_directory) + """FITS/isotopologues_""" + str(cores[j]) + """.dat</IsoTableFileName>
</ExpFiles>"""
			)
			file.close() 
			
	# raise error if "method" keyword is wrong			
	else:
		
		raise Exception('Only method = vLSR, single or all possible!')
		
	print('Created observation xml files!') 

		
def carbon_12_13_ratio(d):
	####12C/13C Iso-ratio from Wilson & Rood (1994)
	
	return round(7.5 * d + 7.6)
	
def nitrogen_14_15_ratio(d):
	####14N/15N Iso-ratio from Wilson & Rood (1994)
	
	return round(19.0 * d + 288.6)
	
def oxygen_16_18_ratio(d):
	####16O/18O Iso-ratio from Wilson & Rood (1994)
	
	return round(58.8 * d + 37.1)
	
def sulfur_32_34_ratio():
	####32S/23S Iso-ratio from Wilson & Rood (1994)
	
	return 22
			
			
def create_XCLASS_isoratio_file(data_directory):
	### create input iso-ratio files for XCLASS fitting
	
	#empty galactic distance array
	d_to_GC_arr = np.zeros(regions.size)
	
	#loop over all regions
	for s in range(regions.size):
		
		#open fits file
		hdu = fits.open(data_directory + filenames[s])[0]

		#get coordinates (Right Ascension and Declination) of phase center
		RA = hdu.header['CRVAL1']
		DEC = hdu.header['CRVAL2']
		source_coords = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, distance=distances[s]*u.kpc, frame='icrs')
		
		#compute distance from region to Galactic Center
		d_to_GC = source_coords.separation_3d(GC_coords).value

		#open and write iso-ratio file
		file = open('FITS/isotopologues_' + str(regions[s]) + '.dat','w')
		file.write(
"""%Isotopologue MainIsotpologogue Ratio
H2C-13-O;v=0;	H2CO;v=0;	""" + str(carbon_12_13_ratio(d_to_GC)) + """ 
OC-13-S;v=0;	OCS;v=0;	""" + str(carbon_12_13_ratio(d_to_GC)) + """ 
CH3C-13-N;v=0;	CH3CN;v=0;	""" + str(carbon_12_13_ratio(d_to_GC)) + """ 
HCCC-13-N;v=0;	HCCCN;v=0;	""" + str(carbon_12_13_ratio(d_to_GC)) + """
S-34-O2;v=0;	SO2;v=0;	""" + str(sulfur_32_34_ratio())
		)
		file.close()
		
		#add to galactic distance array
		d_to_GC_arr[s] = d_to_GC
		
	#save galactic distances in table
	np.savetxt('RESULTS/TABLES/GalacticDistances.dat', np.c_[regions,d_to_GC_arr], delimiter=' ', fmt='%s')
	
	print('Created iso-ratio files!') 
	
	
def setup_XCLASS_files(data_directory, working_directory,do_error_estimation,vlsr_molec,vlsr_freq):
	### create isoratio, molfits and xml files
	
	#isotopologue ratio files
	create_XCLASS_isoratio_file(data_directory)
	
	#molfits files for vlsr determination
	create_XCLASS_molfits_file(vlsr_molec,method='vLSR')
	#xml files for vlsr determination
	create_XCLASS_obsxml_file(data_directory,working_directory,do_error_estimation,vlsr_freq, method='vLSR')
	
	#molfits files for molecule fits
	create_XCLASS_molfits_file(vlsr_molec,method='single')
	#xml files for molecule fits
	create_XCLASS_obsxml_file(data_directory,working_directory,do_error_estimation,vlsr_freq,method='single')
	

def get_velocity_offset():
	### extract bestfit systemic velocity and plot fitted vlsr determination line
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#load noise table
	noise = np.loadtxt('RESULTS/TABLES/Noise.dat',usecols=2)
	
	#empty array in which v_off values are stored
	VelocityOffset = np.zeros(cores.size)
	
	#loop over all cores
	for j in range(cores.size):
		
		#load spectrum
		data = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '.dat')
		X = data[:,0] 
		Y = data[:,1]
		
		#load fit		
		fit = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_vLSR.out.dat')
		X_fit = fit[:,0] 
		Y_fit = fit[:,1]
		
		#plot spectrum and fit
		plt.ioff()
		plt.figure() 
		#create subplot
		ax = plt.subplot(1, 1, 1)
		plt.step(X,Y,'k', label='Data', where='mid')
		plt.step(X_fit,Y_fit,'tab:red', label='Fit',linestyle='--', where='mid')
		plt.axhline(y=3.0*noise[j], xmin=0, xmax=1,color='tab:green', linestyle=':', label=r'3$\sigma$')
		plt.xlim(np.amin(X_fit)-0.001,np.amax(X_fit)+0.001)
		plt.xlabel('Frequency (GHz)')
		plt.ylabel('Brightness Temperature (K)')
		plt.ticklabel_format(useOffset=False)
		plt.annotate(str(cores[j]) + ' ' + str(number[j]), xy=(0.1, 0.9), xycoords='axes fraction')  
		plt.legend(loc='upper right')
		#apply ylimits depending on highest brightness temperature in fitted range
		if np.amax(Y_fit) > 5.0*noise[j]:
			plt.ylim(-5.0*noise[j],np.amax(Y_fit*1.5))
		else:
			plt.ylim(-5.0*noise[j],10.0*noise[j])
		plt.savefig('PLOTS/VLSR/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_vLSR.png', format='png', bbox_inches='tight')
		plt.close()
		
		#extract v_off
		#load fit results table
		fit_results = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(number[j]) + '_vLSR.molfit', dtype='U', skiprows=1) 
		#get best fit velocity offset
		VelocityOffset[j] = fit_results[19] #km/s
	
	#save velocity offset table
	np.savetxt('RESULTS/TABLES/VelocityOffset.dat', np.c_[cores,number,VelocityOffset], delimiter=' ', fmt='%s')
	
	print('Extracted all velocity offsets!')
	
	return VelocityOffset
	
		
def extract_spectrum(data_directory,v_off,do_error_estimation):
	### extract and save spectra of each core which will be fitted with XCLASS
	
	#loop over all cores
	for j in range(cores.size):
		
		# get fits filename of datacube
		mask = np.where(regions == cores[j])
		filename = filenames[mask]
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
		np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_vLSRcorr.dat', np.c_[X.value,Y.value])
		
		#get tags if or if not error estimation should be performed
		tag = check_error_estimation(do_error_estimation)
		
		if do_error_estimation =='yes':
			
			#scale spectra by +-20 % to estimate how fit parameters vary
			np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + tag[1] + '_vLSRcorr.dat', np.c_[X.value,Y.value*0.8])
			np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + tag[2] +'_vLSRcorr.dat', np.c_[X.value,Y.value*1.2])
		
		print('Extracted ' + str(cores[j]) + ' ' + str(number[j]) + ' Spectrum (spectral axis corrected for systemic velocity)!')
	
	
def rm_casa_files():
	### remove unnecessary casa files
	
	os.system('rm casa*.log')
	os.system('rm *.last')
	
	print('Removed casa log files!')
	
	
def run_XCLASS_fit(data_directory,do_error_estimation,vlsr_corr,vlsr_molec,vlsr_freq):
	### perform XCLASS fit in casa
	
	# use specific line to determine precise systemic velocity
	if vlsr_corr==True:
		
		#run inital XCLASS Fit on specific line to determine the vLSR of each core
		os.system('casa -c XCLASS_VLSR_determination.py')
		rm_casa_files()
		
		#get velocity offset from specific line fit
		v_off = get_velocity_offset()
	
	# use general region systemic velocity
	else:
		v_off = np.zeros(cores.size)
		
	#extract spectra & correct for systemic velocity
	extract_spectrum(data_directory,v_off,do_error_estimation)
		
	#run XCLASS Fit
	os.system('casa -c XCLASS_fit.py')
	rm_casa_files()
	

def extract_results(do_error_estimation,plotting=True,flagging=True):
	###extract best fit parameters from molfit files
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#load noise table
	std_line = np.loadtxt('RESULTS/TABLES/Noise.dat', delimiter=' ', usecols=2)	
	
	#create empty best fit parameter arrays
	#source size
	Theta_source = np.zeros(shape=(mol_name_file.size,cores.size))
	#temperature
	T = np.zeros(shape=(mol_name_file.size,cores.size))
	#column density
	N = np.zeros(shape=(mol_name_file.size,cores.size))
	#line width
	Delta_v = np.zeros(shape=(mol_name_file.size,cores.size))
	#velocity offset
	v_off = np.zeros(shape=(mol_name_file.size,cores.size))
	
	#create empty error parameter array
	if do_error_estimation == 'yes':
		
		#source size err
		Theta_source_err = np.zeros(shape=(mol_name_file.size,cores.size))
		#temperature
		T_err = np.zeros(shape=(mol_name_file.size,cores.size))
		#column density
		N_err = np.zeros(shape=(mol_name_file.size,cores.size))
		#line width
		Delta_v_err = np.zeros(shape=(mol_name_file.size,cores.size))
		#velocity offset
		v_off_err = np.zeros(shape=(mol_name_file.size,cores.size))

	#loop over all fitted molecules
	for k in range(mol_name_file.size):
		
		#loop over all cores
		for j in range(cores.size):
			
			#load data
			data = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_vLSRcorr.dat')
			#frequency
			X = data[:,0]/1000.0 #GHz
			#brightness temperature
			Y = data[:,1] #K
			
			#load fit
			fit = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '.out.dat')
			#fit frequency			
			X_fit = fit[:,0]/1000.0 #GHz
			#fit brightness temperature
			Y_fit = fit[:,1] #K
			
			#load scaled spectra if error estimation is performed
			if do_error_estimation == 'yes':
				
				#load data
				fit_lowErr = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '_lowErr.out.dat')
				#fit frequency
				X_fit_lowErr = fit_lowErr[:,0]/1000.0 #GHz
				#fit brightness temperature
				Y_fit_lowErr = fit_lowErr[:,1] #K
				
				#load data
				fit_uppErr = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '_uppErr.out.dat')
				#fit frequency
				X_fit_uppErr = fit_uppErr[:,0]/1000.0 #GHz
				#fit brightness temperature
				Y_fit_uppErr = fit_uppErr[:,1] #K
				
			#load line frequencies for molecule
			mask = np.where(mol_lines_name == mol_name[k])
			mol_lines_name_mask = mol_lines_name[mask]
			mol_lines_freq_mask = mol_lines_freq[mask]
			
			
			if mol_lines_name_mask.size % 2 == 1:
				n_row = np.int(mol_lines_name_mask.size/2 + 0.5)
			if mol_lines_name_mask.size % 2 == 0:
				n_row = np.int(mol_lines_name_mask.size/2)
			if mol_lines_name_mask.size == 1:
				n_row = 1
				
			if plotting == True:
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
						plt.annotate(str(cores[j]) + ' ' + str(number[j]) + ' ' + str(mol_name_file[k]), xy=(0.1, 1.1), xycoords='axes fraction')  
						plt.legend(loc='upper right')
					
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
				if plotting == True:	
				
					plt.savefig('PLOTS/GOODFIT/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '.png', format='png', bbox_inches='tight')
					plt.close()
			
				#load best fit molfits file
				fit_table = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '.molfit', dtype='U', skiprows=1) 
				
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
					fit_table_upp = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '_uppErr.molfit', dtype='U', skiprows=1) 
				
					#load best fit molfits file (lower error)
					fit_table_low = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '_lowErr.molfit', dtype='U', skiprows=1) 
					
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
				
				if plotting == True:
					#save plot
					plt.ylim(-5.0*std_line[j],5*std_line[j])
					plt.savefig('PLOTS/BADFIT/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '.png', format='png', bbox_inches='tight')
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
	#
	#if flagging == True:
	#	flag_tab = np.loadtxt('flag.dat',dtype='U',comments='#')
	#	core_flag = flag_tab[:,0].astype(np.str)
	#	number_flag = flag_tab[:,1].astype(np.int)
	#	molecule_flag = flag_tab[:,2].astype(np.str)
		
	#	for i in range(core_flag.size):		
				
	#		for k in range(mol_name_file.size):
			
	#			for j in range(cores.size):
					
	#				if core_flag[i] == cores[j] and number_flag[i] == number[j] and molecule_flag[i] == mol_name_file[k]:
	#					Theta_source[k,j] = np.nan					
	#					T[k,j] = np.nan
	#					N[k,j] = np.nan
	#					Delta_v[k,j] = np.nan
	#					v_off[k,j] = np.nan
						
	#					if do_error_estimation == 'yes':
	#						Theta_source_err[k,j] = np.nan	
	#						T_err[k,j] = np.nan
	#						N_err[k,j] = np.nan
	#						Delta_v_err[k,j] = np.nan
	#						v_off_err[k,j] = np.nan
						
					
	#create output table for each molecule	
	for k in range(mol_name_file.size):
		if do_error_estimation == 'yes':
			np.savetxt('RESULTS/TABLES/Results_' + str(mol_name_file[k]) + '.dat', np.c_[cores,number,Theta_source[k,:],T[k,:], N[k,:],Delta_v[k,:],v_off[k,:],Theta_source_err[k,:],T_err[k,:], N_err[k,:],Delta_v_err[k,:],v_off_err[k,:]], delimiter=' ', fmt='%s')
		elif do_error_estimation == 'no':
			np.savetxt('RESULTS/TABLES/Results_' + str(mol_name_file[k]) + '.dat', np.c_[cores,number,Theta_source[k,:],T[k,:], N[k,:],Delta_v[k,:],v_off[k,:]], delimiter=' ', fmt='%s')
				
	print('Extracted all best fit parameters for each molecules!')
	
	#create output table for each core
	for j in range(cores.size):
		if do_error_estimation == 'yes':
			np.savetxt('RESULTS/TABLES/Results_' + str(cores[j]) + '_' + str(number[j]) + '.dat', np.c_[mol_name_file, Theta_source[:,j], T[:,j], N[:,j],Delta_v[:,j],v_off[:,j], Theta_source_err[:,j],T_err[:,j], N_err[:,j],Delta_v_err[:,j],v_off_err[:,j]], delimiter=' ', fmt='%s')
		elif do_error_estimation == 'no':
			np.savetxt('RESULTS/TABLES/Results_' + str(cores[j]) + '_' + str(number[j]) + '.dat', np.c_[mol_name_file, Theta_source[:,j], T[:,j], N[:,j],Delta_v[:,j],v_off[:,j]], delimiter=' ', fmt='%s')
			
	print('Extracted all best fit parameters for each core!')		
	
			
def plot_results_cores(do_error_estimation):
	### plot results for temperature, column density, linewidth and velocity offset for each core
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#loop over all cores
	for j in range(cores.size):
		
		#load table with best fit parameters
		fit_results = np.loadtxt('RESULTS/TABLES/Results_' + str(cores[j]) + '_' + str(number[j]) + '.dat', dtype='U', comments='#')
		
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
			T_Err = fit_results[:,7].astype(np.float)
			T_Err = T_Err[mask]
			N_Err = fit_results[:,8].astype(np.float)
			N_Err = N_Err[mask]
			delta_v_Err = fit_results[:,9].astype(np.float)
			delta_v_Err = delta_v_Err[mask]
			v_off_Err = fit_results[:,10].astype(np.float)
			v_off_Err = v_off_Err[mask]
			
		#number for each molecule
		x = np.arange(1,T.size+1,1)
		
		#make 2x2 barchart plot
		fig = plt.figure(1)
		ax = plt.subplot(2, 2, 1)
		
		plt.annotate(str(cores[j]) + ' ' + str(number[j]), xy=(0.1, 1.1), xycoords='axes fraction') 
		
		if do_error_estimation == 'yes':
			plt.bar(x, T, width=0.8, yerr=T_Err,align='center',color='tab:orange',edgecolor='black',linewidth=0.5, log=False)
		elif do_error_estimation == 'no':
			plt.bar(x, T, width=0.8,align='center',color='tab:orange',edgecolor='black',linewidth=0.5, log=False)
		plt.ylabel('Temperature (K)')
		plt.xticks(x, molecule, rotation='vertical')
		ax.xaxis.set_ticklabels([])
		ax = plt.subplot(2, 2, 2)
		if do_error_estimation == 'yes':
			plt.bar(x, N, width=0.8, yerr=N_Err, align='center',color='tab:blue',edgecolor='black',linewidth=0.5, log=True)
		elif do_error_estimation == 'no':
			plt.bar(x, N, width=0.8, align='center',color='tab:blue',edgecolor='black',linewidth=0.5, log=True)
		plt.ylabel('Column Density (cm$^{-2}$)')
		plt.xticks(x, molecule, rotation='vertical')
		ax.xaxis.set_ticklabels([])
		ax = plt.subplot(2, 2, 3)
		if do_error_estimation == 'yes':
			plt.bar(x, delta_v, width=0.8, yerr=delta_v_Err,align='center',color='tab:green',edgecolor='black',linewidth=0.5, log=False)
		elif do_error_estimation == 'no':
			plt.bar(x, delta_v, width=0.8,align='center',color='tab:green',edgecolor='black',linewidth=0.5, log=False)
		plt.xticks(x, molecule, rotation='vertical')
		plt.ylabel('Line Width (km s$^{-1}$)')
		ax = plt.subplot(2, 2, 4)
		if do_error_estimation == 'yes':
			plt.bar(x, v_off, width=0.8, yerr=v_off_Err,align='center',color='tab:purple',edgecolor='black',linewidth=0.5, log=False)
		elif do_error_estimation == 'no':
				plt.bar(x, v_off, width=0.8,align='center',color='tab:purple',edgecolor='black',linewidth=0.5, log=False)
		plt.ylabel('Velocity Offset (km s$^{-1}$)')
		
		plt.subplots_adjust(wspace=0.3, hspace=0.1)
		plt.xticks(x, molecule, rotation='vertical')
		plt.savefig('RESULTS/BARCHART/Barchart_' + str(cores[j]) + '_' + str(number[j]) + '.pdf', format='pdf', bbox_inches='tight')
		plt.close()
		
	print('Created barchart plot for each core!')
	
	
def plot_results_molecule():
	### plot histogram for temperature, column density, linewidth and velocity offset for each molecule
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#loop over all molecules
	for k in range(mol_name_file.size):
		
		#load table with best fit parameters
		fit_results = np.loadtxt('RESULTS/TABLES/Results_' + str(mol_name_file[k]) + '.dat', dtype='U', comments='#')
		
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
		
		#make 2x2 histogram
		fig = plt.figure(1) 
		ax = plt.subplot(2, 2, 1)
		plt.annotate(str(mol_name_file[k]), xy=(0.1, 0.9), xycoords='axes fraction') 
		plt.hist(T, range=(0.0,300.0), log=False, color='tab:orange', label=None)
		plt.xlabel('Temperature (K)')
		ax = plt.subplot(2, 2, 2)
		plt.hist(np.log10(N), range=(12.0,19.0), log=False, color='tab:blue', label=None)
		plt.xlabel('LOG(Column Density) (LOG(cm$^{-2}$))')
		ax = plt.subplot(2, 2, 3)
		plt.hist(delta_v, range=(0.5,15), log=False, color='tab:green', label=None)
		plt.xlabel('Line Width (km s$^{-1}$)')
		ax = plt.subplot(2, 2, 4)
		plt.hist(v_off, range=(-6,6), log=False, color='tab:purple', label=None)
		plt.xlabel('Velocity Offset (km s$^{-1}$)')
		
		plt.subplots_adjust(wspace=0.3, hspace=0.3)
		
		plt.savefig('RESULTS/HISTOGRAM/Histogram_' + str(mol_name_file[k]) + '.pdf', format='pdf', bbox_inches='tight')
		plt.close()

	print('Created histogram plot for each molecule!')
	
def create_plots(do_error_estimation):
	###make plots
	
	#plot barchart for each core
	plot_results_cores(do_error_estimation)
	
	#plot histogram for each molecule
	plot_results_molecule()


def plot_fit_residuals_optical_depth():
	###plot observed spectrum + fit, residuals and optical depth
	
	#plotting parameters
	plt.rcParams.update(params)
	
	#load noise table
	std_line = np.loadtxt('RESULTS/TABLES/Noise.dat', delimiter=' ', usecols=2)	
		
	#loop over all cores
	for j in range(cores.size):
		
		#load table with line properties	
		trans_tab =np.loadtxt('FITS/' + str(cores[j]) + '_' + str(number[j]) + '_transition_energies.dat',dtype='U',comments='%')
		if trans_tab.ndim > 1:
			freq = trans_tab[:,0].astype(np.float)/1000.0 #GHz
			molec = trans_tab[:,7].astype(np.str) #Molecule label
			intensity = trans_tab[:,2].astype(np.float)
			
			#mask out nonfitted lines
			mask = np.where(intensity > 3.0*std_line[j])
			freq = freq[mask]
			molec = molec[mask]
			
		else:
			freq = np.array(trans_tab[0].astype(np.float)/1000.0) #GHz
			molec = np.array(trans_tab[7].astype(np.str)) #Molecule label
			intensity = np.array(trans_tab[2].astype(np.float))
		

		
		#get list of files
		listOfFiles = os.listdir('FITS/')  
		pattern = str(cores[j]) + '_' + str(number[j]) + '_optical_depth__*.dat'
		
		#load spectral axis
		for tau_file in listOfFiles:  
			if fnmatch.fnmatch(tau_file, pattern):
				tau_freq = np.loadtxt('FITS/' + tau_file, skiprows=4, usecols=0) / 1000.0
				
		#create empty total optical depth array
		tau_dat = np.zeros(tau_freq.size)
		
		#loop over all files
		for tau_file in listOfFiles:  
		
			#extract optical depth files
			if fnmatch.fnmatch(tau_file, pattern):
				
				#load optical depth values
				tau_dat = tau_dat + np.loadtxt('FITS/' + tau_file, skiprows=4, usecols=1)
		
		#load observed spectrum
		data = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_vLSRcorr.dat')
		X = data[:,0] / 1000.0
		Y = data[:,1]		
		
		#load model spectrum
		fit = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_compl.out.dat')
		X_fit = fit[:,0] / 1000.0
		Y_fit = fit[:,1]
		
		#plot spectrum & fit
		plt.ioff()
		plt.figure(1,figsize=(8,4.27)) 
		ax = plt.subplot(311)
		plt.step(X,Y,'k', label='Observed Flux Density',lw=0.5, where='mid')
		plt.step(X_fit,Y_fit,'tab:red', label='XCLASS Fit',lw=0.5,alpha=0.8, where='mid')
		plt.xlim(float(min(X))-0.02, max(X)+0.02)
		xmajor = MultipleLocator(0.5)
		ax.xaxis.set_major_locator(xmajor)
		xminor = MultipleLocator(0.1)
		ax.xaxis.set_minor_locator(xminor)
		ax.tick_params(axis='x', labelcolor='white')
		plt.ylabel('Brightness Temperature (K)')
		plt.ticklabel_format(useOffset=False)
		plt.legend(loc='upper left')

		#plot residuals
		ax = plt.subplot(312)
		res = Y - Y_fit
		plt.step(X,res,color='tab:purple', label='Observed Flux Density $-$ XCLASS Fit',lw=0.5, where='mid')
		plt.ylabel('Residuals (K)')    
		ax.axhline(y=3*std_line[j], xmin=0, xmax=1, ls=':', color='tab:green',lw=0.5, label='$\pm 3\sigma$') #plot threshold
		ax.axhline(y=-3*std_line[j], xmin=0, xmax=1, ls=':', color='tab:green',lw=0.5) #plot threshold
		plt.xlim(float(min(X))-0.02, max(X)+0.02)
		plt.ticklabel_format(useOffset=False, axis='x')
		ymajor = MultipleLocator(5)
		ax.yaxis.set_major_locator(ymajor)
		yminor = MultipleLocator(1)
		ax.yaxis.set_minor_locator(yminor)
		xmajor = MultipleLocator(0.5)
		ax.xaxis.set_major_locator(xmajor)
		xminor = MultipleLocator(0.1)
		ax.xaxis.set_minor_locator(xminor)
		plt.legend(loc='lower left')
		
		#plot optical depth
		ax = plt.subplot(313)
		plt.step(tau_freq,tau_dat,'tab:pink', label='Optical Depth',lw=0.5, where='mid')
		plt.xlabel('Frequency (GHz)')
		plt.ylabel(r'Optical Depth $\tau$')    
		plt.xlim(float(min(X))-0.02, max(X)+0.02)
		plt.ticklabel_format(useOffset=False, axis='x')
		plt.ylim(-np.amax(tau_dat)*0.1, np.amax(tau_dat)*1.1)
		plt.ylabel(r'Optical Depth $\tau$')
		#ymajor = MultipleLocator(1)
		#ax.yaxis.set_major_locator(ymajor)
		#yminor = MultipleLocator(0.1)
		#ax.yaxis.set_minor_locator(yminor)
		#xmajor = MultipleLocator(0.5)
		#ax.xaxis.set_major_locator(xmajor)
		#xminor = MultipleLocator(0.1)
		#ax.xaxis.set_minor_locator(xminor)
		plt.legend(loc='upper left')
		
		#label fitted lines
		if trans_tab.ndim > 1:
			for i in range(0, freq.size):
				plt.annotate(molec[i], xy=(freq[i], 0.0), xytext=(freq[i], np.amax(tau_dat)*4.5), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.1, color='tab:blue',linewidth=0.5), fontsize=3, rotation = 90, alpha=1.0, color='tab:blue')
		else:
			plt.annotate(molec, xy=(freq, 0.0), xytext=(freq, np.amax(tau_dat)*4.5), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.1, color='tab:blue',linewidth=0.5), fontsize=3, rotation = 90, alpha=1.0, color='tab:blue')
		
			
		#save plot
		plt.subplots_adjust(hspace=0.07)		
		plt.savefig('RESULTS/XCLASS_SPECTRA/XCLASSFit_' + str(cores[j]) + '_' + str(number[j]) + '.pdf', format='pdf', bbox_inches='tight')
		plt.close()
		
		
def run_XCLASS_fit_all_fixed(data_directory,working_directory,do_error_estimation,vlsr_molec,vlsr_freq):
	###create molfits and observation xml files to compute total XCLASS fit spectrum
	
	create_XCLASS_molfits_file(vlsr_molec,method='all')
	create_XCLASS_obsxml_file(data_directory,working_directory, do_error_estimation,vlsr_freq, method='all')

	#run XCLASS with fixed parameters
	os.system('casa -c XCLASS_fit_RESULTS_all.py')
	rm_casa_files()
	
	os.system('casa -c XCLASS_optical_depth.py')
	rm_casa_files()
	
	plot_fit_residuals_optical_depth()
	
def compute_T_kin(core,nr,do_error_estimation):
	###extract the kinetic temperature
	
	#load table with fit parameter results
	fit_results = np.loadtxt('RESULTS/TABLES/Results_' + str(core) + '_' + str(nr) + '.dat', dtype='U', comments='#')
	
	#extract molecule tag & column density
	molecule = fit_results[:,0].astype(np.str)
	T = fit_results[:,2].astype(np.str)
	
	if do_error_estimation == 'yes':
		T_err = fit_results[:,7].astype(np.str)
	
	mask_CH3CN, = np.where(molecule == ['CH3CN'])
	mask_HNCO, = np.where(molecule == ['HNCO'])
	mask_H2CO, = np.where(molecule == ['H2CO'])
	
	if T[mask_CH3CN] != 'nan':
		
		T_kin = T[mask_CH3CN].astype(np.float)
		
		if do_error_estimation == 'yes':
			T_kin_err = T_err[mask_CH3CN].astype(np.float)
		
	elif T[mask_HNCO] != 'nan':
		
		T_kin = T[mask_HNCO].astype(np.float)
		
		if do_error_estimation == 'yes':
			T_kin_err = T_err[mask_HNCO].astype(np.float)
		
	elif T[mask_H2CO] != 'nan':
		
		T_kin = T[mask_H2CO].astype(np.float)
		
		if do_error_estimation == 'yes':
			T_kin_err = T_err[mask_H2CO].astype(np.float)
	
	else:
		
		T_kin = 20.0
		if do_error_estimation == 'yes':
			T_kin_err = 10.0
		
	if do_error_estimation == 'yes':
		return T_kin, T_kin_err
		
	if do_error_estimation == 'no':
		return T_kin
	
def determine_H2_col_dens(data_directory,do_error_estimation):
	###determine H2 column density from dust continuum	
	
	#Constants
	h = 6.626 * 10.0 ** (-34.0) #J * s
	k = 1.38 * 10.0 ** (-23.0) #J/K
	c_m_s = 299792458.0 # m/s
	gasdustratio = 150.0
	mu = 2.8
	m_H = 1.67 * 10.0**(-24.0) # in g
	kappa = 0.9 #g/(cm^2)
	
	#create empty H2 column density and mass arrays + error arrays
	N_H2 = np.zeros(cores.size)
	M = np.zeros(cores.size)
	T_kin = np.zeros(cores.size)
	N_H2_err = np.zeros(cores.size)
	M_err = np.zeros(cores.size)
	T_kin_err = np.zeros(cores.size)
	
	#loop over all cores
	for j in range(cores.size):
		
		# get fits filename of datacube
		mask, = np.where(regions == cores[j])[0]
		filename_continuum = filenames_continuum[mask]
		
		#open continuum data		
		hdu = fits.open(data_directory + filename_continuum)[0]
		
		#extract flux at position
		flux = hdu.data # in Jy
		data = flux[0,y_pix[j],x_pix[j]]
		data_err = np.std(hdu.data[0,400:450,400:450])
		
		#extract conversion factor
		delta = hdu.header['CDELT2'] * 3600.0 #arcsec/pixel
		
		#extract beam properties
		bmin = hdu.header['BMIN'] * 3600.0 #arcsec
		bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
		
		#extract rest-frequency
		frequency = hdu.header['RESTFREQ']
		
		#compute beam area in steradian
		beam_FWHM = bmin*bmaj #arcsec
		beam_FWHM_rad = beam_FWHM / (206265.0**2) #radian
		beam = beam_FWHM_rad * np.pi / (4.0 * np.log(2.0)) #steradian
		
		#convert Jy/beam to Jy/pixel
		Jy_beam_to_Jy_pixel = delta*delta/(1.1331*beam_FWHM) #convert Jansky/beam to Jansky/pixel
		
		#compute distance in m
		distance = distances[mask]
		distance = distance * 10.0**3.0 * 3.086 * 10.0**16.0 # in m
		
		#extract kinetic temperature
		if do_error_estimation == 'yes':
			T_kin[j],T_kin_err[j] = compute_T_kin(cores[j],number[j],do_error_estimation)
			
		if do_error_estimation == 'no':
			T_kin[j] = compute_T_kin(cores[j],number[j],do_error_estimation)
		
		#compute Planck function
		planck = 2.0 * h * frequency ** 3.0 / (c_m_s**2.0 * (np.exp(h * frequency /(k * T_kin[j])) -1.0))
		
		#compute H2 column density
		N_H2[j] = data * 10**(-26.0) * gasdustratio / (beam * kappa * planck * mu * m_H) #cm-2
		
		#compute mass
		M[j] = Jy_beam_to_Jy_pixel * 10**4*data * 10**(-26.0) * gasdustratio * distance**2.0 / (kappa * planck) / (1.988*10.0**33) #M_sun
		
		if do_error_estimation == 'yes':
			planck_err = T_kin_err[j]* planck * h * frequency * np.exp(h * frequency /(k * T_kin[j])) / (k * T_kin[j]**2.0 * (np.exp(h * frequency /(k * T_kin[j])) -1.0))
			N_H2_err[j] = N_H2[j] * np.sqrt((data_err/data)**2.0 + (planck_err/planck)**2.0)
			M_err[j] = M[j] * np.sqrt((data_err/data)**2.0 + (planck_err/planck)**2.0)
			
	np.savetxt('RESULTS/TABLES/Physical_Properties.dat', np.c_[cores,number,T_kin,N_H2,M,T_kin_err,N_H2_err,M_err], delimiter=' ', fmt='%s')
	
	
def create_MUSCLE_input(data_directory):
	###create MUSCLE input files

	#load N(H2) table
	N_H2 = np.loadtxt('RESULTS/TABLES/Physical_Properties.dat', delimiter=' ', usecols=3)
	
	#correct format of H2 column density
	N_H2 = ["%.2E" % i for i in N_H2]
	
	#load distances to galactic center
	d_to_GC = np.loadtxt('RESULTS/TABLES/GalacticDistances.dat',usecols=1)
	
	#loop over all cores
	for j in range(cores.size):
		
		#mask out region
		mask_reg, = np.where(regions == cores[j])[0]
		
		#extract distance
		d = d_to_GC[mask_reg]
		ratio = oxygen_16_18_ratio(d)
		
		#load datacube
		hdu = fits.open(data_directory + filenames[mask_reg])[0]
		
		#get beam major and minor axis and frequency resolution
		bmin = hdu.header['BMIN'] * 3600.0 #arcsec
		bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
		
		#create average beam FWHM
		beam_avg = (bmin + bmaj ) / 2.0
		
		#compute beam radius in AU
		beam_rad = np.around(beam_avg*d*10.0**3.0,decimals=1)

		#load table with fit parameter results
		fit_results = np.loadtxt('RESULTS/TABLES/Results_' + str(cores[j]) + '_' + str(number[j]) + '.dat', dtype='U', comments='#')
		
		#extract molecule tag & column density
		molecule = fit_results[:,0].astype(np.str)
		N = fit_results[:,3].astype(np.str)
		
		#mask out bad fits
		mask, = np.where(N != 'nan')
		N = N[mask].astype(np.float)
		molecule = molecule[mask]
		model_name = mol_name_MUSCLE[mask]
		
		#mask out molecules which do not have an muscle input
		mask2 = np.where(model_name != 'None')
		molecule = molecule[mask2]
		model_name = model_name[mask2]
		N = N[mask2]
		
		#create empty array for MUSCLE file input
		NAME = np.zeros(N.size).astype(np.str)
		COLDENS = np.zeros(N.size)
		
		#loop over all fitted molecules
		for i in range(N.size):
			
			#find all positions of that molecule
			count_arr, = np.where(model_name == model_name[i])
			counter = count_arr.size
			
			#copy results if only one entry
			if counter == 1:
				
				#compute CO column density with C18O isotopologue
				if model_name[i] =='C18O':
					
					NAME[i] = 'CO'
					COLDENS[i] = N[i] * ratio
				
					
				else:
					NAME[i] = model_name[i]
					COLDENS[i] = N[i]
					
			#create average if multiple transitions were fitted
			if counter > 1:
				
				N_arr = N[count_arr]
				
				NAME[count_arr[0]] = model_name[i]
				
				for k in range(1,counter):
					NAME[count_arr[k]] = 'None'
				
				COLDENS[i] = np.sum(N_arr) / 2.0
				
			#do nothing if there is no MUSCLE input
			if counter == 0:
				print("Zero entries")
				
				
		#mask out positions with no content
		mask3 = np.where(NAME != 'None')
		NAME = NAME[mask3]
		COLDENS = COLDENS[mask3]
		
		#change format for column density
		COLDENS = ["%.2E" % i for i in COLDENS]
		
		total = NAME.size + 2
		#write results in file
		file = open('RESULTS/MODELINPUT/Model_Input_' + str(cores[j]) + '_' + str(number[j]) + '.dat','w') 
		file.write(
	"""        Species       N(cm^-2)     error        radius(AU) det lim    nul\n"""
+ str(total) + """ \n"""  
"""det     H2            """ + str(N_H2[j]) + """   0.00E+00       """ + (""" """)*(6-len(str(beam_rad))) + str(beam_rad) + """      1    0     0      
det     H2            """ + str(N_H2[j]) + """   0.00E+00       """ + (""" """)*(6-len(str(beam_rad))) + str(beam_rad) + """      1    0     0\n"""
		)
		
		for o in range(NAME.size):
			file.write(      
	"""det     """ + NAME[o] + (""" """)*(14-len(NAME[o])) + COLDENS[o] + """   0.00E+00       """ + (""" """)*(6-len(str(beam_rad))) + str(beam_rad) + """      1    0     0\n"""
			)
		file.close() 
	
	
def plot_fit():
	###plot observed spectrum + fit
	
	#plotting parameters
	plt.rcParams.update(params)
	
	#load noise table
	std_line = np.loadtxt('RESULTS/TABLES/Noise.dat', delimiter=' ', usecols=2)	
		
	#loop over all cores
	for j in range(cores.size):
		
		#load table with line properties	
		trans_tab =np.loadtxt('FITS/' + str(cores[j]) + '_' + str(number[j]) + '_transition_energies.dat',dtype='U',comments='%')
		if trans_tab.ndim > 1:
			freq = trans_tab[:,0].astype(np.float)/1000.0 #GHz
			molec = trans_tab[:,7].astype(np.str) #Molecule label
			intensity = trans_tab[:,2].astype(np.float)
			
			#mask out nonfitted lines
			mask = np.where(intensity > 3.0*std_line[j])
			freq = freq[mask]
			molec = molec[mask]
			
		else:
			freq = np.array(trans_tab[0].astype(np.float)/1000.0) #GHz
			molec = np.array(trans_tab[7].astype(np.str)) #Molecule label
			intensity = np.array(trans_tab[2].astype(np.float))
	
		#load observed spectrum
		data = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_vLSRcorr.dat')
		X = data[:,0] / 1000.0
		Y = data[:,1]		
		
		#load model spectrum
		fit = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_compl.out.dat')
		X_fit = fit[:,0] / 1000.0
		Y_fit = fit[:,1]
		
		#plot spectrum & fit
		plt.ioff()
		plt.figure(1) 
		ax = plt.subplot(111)
		plt.step(X,Y,'k', label='Observed Flux Density',lw=0.5, where='mid')
		plt.step(X_fit,Y_fit,'tab:red', label='XCLASS Fit',lw=0.5,alpha=0.8, where='mid')
		plt.xlim(float(min(X))-0.02, max(X)+0.02)
		xmajor = MultipleLocator(0.5)
		ax.xaxis.set_major_locator(xmajor)
		xminor = MultipleLocator(0.1)
		ax.xaxis.set_minor_locator(xminor)
		plt.ylabel('Brightness Temperature (K)')
		plt.xlabel('Frequency (GHz)')
		plt.ticklabel_format(useOffset=False)
		plt.legend(loc='upper left')

		#label fitted lines
		if trans_tab.ndim > 1:
			for i in range(0, freq.size):
				plt.annotate(molec[i], xy=(freq[i], 0.0), xytext=(freq[i], np.amax(Y)*1.25), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.1, color='tab:blue',linewidth=0.5), fontsize=5, rotation = 90, alpha=1.0, color='tab:blue')
		else:
			plt.annotate(molec, xy=(freq, 0.0), xytext=(freq, np.amax(Y)*1.25), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.1, color='tab:blue',linewidth=0.5), fontsize=5, rotation = 90, alpha=1.0, color='tab:blue')
		
			
		#save plot
		plt.savefig('PLOTS/SPECTRA/XCLASSFit_' + str(cores[j]) + '_' + str(number[j]) + '_single.pdf', format='pdf', bbox_inches='tight')
		plt.close() 


def compute_luminosity_mass_array():
	###extract the region luminosity for each core
	
	L_arr = np.zeros(cores.size)
	M_arr = np.zeros(cores.size)

	for j in range(cores.size):
		mask_reg, = np.where(regions == cores[j])[0]
		L_arr[j] = region_luminosity[mask_reg]
		M_arr[j] = region_mass[mask_reg]

	return L_arr, M_arr
	

def compute_abund_vs_luminosity(do_error_estimation):
	###create relative abundance vs luminosity plots for each molecule
	
	L_arr, M_arr = compute_luminosity_mass_array()
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#loop over all molecules
	for k in range(mol_name_file.size):
		
		#load table with best fit parameters
		fit_results = np.loadtxt('RESULTS/TABLES/Results_' + str(mol_name_file[k]) + '.dat', dtype='U', comments='#')
		
		#mask out bad fits
		N = fit_results[:,4].astype(np.str)
		mask = np.where(N != 'nan')
		
		#load best fit parameters
		N = fit_results[:,4].astype(np.float)

		#apply mask
		N = N[mask]	
		L_arr_mask = L_arr[mask]
		core_label_mask = core_label[mask]
		
		#load error bars
		if do_error_estimation == 'yes':
			
			N_err = fit_results[:,9].astype(np.float)
			N_err = N_err[mask]
		
		#only plot if 30 or more data points 
		if mask[0].size > 0:
			
			#create plot
			fig = plt.figure(1) 
			ax = plt.subplot(1, 1, 1)
			plt.loglog(L_arr_mask, N,color='black',marker='o', markersize=2.0,ls='None')
			
			if do_error_estimation == 'yes':
				plt.errorbar(L_arr_mask, N, yerr=N_err, xerr=None, ecolor='black', fmt='none')

			plt.xlabel(r'Region Luminosity ($10^4$ $L_\odot$)')
			plt.ylabel(r'N(' + mol_name_plot[k] + ') (cm$^{-2}$)')
			plt.savefig('PLOTS/ABUNDANCES/L_' + str(mol_name_file[k]) + '.pdf', format='pdf', bbox_inches='tight')
			plt.close()
			
		
def compute_abund_vs_mass(do_error_estimation):
	###create relative abundance vs mass plots for each molecule
	
	L_arr, M_arr = compute_luminosity_mass_array()
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#loop over all molecules
	for k in range(mol_name_file.size):
		
		#load table with best fit parameters
		fit_results = np.loadtxt('RESULTS/TABLES/Results_' + str(mol_name_file[k]) + '.dat', dtype='U', comments='#')
		
		#mask out bad fits
		N = fit_results[:,4].astype(np.str)
		mask = np.where(N != 'nan')
		
		#load best fit parameters
		N = fit_results[:,4].astype(np.float)
		
		#apply mask
		N = N[mask]		
		M_arr_mask = M_arr[mask]
		
		#load error bars
		if do_error_estimation == 'yes':
			
			N_err = fit_results[:,9].astype(np.float)
			N_err = N_err[mask]
		
		#only plot if 30 or more data points 
		if mask[0].size > 0:

			#create plot
			fig = plt.figure(1) 
			ax = plt.subplot(1, 1, 1)
			plt.loglog(M_arr_mask, N,color='black',marker='o', markersize=2.0,ls='None')
			
			if do_error_estimation == 'yes':
				plt.errorbar(M_arr_mask, N, yerr=N_err, xerr=None, ecolor='black', fmt='none')

			plt.xlabel(r'Region Mass ($M_\odot$)')
			plt.ylabel(r'N(' + mol_name_plot[k] + ') (cm$^{-2}$)')
			plt.savefig('PLOTS/ABUNDANCES/M_' + str(mol_name_file[k]) + '.pdf', format='pdf', bbox_inches='tight')
			plt.close()
		
		
def compute_abund_vs_T_kin(do_error_estimation):
	###create relative abundance vs kinetic temperature plots for each molecule
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#load kinetic temperature table
	T_kin = np.loadtxt('RESULTS/TABLES/Physical_Properties.dat', delimiter=' ', usecols=2)
	
	if do_error_estimation == 'yes':
		T_kin_err = np.loadtxt('RESULTS/TABLES/Physical_Properties.dat', delimiter=' ', usecols=5)

	#loop over all molecules
	for k in range(mol_name_file.size):
		
		#load table with best fit parameters
		fit_results = np.loadtxt('RESULTS/TABLES/Results_' + str(mol_name_file[k]) + '.dat', dtype='U', comments='#')
		
		#mask out bad fits
		N = fit_results[:,4].astype(np.str)
		mask = np.where(N != 'nan')
		
		#load best fit parameters
		N = fit_results[:,4].astype(np.float)
		
		#apply mask
		N = N[mask]		
		T_kin_mask = T_kin[mask]
		
		#load error bars
		if do_error_estimation == 'yes':
			
			N_err = fit_results[:,9].astype(np.float)
			N_err = N_err[mask]
			T_kin_mask_err = T_kin_err[mask]
		
		#only plot if 30 or more data points 
		if mask[0].size > 0:

			#create plot
			fig = plt.figure(1) 
			ax = plt.subplot(1, 1, 1)
			plt.loglog(T_kin_mask, N,color='black',marker='o', markersize=2.0,ls='None')
			
			if do_error_estimation == 'yes':
				plt.errorbar(T_kin_mask, N, yerr=N_err, xerr=T_kin_mask_err, ecolor='black', fmt='none')

			plt.xlabel(r'Kinetic Temperature (K)')
			plt.ylabel(r'N(' + mol_name_plot[k] + ') (cm$^{-2}$)')
			plt.savefig('PLOTS/ABUNDANCES/T_kin_' + str(mol_name_file[k]) + '.pdf', format='pdf', bbox_inches='tight')
			plt.close()
					
				
def compute_abund_vs_abund(do_error_estimation):
	###create relative abundance vs relative abundance plots for each molecule combination
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#loop over all molecules
	for i in range(mol_name_file.size):
		for j in range(mol_name_file.size):
			if (i != j) and (j > i):
		
				#load table with best fit parameters
				fit_results_1 = np.loadtxt('RESULTS/TABLES/Results_' + str(mol_name_file[i]) + '.dat', dtype='U', comments='#')
				fit_results_2 = np.loadtxt('RESULTS/TABLES/Results_' + str(mol_name_file[j]) + '.dat', dtype='U', comments='#')
				
				#mask out bad fits
				N_1 = fit_results_1[:,4].astype(np.str)
				N_2 = fit_results_2[:,4].astype(np.str)
				mask = np.where((N_1 != 'nan') & (N_2 != 'nan'))
				
				#load best fit parameters
				N_1 = fit_results_1[:,4].astype(np.float)
				N_2 = fit_results_2[:,4].astype(np.float)
				
				#apply mask
				N_1 = N_1[mask]	
				N_2 = N_2[mask]	
				
				#load error bars
				if do_error_estimation == 'yes':
			
					N_1_err = fit_results_1[:,9].astype(np.float)
					N_1_err = N_1_err[mask]
					N_2_err = fit_results_2[:,9].astype(np.float)
					N_2_err = N_2_err[mask]

		
	
				#only plot if 30 or more data points 
				if mask[0].size > 0:
					
					#create plot
					fig = plt.figure(1) 
					ax = plt.subplot(1, 1, 1)
					plt.loglog(N_1, N_2,color='black',marker='o', markersize=2.0,ls='None')
					
					if do_error_estimation == 'yes':
						plt.errorbar(N_1, N_2, yerr=N_2_err, xerr=N_1_err, ecolor='black', fmt='none')
						
					plt.xlabel(r'N(' + mol_name_plot[i] + ') (cm$^{-2}$)')
					plt.ylabel(r'N(' + mol_name_plot[j] + ') (cm$^{-2}$)')
					
					plt.savefig('PLOTS/ABUNDANCES/ab_vs_ab_' + str(mol_name_file[i]) + '_' + str(mol_name_file[j]) + '.pdf', format='pdf', bbox_inches='tight')
					plt.close()


def compute_abund_vs_abund_noH2(do_error_estimation):
	###create relative abundance vs relative abundance plots for each molecule combination, with respect to molecule (excluding H2)
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#loop over all molecules
	for k in range(mol_name_file.size):
		for i in range(mol_name_file.size):
			for j in range(mol_name_file.size):
				if (i != j) and (j > i) and (k != i) and (k != j):
			
					#load table with best fit parameters
					fit_results_1 = np.loadtxt('RESULTS/TABLES/Results_' + str(mol_name_file[i]) + '.dat', dtype='U', comments='#')
					fit_results_2 = np.loadtxt('RESULTS/TABLES/Results_' + str(mol_name_file[j]) + '.dat', dtype='U', comments='#')
					fit_results_3 = np.loadtxt('RESULTS/TABLES/Results_' + str(mol_name_file[k]) + '.dat', dtype='U', comments='#')
					
					#mask out bad fits
					N_1 = fit_results_1[:,4].astype(np.str)
					N_2 = fit_results_2[:,4].astype(np.str)
					N_rel = fit_results_3[:,4].astype(np.str)
					mask = np.where((N_1 != 'nan') & (N_2 != 'nan') & (N_rel != 'nan'))
					
					#load best fit parameters
					N_1 = fit_results_1[:,4].astype(np.float)
					N_2 = fit_results_2[:,4].astype(np.float)
					N_rel = fit_results_3[:,4].astype(np.float)
					
					#apply mask
					N_1 = N_1[mask]	
					N_2 = N_2[mask]	
					N_rel_mask = N_rel[mask]
					
					#compute relative abundance
					relative_abundance_1 = N_1/N_rel_mask
					relative_abundance_2 = N_2/N_rel_mask
					
					#load error bars
					if do_error_estimation == 'yes':
			
						N_1_err = fit_results_1[:,9].astype(np.float)
						N_1_err = N_1_err[mask]

						N_2_err = fit_results_2[:,9].astype(np.float)
						N_2_err = N_2_err[mask]

						N_rel_err = fit_results_3[:,9].astype(np.float)
						N_rel_err = N_rel_err[mask]

						
						relative_abundance_1_err = relative_abundance_1 * np.sqrt((N_1_err/N_1)**2.0 + (N_rel_err/N_rel_mask)**2.0)
						relative_abundance_2_err = relative_abundance_2 * np.sqrt((N_2_err/N_2)**2.0 + (N_rel_err/N_rel_mask)**2.0)
			

						
					#only plot if 30 or more data points 
					if mask[0].size > 0:
						
						#create plot
						fig = plt.figure(1) 
						ax = plt.subplot(1, 1, 1)
						plt.loglog(relative_abundance_1, relative_abundance_2,color='black',marker='o', markersize=2.0,ls='None')
						
						if do_error_estimation == 'yes':
							plt.errorbar(relative_abundance_1, relative_abundance_2, yerr=relative_abundance_2_err, xerr=relative_abundance_1_err, ecolor='black', fmt='none')
							
						plt.xlabel(r'N(' + mol_name_plot[i] + ')/N(' + mol_name_plot[k] + ')')
						plt.ylabel(r'N(' + mol_name_plot[j] + ')/N(' + mol_name_plot[k] + ')')
						plt.savefig('PLOTS/ABUNDANCES/ab_vs_ab_' + str(mol_name_file[i]) + '_' + str(mol_name_file[j]) + '_' + str(mol_name_file[k]) + '.pdf', format='pdf', bbox_inches='tight')
						plt.close()
			
					
def compute_N_H2_from_C18O(do_error_estimation):
	###compute N(H2) from N(C18O)
	
	#load distances to galactic center
	d_to_GC = np.loadtxt('RESULTS/TABLES/GalacticDistances.dat',usecols=1)
	
	#create empty N(H2) array
	N_H2_from_C18O = np.zeros(cores.size)
	
	if do_error_estimation == 'yes':
		N_H2_from_C18O_err = np.zeros(cores.size)
		N_H2_from_C18O_tag = np.zeros(cores.size).astype(np.str)
	
	#load table with best fit parameters
	fit_results = np.loadtxt('RESULTS/TABLES/Results_C18O.dat', dtype='U', comments='#')
	
	#load best fit parameters
	N = fit_results[:,4].astype(np.str)
	
	if do_error_estimation == 'yes':
		N_err = fit_results[:,9].astype(np.str)

	#loop over all cores (j)
	for j in range(cores.size):
		
		#mask out position of corresponding region
		mask_reg, = np.where(regions == cores[j])[0]
		
		#extract distance
		d = d_to_GC[mask_reg]
		
		#compute isotopologue ratio
		ratio = oxygen_16_18_ratio(d)	
		
		#take C18O column density when available
		if N[j] != 'nan':
			
			#calculate N(H2)
			N_H2_from_C18O[j] = ratio * np.float(N[j]) * 10.0**4.0
			
			if do_error_estimation == 'yes':
				N_H2_from_C18O_err[j] = N_H2_from_C18O[j] * np.float(N_err[j]) / np.float(N[j])
			
		#otherwise use an upper limit
		else:
			
			#calculate N(H2)
			N_H2_from_C18O[j] = ratio * 10.0**16.0 * 10.0**4.0
			
			if do_error_estimation == 'yes':
				N_H2_from_C18O_err[j] = N_H2_from_C18O[j] * 0.5
				N_H2_from_C18O_tag[j] = 'upplim'
				
	if do_error_estimation == 'yes':
		return N_H2_from_C18O, N_H2_from_C18O_err, N_H2_from_C18O_tag
		
	if do_error_estimation == 'no':
		return N_H2_from_C18O
	
		
def N_H2_comparison(do_error_estimation):
	#plot N(H2) from N(C18O) vs. N(H2) from dust continuum
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#compute N(H2) from gas
	if do_error_estimation == 'yes':
		N_H2_from_C18O, N_H2_from_C18O_err, N_H2_from_C18O_tag = compute_N_H2_from_C18O(do_error_estimation)
	if do_error_estimation == 'no':
		N_H2_from_C18O = compute_N_H2_from_C18O(do_error_estimation)
	
	#load N(H2) table (from dust)
	N_H2 = np.loadtxt('RESULTS/TABLES/Physical_Properties.dat', delimiter=' ', usecols=3)
	
	if do_error_estimation == 'yes':
		N_H2_err = np.loadtxt('RESULTS/TABLES/Physical_Properties.dat', delimiter=' ', usecols=6)
		
	#create plot
	fig = plt.figure(1) 
	ax = plt.subplot(1, 1, 1)
	plt.loglog(N_H2, N_H2_from_C18O,color='black',marker='o', markersize=2.0,ls='None')
	
	if do_error_estimation == 'yes':
		plt.errorbar(N_H2, N_H2_from_C18O, yerr=N_H2_from_C18O_err, xerr=N_H2_err, ecolor='black', fmt='none')
		
	plt.xlabel(r'N(H$_2$) from dust (cm$^{-2}$)')
	plt.ylabel(r'N(H$_2$) from C$^{18}$O (cm$^{-2}$)')
	plt.savefig('PLOTS/ABUNDANCES/N_H2_comparison.pdf', format='pdf', bbox_inches='tight')
	plt.close()		
	
	
def abundance_analysis(do_error_estimation):	
	
	compute_abund_vs_luminosity(do_error_estimation)
	
	compute_abund_vs_mass(do_error_estimation)
	
	compute_abund_vs_T_kin(do_error_estimation)
	
	compute_abund_vs_abund(do_error_estimation)
	
	compute_abund_vs_abund_noH2(do_error_estimation)
	
	
	N_H2_comparison(do_error_estimation)
