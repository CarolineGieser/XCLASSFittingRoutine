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
c = 299792.458 #km/s
GC_coords = SkyCoord('17 45 40.04 -29 00 28.1', unit=(u.hourangle, u.deg),distance=8.122*u.kpc)

def make_directory(dirName):
	### setup working directory 
	
	try:
		#create required directories if not already done
		
	    os.makedirs(dirName)    
	    print('Directory ' + dirName +  ' created. ')
	    
	except FileExistsError:
		#do not create directories if they already exist so nothing will be overwritten
		
		print('Directory ' + dirName +  ' already exists.')	
		
def workingdir():
	### print working directory 
	
	working_dir = os.getcwd() + '/'
	print('Working Directory: ' + working_dir)
	
	return working_dir
	
def rm_previous_fitting_results():
	###WARNING! All results will be lost!
	
	os.system('rm -r FITS')
	os.system('rm -r PLOTS')
	os.system('rm -r Results')
	print('Removed previous results!')
	
def setup_directory(delete_previous_results=False):
	### setup directory with all folders
	
	if delete_previous_results==True:
		# remove previous results		
		rm_previous_fitting_results()
		
	# get working directory
	working_directory = workingdir()
	
	#create directories
	make_directory(dirName='FITS')
	make_directory(dirName='PLOTS')
	make_directory(dirName='PLOTS/BADFIT')
	make_directory(dirName='PLOTS/GOODFIT')
	make_directory(dirName='PLOTS/VLSR')
	make_directory(dirName='PLOTS/CONTINUUM')
	make_directory(dirName='PLOTS/SPECTRA_LINEID')
	make_directory(dirName='Results')
	
	print('Working directory: ' + working_directory)
	return working_directory
	
def load_input_table():
	#### input table of regions
	
	input_tab=np.loadtxt('input.dat', dtype='U', comments='#',usecols=1)
	
	data_directory = input_tab[0].astype(np.str) #directory in which fits datacubes are stored
	do_error_estimation = input_tab[1].astype(np.str) #chose if error estimation or not
	channel1 = input_tab[2].astype(np.int) #lower channel for noise estimation
	channel2 = input_tab[3].astype(np.int) #upper channel for noise estimation
	
	return data_directory, do_error_estimation, channel1 , channel2
	
def load_regions_table():
	#### input table of regions
	
	regions_tab=np.loadtxt('regions.dat', dtype='U', comments='#')
	
	regions = regions_tab[:,0].astype(np.str) #CORE region name
	regions_plot = regions_tab[:,1].astype(np.str) #CORE region name for plotting
	distances = regions_tab[:,2].astype(np.float) #Distance (kpc)
	filenames = regions_tab[:,3].astype(np.str) #fits filenames of spectral line datacubes
	filenames_continuum = regions_tab[:,4].astype(np.str) #fits filenames of continuum data
	
	return regions, regions_plot, distances, filenames, filenames_continuum


def load_cores_table():
	#### input table of cores
	
	cores_tab=np.loadtxt('cores.dat', dtype='U', comments='#')
	cores = cores_tab[:,0].astype(np.str)
	number = cores_tab[:,1].astype(np.int)
	x_pix = cores_tab[:,2].astype(np.int)
	y_pix = cores_tab[:,3].astype(np.int)
	core_label = cores_tab[:,4].astype(np.str)
		
	return cores, number, x_pix, y_pix, core_label
	
	
def load_molecules_table():
	### input table of molecules
	
	mol_data=np.loadtxt('molecules.dat', dtype='U', comments='%')
	
	mol_name=mol_data[:,0] #XCLASS molecule label
	mol_name_file=mol_data[:,1] #filename molecule label
	
	return mol_name, mol_name_file
	
	
def load_molecule_ranges_table():
	### input table of molecule ranges
	
	mol_ranges=np.loadtxt('molecule_ranges.dat', dtype='U', comments='%')
	
	mol_ranges_name=mol_ranges[:,0] #XCLASS molecule label and range index
	mol_ranges_low=mol_ranges[:,1].astype(np.float) #lower frequency limit
	mol_ranges_upp=mol_ranges[:,2].astype(np.float) #upper frequency limit
	
	return mol_ranges_name, mol_ranges_low, mol_ranges_upp

def check_error_estimation(do_error_estimation):
	if do_error_estimation == 'yes':
		tag = np.array(['','_lowErr','_uppErr'])
	elif do_error_estimation == 'no':
		tag = np.array([''])
	else:
		print('Only yes and no are allowed for do_error_estimation parameter in input.dat!')
	return tag
	
@jit
def determine_noise(data_directory, regions, filenames, cores, number, x_pix, y_pix,channel1,channel2):
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
		cube = SpectralCube.read(datafile)  
		
		#extract spectrum
		subcube = cube[:,y_pix[j],x_pix[j]]
		#extract line-free part of spectrum
		subcube = subcube[channel1:channel2+1]
		
		#compute standard deviation of the spectrum
		std_line[j] = np.around(subcube.std().value, decimals=2)
		
		#print results
		print('Determined Noise in ' + str(cores[j]) + ' ' + str(number[j]) + ' Spectrum: ' + str(std_line[j])+ ' K')
		
	#save noise values in file
	np.savetxt('Results/Noise.dat', np.c_[cores,number,std_line], delimiter=' ', fmt='%s')
	
	return std_line
	
	
def extract_spectrum_init(data_directory, regions, filenames, cores, number, x_pix, y_pix):
	### extract and save spectrum of each core which will be fitted with XCLASS (only C18O line)
	
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
		#convert spectral axis to MHz
		cube = SpectralCube.read(datafile).with_spectral_unit(u.MHz, velocity_convention='radio',rest_value=restfreq * u.Hz)  
	
		#extract spectrum
		sub_cube = cube[:,y_pix[j],x_pix[j]]
	
		#save spectra
		np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '.dat', np.c_[sub_cube.spectral_axis.value/(1.0 - (vlsr/c)),sub_cube.value])
		
		#print results
		print('Extracted ' + str(cores[j]) + ' ' + str(number[j]) + ' Spectrum!')
		
		
def create_XCLASS_molfits_file(cores, number,mol_name,mol_name_file,method):
	### create input molfits for XCLASS fitting
	
	#molfits file for initial C18O fit
	if method == 'vLSR':
		
		#open and write molfits file
		file = open('FITS/molecules_vLSR.molfit','w') 
		file.write(
"""CO-18;v=0;   1 \n"""
"""y   0.1   1.0   0.4   y   5.0   300.0   100.0   y   1.0E+10   1.0E+21   1.0e+15   y   0.5   20.0   5.0   y   -15.0   15.0   0.0   c"""
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
"""y   0.1   1.0   0.4   y   5.0   220.0   100.0   y   1.0E+10   1.0E+21   1.0e+15   y   0.5   20.0   5.0   y   -6.0   6.0   0.0   c"""
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
				fitdata = np.loadtxt('Results/Results_' + str(mol_name_file[k]) +'.dat', dtype='U', comments='#')
				Theta_source = fitdata[:,2]
				T = fitdata[:,3]
				N = fitdata[:,4]
				Delta_v = fitdata[:,5]
				v_off = fitdata[:,6]
				
				#mask bad fits
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
	
def create_XCLASS_obsxml_file(data_directory,working_directory,regions,filenames,cores, number,mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp, do_error_estimation,method):
	### create input observations xml files for XCLASS fitting
	
	#observational xml file for initial C18O fit
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
			delta_nu = hdu.header['CDELT3'] #MHz
			
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
         <MinExpRange>219530.0</MinExpRange>
         <MaxExpRange>219590.0</MaxExpRange>
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
			
			#load frequency ranges for molecule
			mask = np.where(mol_ranges_name == mol_name[k])
			mol_ranges_name_mask = mol_ranges_name[mask]
			mol_ranges_low_mask = mol_ranges_low[mask]
			mol_ranges_upp_mask = mol_ranges_upp[mask]
			
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
					delta_nu = hdu.header['CDELT3'] #MHz
					
					#create average beam FWHM
					beam_avg = (bmin + bmaj ) / 2.0
					
					#open and write observation xml file
					file = open('FITS/observation_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + str(tag[z]) + '.xml','w') 
					file.write(
"""<?xml version="1.0" encoding="UTF-8"?>
<ExpFiles>
 <NumberExpFiles>1</NumberExpFiles>
 <file>
     <FileNamesExpFiles>""" + str(working_directory) +"""FITS/spectrum_""" + str(cores[j]) + """_""" + str(number[j]) + str(tag[z]) + """.dat</FileNamesExpFiles>
     <ImportFilter>xclassASCII</ImportFilter>
     <NumberExpRanges>""" + str(mol_ranges_name_mask.size) + """</NumberExpRanges>\n"""
			        )
			        
			      #loop over all frequency ranges
					for p in range(mol_ranges_name_mask.size):
						file.write(
"""     <FrequencyRange>
         <MinExpRange>""" + str(mol_ranges_low_mask[p]) + """</MinExpRange>
         <MaxExpRange>""" + str(mol_ranges_upp_mask[p]) + """</MaxExpRange>
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
			delta_nu = hdu.header['CDELT3'] #MHz
			
			#create average beam FWHM
			beam_avg = (bmin + bmaj ) / 2.0
		
			#open and write observation.xml file (optimized for C18O line!)
			file = open('FITS/observation_' + str(cores[j]) + '_' + str(number[j]) + '_all.xml','w') 
			file.write(
"""<?xml version="1.0" encoding="UTF-8"?>
<ExpFiles>
 <NumberExpFiles>1</NumberExpFiles>
 <file>
     <FileNamesExpFiles>""" + str(working_directory) +"""FITS/spectrum_""" + str(cores[j]) + """_""" + str(number[j]) + """.dat</FileNamesExpFiles>
     <ImportFilter>xclassASCII</ImportFilter>
     <NumberExpRanges>1</NumberExpRanges>
     <FrequencyRange>
         <MinExpRange>217000.0</MinExpRange>
         <MaxExpRange>221000.0</MaxExpRange>
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
			
def create_XCLASS_isoratio_file(regions,data_directory,filenames,distances):
	### create input iso-ratio files for XCLASS fitting
	
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
	print('Created iso-ratio files!') 
	
def setup_XCLASS_files(data_directory, working_directory, regions, filenames, distances, cores, number, x_pix, y_pix, mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp,do_error_estimation):
	
	create_XCLASS_isoratio_file(regions,data_directory,filenames,distances)
	create_XCLASS_molfits_file(cores, number,mol_name,mol_name_file,method='vLSR')
	create_XCLASS_obsxml_file(data_directory,working_directory,regions, filenames,cores, number,mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp, do_error_estimation, method='vLSR')
	create_XCLASS_molfits_file(cores, number,mol_name,mol_name_file,method='single')
	create_XCLASS_obsxml_file(data_directory,working_directory,regions, filenames,cores, number,mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp, do_error_estimation, method='single')
	

def get_velocity_offset(cores, number):
	### extract bestfit systemic velocity and plot fitted C18O line
	
	#set plot parameters
	plt.rcParams.update(params)
	
	noise = np.loadtxt('Results/Noise.dat',usecols=2)
	
	#array in which v_off values are stored
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
		plt.step(X,Y,'k', label='Data')
		plt.step(X_fit,Y_fit,'r', label='Fit',linestyle='--')
		plt.axhline(y=5.0*noise[j], xmin=0, xmax=1,color='blue',label='5$\sigma$')
		plt.xlim(np.amin(X_fit)-0.001,np.amax(X_fit)+0.001)
		plt.xlabel('Frequency [GHz]')
		plt.ylabel('Brightness Temperature [K]')
		plt.ticklabel_format(useOffset=False)
		plt.annotate(str(cores[j]) + ' ' + str(number[j]), xy=(0.1, 0.9), xycoords='axes fraction')  
		plt.legend(loc='upper right')
		plt.ylim(np.amin(Y_fit)-0.5,np.amax(Y_fit)*1.5)
		plt.savefig('PLOTS/VLSR/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_vLSR.png', format='png', bbox_inches='tight')
		plt.close()
		
		#extract v_off
		fit_results = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(number[j]) + '_vLSR.molfit', dtype='U', skiprows=1) 
		VelocityOffset[j] = fit_results[19] 
	
	#save v_off values in file				
	np.savetxt('Results/VelocityOffset.dat', np.c_[cores,number,VelocityOffset], delimiter=' ', fmt='%s')
	print('Extracted all velocity offsets!')
	
	return VelocityOffset
	
		
def extract_spectrum(data_directory, regions, filenames, cores, number, x_pix, y_pix, v_off,do_error_estimation):
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
		#convert spectral axis to MHz
		cube = SpectralCube.read(datafile).with_spectral_unit(u.MHz, velocity_convention='radio',rest_value=restfreq * u.Hz)
	
		#extract spectrum
		sub_cube = cube[:,y_pix[j],x_pix[j]]
		
		#compute core systemic velocity from C18O fit
		vlsr_core = vlsr+v_off[j]
	
		#save spectra
		np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '.dat', np.c_[sub_cube.spectral_axis.value/(1.0 - (vlsr_core/c)),sub_cube.value])
		
		tag = check_error_estimation(do_error_estimation)
		
		if do_error_estimation =='yes':
			
			np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + tag[1] + '.dat', np.c_[sub_cube.spectral_axis.value/(1.0 - (vlsr_core/c)),sub_cube.value*0.8])
			np.savetxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + tag[2] +'.dat', np.c_[sub_cube.spectral_axis.value/(1.0 - (vlsr_core/c)),sub_cube.value*1.2])
		
		print('Extracted ' + str(cores[j]) + ' ' + str(number[j]) + ' Spectrum (spectral axis corrected for systemic velocity)!')
	
	
def rm_casa_files():
	### remove unnecessary casa files
	
	os.system('rm casa*.log')
	os.system('rm *.last')
	
	print('Removed casa log files!')
	
	
def run_XCLASS_fit(data_directory, regions, filenames,cores, number, x_pix, y_pix,std_line,do_error_estimation,C18O_vlsr=True):
	### perform XCLASS fit in casa
	
	# use C18O line to determine precise systemic velocity
	if C18O_vlsr==True:
		
		#run inital XCLASS Fit on C18O line to determine the vLSR of each core
		os.system('casa -c XCLASS_VLSR_determination.py')
		rm_casa_files()
		
		#get velocity offset from C18O line fit
		v_off = get_velocity_offset(cores, number)
	
	# use general region systemic velocity
	else:
		v_off = np.zeros(cores.size)
		
	#extract spectra & correct for systemic velocity
	extract_spectrum(data_directory, regions, filenames, cores, number, x_pix, y_pix, v_off,do_error_estimation)
		
	#run XCLASS Fit
	os.system('casa -c XCLASS_fit.py')
	rm_casa_files()
	
@jit
def extract_results(cores, number, mol_name_file,std_line,do_error_estimation):
	###extract best fit parameters from .molfit files
	
	##set plot parameters
	plt.rcParams.update(params)
	
	#create empty best fit parameter arrays
	Theta_source = np.zeros(shape=(mol_name_file.size,cores.size))
	T = np.zeros(shape=(mol_name_file.size,cores.size))
	N = np.zeros(shape=(mol_name_file.size,cores.size))
	Delta_v = np.zeros(shape=(mol_name_file.size,cores.size))
	v_off = np.zeros(shape=(mol_name_file.size,cores.size))
	
	if do_error_estimation == 'yes':
		Theta_source_err = np.zeros(shape=(mol_name_file.size,cores.size))
		T_err = np.zeros(shape=(mol_name_file.size,cores.size))
		N_err = np.zeros(shape=(mol_name_file.size,cores.size))
		Delta_v_err = np.zeros(shape=(mol_name_file.size,cores.size))
		v_off_err = np.zeros(shape=(mol_name_file.size,cores.size))

	#loop over all fitted molecules (k) and cores (j)
	for k in range(mol_name_file.size):
		
		for j in range(cores.size):
			
			#load data
			data = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '.dat')
			X = data[:,0]/1000.0 #GHz
			Y = data[:,1] #K
			
			#load fit
			fit = np.loadtxt('FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '.out.dat')
			X_fit = fit[:,0]/1000.0 #GHz
			Y_fit = fit[:,1] #K
			
			#exclude non-fitted spectral range
			mask = np.where(Y_fit > std_line[j])
			Y_fit_masked = Y_fit[mask]
				
			
			
			#compute minimum and maximum frequency where XCLASS fit was performed
			start = np.isclose(X, np.amin(X_fit), rtol=1e-06, atol=1e-08, equal_nan=False)
			for i in range(start.size):
				if start[i] == True:
					low = i
			stop = np.isclose(X, np.amax(X_fit), rtol=1e-06, atol=1e-08, equal_nan=False)
			for l in range(stop.size):
				if stop[l] == True:
					upp = l+1
			
			#apply ranges to data
			X = X[low:upp]
			Y = Y[low:upp]
			
			
			#signal-to-noise-ratio
			threshold = 5.0 

			#plot spectrum and fit
			plt.ioff()
			plt.figure() 
			plt.step(X,Y,'k', label='Data')
			plt.step(X_fit,Y_fit,'r', label='Fit',linestyle='dashed')
			plt.xlim(np.amin(X_fit)-0.001,np.amax(X_fit)+0.001)
			plt.axhline(y=threshold*std_line[j], xmin=0, xmax=1)
			plt.xlabel('Frequency [GHz]')
			plt.ylabel('Brightness Temperature [K]')
			plt.ticklabel_format(useOffset=False)
			plt.annotate(str(cores[j]) + ' ' + str(number[j]) + ' ' + str(mol_name_file[k]), xy=(0.1, 0.9), xycoords='axes fraction')  
			plt.legend(loc='upper right')
			
			#check, if fitted flux > 0 K and if signal-to-noise ratio of fitted flux >= 5
			if Y_fit_masked.size > 0 and np.amax(Y_fit_masked) >= threshold*std_line[j]:
				
				#save plot
				plt.ylim(np.amin(Y)-0.5,np.amax(Y)*1.1)
				plt.savefig('PLOTS/GOODFIT/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '.png', format='png', bbox_inches='tight')
				plt.close()
				
				#load best fit molfits file
				a = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '.molfit', dtype='U', skiprows=1) 
				Theta_source[k,j] = a[3]
				T[k,j] = a[7]
				N[k,j] = a[11]
				Delta_v[k,j] = a[15]
				v_off[k,j] = a[19] 
				
				#error computation
				if do_error_estimation == 'yes':
					
					#load best fit molfits file (upper error)
					b = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '_uppErr.molfit', dtype='U', skiprows=1) 
				
					#load best fit molfits file (lower error)
					c = np.loadtxt('FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '_lowErr.molfit', dtype='U', skiprows=1) 
					
					#compute average deviation from upper/lower errors to best fit parameter
					Theta_source_err[k,j] = (np.abs(np.float(b[3])-np.float(a[3])) + np.abs(np.float(c[3])-np.float(a[3]))) * 0.5
					T_err[k,j] = (np.abs(np.float(b[7])-np.float(a[7])) + np.abs(np.float(c[7])-np.float(a[7]))) * 0.5
					N_err[k,j] = (np.abs(np.float(b[11])-np.float(a[11])) + np.abs(np.float(c[11])-np.float(a[11]))) * 0.5
					Delta_v_err[k,j] = (np.abs(np.float(b[15])-np.float(a[15])) + np.abs(np.float(c[15])-np.float(a[15]))) * 0.5
					v_off_err[k,j] = (np.abs(np.float(b[19])-np.float(a[19])) + np.abs(np.float(c[19])-np.float(a[19]))) * 0.5

			#if no flux was recovered from XCLASS fit, remove best fit parameters
			else:
				
					#save plot
					plt.ylim(np.amin(Y)-0.5,3.0)
					plt.savefig('PLOTS/BADFIT/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_' + str(mol_name_file[k]) + '.png', format='png', bbox_inches='tight')
					plt.close()
					
					#set best fit parameters to nan
					Theta_source[k,j] = np.nan					
					T[k,j] = np.nan
					N[k,j] = np.nan
					Delta_v[k,j] = np.nan
					v_off[k,j] = np.nan
					
					if do_error_estimation == 'yes':
						Theta_source_err[k,j] = np.nan	
						T_err[k,j] = np.nan
						N_err[k,j] = np.nan
						Delta_v_err[k,j] = np.nan
						v_off_err[k,j] = np.nan
					
					print('Excluded ' + str(cores[j]) + ' ' + str(number[j]) + ' ' + str(mol_name_file[k]) + ' !')
					
	#create output table for each molecule	
	for k in range(mol_name_file.size):
		if do_error_estimation == 'yes':
			np.savetxt('Results/Results_' + str(mol_name_file[k]) + '.dat', np.c_[cores,number,Theta_source[k,:],T[k,:], N[k,:],Delta_v[k,:],v_off[k,:],Theta_source_err[k,:],T_err[k,:], N_err[k,:],Delta_v_err[k,:],v_off_err[k,:]], delimiter=' ', fmt='%s')
		elif do_error_estimation == 'no':
			np.savetxt('Results/Results_' + str(mol_name_file[k]) + '.dat', np.c_[cores,number,Theta_source[k,:],T[k,:], N[k,:],Delta_v[k,:],v_off[k,:]], delimiter=' ', fmt='%s')
				
	print('Extracted all best fit parameters for each molecules!')
	
	#create output table for each core
	for j in range(cores.size):
		if do_error_estimation == 'yes':
			np.savetxt('Results/Results_' + str(cores[j]) + '_' + str(number[j]) + '.dat', np.c_[mol_name_file, Theta_source[:,j], T[:,j], N[:,j],Delta_v[:,j],v_off[:,j], Theta_source_err[:,j],T_err[:,j], N_err[:,j],Delta_v_err[:,j],v_off_err[:,j]], delimiter=' ', fmt='%s')
		elif do_error_estimation == 'no':
			np.savetxt('Results/Results_' + str(cores[j]) + '_' + str(number[j]) + '.dat', np.c_[mol_name_file, Theta_source[:,j], T[:,j], N[:,j],Delta_v[:,j],v_off[:,j]], delimiter=' ', fmt='%s')
			
	print('Extracted all best fit parameters for each core!')		
	
			
def plot_results_cores(cores, number,do_error_estimation):
	### plot results for temperature, column density, linewidth and velocity offset for each core
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#loop over all cores
	for j in range(cores.size):
		
		#load table with best fit parameters
		fit_results = np.loadtxt('Results/Results_' + str(cores[j]) + '_' + str(number[j]) + '.dat', dtype='U', comments='#')
		
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
			plt.bar(x, T, width=0.8, yerr=T_Err,align='center',color='firebrick',edgecolor='black',linewidth=0.5, log=False)
		elif do_error_estimation == 'no':
			plt.bar(x, T, width=0.8,align='center',color='firebrick',edgecolor='black',linewidth=0.5, log=False)
		plt.ylabel('Temperature [K]')
		plt.xticks(x, molecule, rotation='vertical')
		ax.xaxis.set_ticklabels([])
		ax = plt.subplot(2, 2, 2)
		if do_error_estimation == 'yes':
			plt.bar(x, N, width=0.8, yerr=N_Err, align='center',color='steelblue',edgecolor='black',linewidth=0.5, log=True)
		elif do_error_estimation == 'no':
			plt.bar(x, N, width=0.8, align='center',color='steelblue',edgecolor='black',linewidth=0.5, log=True)
		plt.ylabel('Column Density [cm$^{-2}$]')
		plt.xticks(x, molecule, rotation='vertical')
		ax.xaxis.set_ticklabels([])
		ax = plt.subplot(2, 2, 3)
		if do_error_estimation == 'yes':
			plt.bar(x, delta_v, width=0.8, yerr=delta_v_Err,align='center',color='seagreen',edgecolor='black',linewidth=0.5, log=False)
		elif do_error_estimation == 'no':
			plt.bar(x, delta_v, width=0.8,align='center',color='seagreen',edgecolor='black',linewidth=0.5, log=False)
		plt.xticks(x, molecule, rotation='vertical')
		plt.ylabel('Line Width [km s$^{-1}$]')
		ax = plt.subplot(2, 2, 4)
		if do_error_estimation == 'yes':
			plt.bar(x, v_off, width=0.8, yerr=v_off_Err,align='center',color='darkorange',edgecolor='black',linewidth=0.5, log=False)
		elif do_error_estimation == 'no':
				plt.bar(x, v_off, width=0.8,align='center',color='darkorange',edgecolor='black',linewidth=0.5, log=False)
		plt.ylabel('Velocity Offset [km s$^{-1}$]')
		plt.subplots_adjust(wspace=0.3, hspace=0.05)
		plt.xticks(x, molecule, rotation='vertical')
		plt.savefig('Results/Bartchart_' + str(cores[j]) + '_' + str(number[j]) + '.pdf', format='pdf', bbox_inches='tight')
		plt.close()
		
	print('Created barchart plot for each core!')
	
	
def plot_results_molecule(mol_name_file):
	### plot histogram for temperature, column density, linewidth and velocity offset for each molecule
	
	#set plot parameters
	plt.rcParams.update(params)
	
	#loop over all molecules
	for k in range(mol_name_file.size):
		
		#load table with best fit parameters
		fit_results = np.loadtxt('Results/Results_' + str(mol_name_file[k]) + '.dat', dtype='U', comments='#')
		
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
		plt.hist(T, range=(0.0,225.0), log=False, color='firebrick', label=None)
		plt.xlabel('Temperature [K]')
		ax = plt.subplot(2, 2, 2)
		plt.hist(np.log10(N), range=(10.0,21.0), log=False, color='steelblue', label=None)
		plt.xlabel('LOG(Column Density) [LOG(cm$^{-2}$)]')
		ax = plt.subplot(2, 2, 3)
		plt.hist(delta_v, range=(0.5,20), log=False, color='seagreen', label=None)
		plt.xlabel('Line Width [km s$^{-1}$]')
		ax = plt.subplot(2, 2, 4)
		plt.hist(v_off, range=(-6,6), log=False, color='darkorange', label=None)
		plt.xlabel('Velocity Offset [km s$^{-1}$]')
		plt.savefig('Results/Histogram_' + str(mol_name_file[k]) + '.pdf', format='pdf', bbox_inches='tight')
		plt.close()

	print('Created histogram plot for each molecule!')
	
def create_plots(cores, number, mol_name_file,std_line,do_error_estimation):
	###extract all best fit parameters and make plots
	
	#extract all best fit parameters and save tables
	extract_results(cores, number, mol_name_file,std_line,do_error_estimation)
	
	#plot barchart for each core
	plot_results_cores(cores, number,do_error_estimation)
	
	#plot histogram for each molecule
	plot_results_molecule(mol_name_file)


def plot_fit_residuals_optical_depth(cores, number,std_line):
	###plot observed spectrum + fit, residuals and optical depth
	
	#plotting parameters
	plt.rcParams.update(params)
	
		
	#loop over all cores
	for j in range(cores.size):
		
		#load table with line properties	
		trans_tab =np.loadtxt('Results/' + str(cores[j]) + '_' + str(number[j]) + '_transition_energies.dat',dtype='U',comments='%')
		print(trans_tab)
		freq = trans_tab[:,0].astype(np.float)/1000.0 #GHz
		molec = trans_tab[:,7].astype(np.str) #Molecule label
		intensity = trans_tab[:,2].astype(np.float)
		
		#mask out nonfitted lines
		mask = np.where(intensity > 0.0)
		freq = freq[mask]
		molec = molec[mask]
		
		#get list of files
		listOfFiles = os.listdir('Results/')  
		pattern = str(cores[j]) + '_' + str(number[j]) + '_optical_depth__*.dat'
		
		#load spectral axis
		for tau_file in listOfFiles:  
			if fnmatch.fnmatch(tau_file, pattern):
				tau_freq = np.loadtxt('Results/' + tau_file, skiprows=4, usecols=0) / 1000.0
				
		#create empty total optical depth array
		tau_dat = np.zeros(tau_freq.size)
		
		#loop over all files
		for tau_file in listOfFiles:  
		
			#extract optical depth files
			if fnmatch.fnmatch(tau_file, pattern):
				
				#load optical depth values
				tau_dat = tau_dat + np.loadtxt('Results/' + tau_file, skiprows=4, usecols=1)
	
		#load observed spectrum
		data = np.loadtxt('FITS/spectrum_' + str(cores[j]) + '_' + str(number[j]) + '.dat')
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
		plt.plot(X,Y,'k', label='Observed Flux Density',lw=0.5)
		plt.plot(X_fit,Y_fit,'r', label='XCLASS Fit',lw=0.5,alpha=0.8)
		plt.xlim(float(min(X))-0.02, max(X)+0.02)
		xmajor = MultipleLocator(0.5)
		ax.xaxis.set_major_locator(xmajor)
		xminor = MultipleLocator(0.1)
		ax.xaxis.set_minor_locator(xminor)
		ax.tick_params(axis='x', labelcolor='white')
		plt.ylabel('Brightness Temperature [K]')
		plt.ticklabel_format(useOffset=False)
		plt.legend(loc='upper left')

		#plot residuals
		ax = plt.subplot(312)
		res = Y - Y_fit
		plt.plot(X,res,'g-', label='Observed Flux Density $-$ XCLASS Fit',lw=0.5)
		plt.ylabel('Residuals [K]')    
		ax.axhline(y=5*std_line[j], xmin=0, xmax=1, ls='--', color='g',lw=0.5, label='$\pm 5\sigma$') #plot threshold
		ax.axhline(y=-5*std_line[j], xmin=0, xmax=1, ls='--', color='g',lw=0.5) #plot threshold
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
		plt.plot(tau_freq,tau_dat,'b', label='Optical Depth',lw=0.5)
		plt.xlabel('Frequency [GHz]')
		plt.ylabel(r'Optical Depth $\tau$')    
		plt.xlim(float(min(X))-0.02, max(X)+0.02)
		plt.ticklabel_format(useOffset=False, axis='x')
		plt.ylim(-0.1, np.amax(tau_dat)*1.1)
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
		counter = 0
		for i in range(0, freq.size):
			plt.annotate(molec[i], xy=(freq[i], 0.0), xytext=(freq[i], np.amax(tau_dat)*4.5), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.1, color='blue',linewidth=0.5), fontsize=3, rotation = 90, alpha=1.0, color='blue')
		
		#save plot
		plt.subplots_adjust(hspace=0.07)		
		plt.savefig('Results/XCLASSFit_' + str(cores[j]) + '_' + str(number[j]) + '.pdf', format='pdf', bbox_inches='tight')
		plt.close()
		
		
def run_XCLASS_fit_all_fixed(data_directory,working_directory,regions, filenames,cores, number,mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp,std_line,do_error_estimation):
	###create molfits and observation xml files to compute total XCLASS fit spectrum
	
	create_XCLASS_molfits_file(cores, number,mol_name,mol_name_file,method='all')
	create_XCLASS_obsxml_file(data_directory,working_directory,regions, filenames,cores, number,mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp, do_error_estimation, method='all')

	#run XCLASS with fixed parameters
	os.system('casa -c XCLASS_fit_RESULTS_all.py')
	rm_casa_files()
	
	os.system('casa -c XCLASS_optical_depth.py')
	rm_casa_files()
	
	#plot_fit_residuals_optical_depth(cores, number,std_line)
	