import numpy as np
import os

#get working directory 
working_directory = os.getcwd() + '/'

#### input table of cores
cores_tab=np.loadtxt('cores.dat', dtype='U', comments='#')
cores = cores_tab[:,0].astype(np.str)
numbers = cores_tab[:,1].astype(np.int)

#input table of molecules
mol_data=np.loadtxt('molecules.dat', dtype='U', comments='%')
mol_names_file=mol_data[:,1] #plotted molecule label

input_tab=np.loadtxt('init.dat', dtype='U', comments='#').astype(np.str)
do_error_estimation = input_tab[1,1]

if do_error_estimation == 'yes':
	tag = np.array(['','_lowErr','_uppErr'])
elif do_error_estimation == 'no':
	tag = np.array([''])
else:
	print 'Only yes and no are allowed for do_error_estimation parameter in input.dat!'

#loop over all three spectra (data product flux, lower, and upper flux estimates), molecules (k) and cores (j)
for z in range(tag.size):
	
	for k in range(mol_names_file.size):
		
		for j in range(cores.size):
	
			#XCLASS setup files
			MolfitsFileName = str(working_directory) + 'FITS/molecules_' + str(mol_names_file[k]) + '.molfit'
			experimentalData = str(working_directory) + 'FITS/observation_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + str(tag[z]) + '.xml'
			AlgorithmXMLFile = str(working_directory) + 'algorithm.xml'
			
			#perform XCLASS Fit on line			
			newmolfit, modeldata, JobDir = myXCLASSFit() 
			
			#copy results from XCLASS default directory to working directory
			os.system('scp -rp ' + JobDir + 'spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + str(tag[z]) + '_vLSRcorr.LM__call_1.out.dat ' + str(working_directory) + 'FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + str(tag[z]) + '.out.dat')
			os.system('scp -rp ' + JobDir + 'molecules_' + str(mol_names_file[k]) + '__LM__call_1.out.molfit ' + str(working_directory) + 'FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(numbers[j]) + '_' + str(mol_names_file[k]) + str(tag[z]) + '.molfit')           

exit