import numpy as np
import os

#get working directory 
working_directory = os.getcwd() + '/'

#### input table of cores
cores_tab=np.loadtxt(str(working_directory) + 'Input/cores.dat', dtype='U', comments='#')
cores = cores_tab[:,0].astype(np.str)
numbers = cores_tab[:,1].astype(np.int)


#loop over all cores
for j in range(cores.size):
	
	#XCLASS setup files
	MolfitsFileName = str(working_directory) + 'FITS/molecules_' + str(cores[j]) + '_' + str(numbers[j]) + '_compl.molfit'
	experimentalData = str(working_directory) + 'FITS/observation_' + str(cores[j]) + '_' + str(numbers[j]) + '_all.xml'
	AlgorithmXMLFile = str(working_directory) + 'Input/algorithm.xml'
	
	#perform XCLASS Fit on line			
	newmolfit, modeldata, JobDir = myXCLASSFit() 
	
	#copy results from XCLASS default directory to working directory
	os.system('scp -rp ' + JobDir + 'spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_vLSRcorr.LM__call_1.out.dat ' + str(working_directory) + 'FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(numbers[j]) + '_compl.out.dat')
	
exit