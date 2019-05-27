import numpy as np
import os

#get working directory 
working_directory = os.getcwd() + "/"

#### input table of cores
cores_tab=np.loadtxt('cores.dat', dtype='U', comments='#')
cores = cores_tab[:,0].astype(np.str)
number = cores_tab[:,1].astype(np.int)


#loop over all cores
for j in range(cores.size):
	
	#XCLASS setup files
	MolfitsFileName = str(working_directory) + "FITS/molecules_vLSR.molfit"
	experimentalData = str(working_directory) + "FITS/observation_" + str(cores[j]) + "_" + str(number[j]) + "_vLSR.xml"
	AlgorithmXMLFile = str(working_directory) + "algorithm.xml"
	
	#perform XCLASS Fit on C18O line	
	newmolfit, modeldata, JobDir = myXCLASSFit() 
	
	#copy results from XCLASS default directory to working directory
	os.system('scp -rp ' + JobDir + 'spectrum_' + str(cores[j]) + '_' + str(number[j]) + '.LM__call_1.out.dat ' + str(working_directory) + 'FITS/BESTFIT_spectrum_' + str(cores[j]) + '_' + str(number[j]) + '_vLSR.out.dat')
	os.system('scp -rp ' + JobDir + 'molecules_vLSR__LM__call_1.out.molfit ' + str(working_directory) + 'FITS/BESTFIT_molecules_' + str(cores[j]) + '_' + str(number[j]) + '_vLSR.molfit')           