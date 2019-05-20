import cont_functs as contfc
import functs as fc

#load input tables
data_directory, do_error_estimation, channel1 , channel2 = fc.load_input_table()
regions, regions_plot, distances, filenames, filenames_continuum = fc.load_regions_table()
cores, number, x_pix, y_pix, core_label = fc.load_cores_table()
mol_name, mol_name_file = fc.load_molecules_table()
mol_ranges_name, mol_ranges_low, mol_ranges_upp = fc.load_molecule_ranges_table()

#setup working directory
working_directory = fc.setup_directory(delete_previous_results=False)

#continuum plots
contfc.plot_continuum()

#determine noise in spectra
std_line = fc.determine_noise(data_directory, regions, filenames, cores, number, x_pix, y_pix,channel1,channel2)

#extract spectra
fc.extract_spectrum_init(data_directory, regions, filenames, cores, number, x_pix, y_pix)
	
#create XCLASS input files
fc.setup_XCLASS_files(data_directory, working_directory, regions, filenames, distances, cores, number, x_pix, y_pix, mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp, do_error_estimation)

#run XCLASS Fit
fc.run_XCLASS_fit(data_directory, regions, filenames,cores, number, x_pix, y_pix,std_line,do_error_estimation,C18O_vlsr=True)

#extract fit results and plot results
fc.create_plots(cores, number, mol_name_file,std_line,do_error_estimation)

#fc.run_XCLASS_fit_all_fixed(data_directory,working_directory,regions, filenames,cores, number,mol_name,mol_name_file,mol_ranges_name, mol_ranges_low, mol_ranges_upp,std_line,do_error_estimation)


