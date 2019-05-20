import functs as fc

data_directory, do_error_estimation, channel1 , channel2 = fc.load_input_table()
regions, regions_plot, distances, filenames = fc.load_regions_table()
cores, number, x_pix, y_pix, core_label = fc.load_cores_table()
mol_name, mol_name_file = fc.load_molecules_table()
mol_ranges_name, mol_ranges_low, mol_ranges_upp = fc.load_molecule_ranges_table()
working_directory = fc.setup_directory(delete_previous_results=False)

std_line = fc.determine_noise(data_directory, regions, filenames, cores, number, x_pix, y_pix,channel1,channel2)

#extract spectra
fc.extract_spectrum_init(data_directory, regions, filenames, cores, number, x_pix, y_pix)
		
#get velocity offset from C18O line fit
v_off = fc.get_velocity_offset(cores, number)
		
		
fc.extract_spectrum(data_directory, regions, filenames, cores, number, x_pix, y_pix, v_off,do_error_estimation)