from Functions import functs as fc

### LOADING INPUT AND SETTING UP DIRECTORIES ###
#setup working directory
working_directory = fc.setup_directory(delete_previous_results=True)

fc.determine_noise()

fc.extract_spectrum_init()
fc.setup_XCLASS_files(working_directory)
fc.run_XCLASS_fit()


fc.extract_results(plotting=True)
fc.create_plots()
fc.run_XCLASS_fit_all_fixed(working_directory)
fc.plot_fit()