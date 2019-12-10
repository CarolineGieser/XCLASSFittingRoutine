from Functions import functs as fc

working_directory = fc.setup_directory(delete_previous_results=True)

fc.setup_files(working_directory)

fc.run_fit()

fc.analysis(working_directory)