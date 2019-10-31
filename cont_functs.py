import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import ticker
import functs as fc

params = {'font.family' : 'serif',
			 'font.size' : 12,
			 'errorbar.capsize' : 3,
			 'lines.linewidth'   : 1.0,
			 'xtick.top' : True,
			 'ytick.right' : True,
			 'legend.fancybox' : False,
			 'xtick.major.size' : 7.0 ,
          'xtick.minor.size' : 4.0,    
          'ytick.major.size' : 7.0 ,
          'ytick.minor.size' : 4.0,  
          'xtick.direction' : 'in',
			 'ytick.direction' : 'in',
			 'xtick.color' : 'white',
			 'ytick.color' : 'white',
          'mathtext.rm' : 'serif',
          'mathtext.default': 'regular', 
			}

regions, regions_plot, distances, filenames, filenames_continuum, region_luminosity, region_mass = fc.load_regions_table()
cores, number, x_pix, y_pix, core_label = fc.load_cores_table()


def determine_noise_continuum(data_directory):
	
	noise_continuum = np.zeros(regions.size)
	
	for i in range(regions.size):
		
		#Open continuum fits file
		hdu = fits.open(data_directory + filenames_continuum[i])[0]
		
		#Extract flux 
		flux = hdu.data[0,:,:]*1000.0 #mJy/beam
		
		#compute standard deviation of continuum image
		noise_continuum[i] = np.std(flux[400:450,400:450])

	return noise_continuum
	
	
def check_flux(data_directory):
	
	noise_continuum = determine_noise_continuum(data_directory)
	
	for i in range(regions.size):
		
		#Open continuum fits file
		hdu = fits.open(data_directory + filenames_continuum[i])[0]
		
		#Extract flux and coordinate axis
		flux = hdu.data[0,:,:]*1000.0 # in mJy/beam
	
		#choose all selected positions in this region
		mask = np.where(cores == regions[i])
		cores_mask = cores[mask]
		number_mask = number[mask]
		x_pix_mask = x_pix[mask]
		y_pix_mask = y_pix[mask]
		core_label_mask = core_label[mask]
		
		##raise error if core_label is not correct	
		for k in range(cores_mask.size):
			
			if core_label_mask[k] != 'C': 
				if core_label_mask[k] != 'E':
			
					print(cores_mask[k])
					print(number_mask[k])
					print(core_label_mask[k])
					print('Tag can only be C or E!')
					
		#raise error if numbering is not by descending flux
		for k in range(cores_mask.size-1):
			
			if (flux[y_pix_mask[k+1],x_pix_mask[k+1]] - flux[y_pix_mask[k],x_pix_mask[k]]) > 0.0:
				print(cores_mask[k])
				print(number_mask[k])
				print('Numbering should be by descending continuum flux! The next entry has a higher flux.')
				
		#raise error if numbering is not continous
		for k in range(cores_mask.size):
			
			if number_mask[k] != k+1:
				print(cores_mask[k])
				print(number_mask[k])
				print('Numbering is not in order (1, 2, 3, ...)')
				
		#raise error if position is below 5 sigma threshold
		for k in range(cores_mask.size):
			
			if flux[y_pix_mask[k],x_pix_mask[k]] <= 5.0 * noise_continuum[i]:
				print(cores_mask[k])
				print(number_mask[k])
				print('Position is below threshold (5sigma).')
				
		for k in range(cores_mask.size):
			
			print('Continuum flux: ' + np.str(cores_mask[k]) + ' ' + np.str(number_mask[k]) + ' ' + np.str(flux[y_pix_mask[k],x_pix_mask[k]]) + ' mJy/beam')
				
				
def plot_continuum(data_directory):
	
	#plotting parameters
	ticklength = 7.0 #for 2d maps
	plt.rcParams.update(params)
	
	noise_continuum = determine_noise_continuum(data_directory)
	
	check_flux(data_directory)
	
	for i in range(regions.size):
		
		#Open continuum fits file
		hdu = fits.open(data_directory + filenames_continuum[i])[0]
		
		#Extract flux and coordinate axis
		flux = hdu.data[0,:,:]*1000.0 # in mJy/beam
		wcs = WCS(hdu.header, naxis=['longitude', 'latitude'])
		delta = hdu.header['CDELT2'] * 3600.0 #arcsec
		
		#Extract beam properties
		bmin = hdu.header['BMIN'] * 3600.0 #arcsec
		bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
		bpa = hdu.header['BPA'] #degree
		
		#compute -5,5 and 10 sigma contour levels
		sigma_cont = noise_continuum[i]
		cont_levels = np.array([-5*sigma_cont,5*sigma_cont,10*sigma_cont])
		
		#plot continuum
		fig = plt.figure(1)
		ax = plt.subplot(1, 1, 1, projection=wcs)
		
		#2D continuum image
		im = ax.imshow(flux, origin='lower', interpolation='nearest', cmap='viridis')
		
		#3sigma contours
		ax.contour(flux, colors='white', alpha=1.0, levels=cont_levels,linewidths=0.5)
		
		#choose image x and y range
		#cut1 = 362
		#cut2 = 662
		cut1 = 256
		cut2 = 768
		
		plt.xlim(cut1,cut2)
		plt.ylim(cut1,cut2)
		
		#plot properties
		RA = ax.coords[0]
		DEC = ax.coords[1]
		RA.set_axislabel('Right Ascension (J2000)')
		DEC.set_major_formatter('dd:mm:ss')
		RA.set_major_formatter('hh:mm:ss.s')
		DEC.set_axislabel('Declination (J2000)')
		DEC.set_ticklabel(rotation=90., color='black', exclude_overlapping=True)
		RA.set_ticklabel(color='black', exclude_overlapping=True)
		DEC.set_ticks(number=3, size=ticklength)
		RA.set_ticks(number=3, size=ticklength)
		
		#add colorbar
		cb=fig.colorbar(im, pad=0.0, shrink=1.0) #, format='%.1e'
		cb.set_label('Flux Density [mJy/beam]')
		tick_locator = ticker.MaxNLocator(nbins=6)
		cb.locator = tick_locator
		cb.update_ticks()
		cb.ax.yaxis.set_tick_params(labelcolor='black')
		
		#add beam
		el = Ellipse((cut1+10, cut1+10), width=bmaj/delta, height=bmin/delta, angle=-90.0+bpa, alpha=0.7, edgecolor='black', facecolor='grey')
		ax.add_artist(el)
		
		#add region label
		ax.text(cut1+((cut2-cut1)/2), cut1+(cut2-cut1)-26, regions_plot[i], color='white')
		
		#choose all selected positions in this region
		mask = np.where(cores == regions[i])
		cores_mask = cores[mask]
		number_mask = number[mask]
		x_pix_mask = x_pix[mask]
		y_pix_mask = y_pix[mask]
		core_label_mask = core_label[mask]
		
		#add positions in plot with two different colors depending on chosen core_label
		for k in range(cores_mask.size):
			
			#core positons
			if core_label_mask[k] == 'C':
				
				labelcol = 'red'
			
			#envelope positions	
			elif core_label_mask[k] == 'E':
				
				labelcol = 'orange'
				
			#plot marker at core position	
			ax.plot([x_pix_mask[k]], [y_pix_mask[k]], '^', color=labelcol, markersize=2)
			#annotate core number at core position
			ax.annotate(str(number_mask[k]), xy=(x_pix_mask[k]+1, y_pix_mask[k]+1),xycoords='data', color=labelcol, fontsize=8)

		#save plot	
		plt.savefig('PLOTS/CONTINUUM/Continuum_color_' + regions[i] + '.pdf', format='pdf', bbox_inches='tight')
		#plt.show()
		plt.close()
		
