import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy import units as u
from matplotlib.patches import Ellipse
from matplotlib import ticker
import functs as fc

data_directory, do_error_estimation, channel1 , channel2 = fc.load_input_table()

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

data_directory, do_error_estimation, channel1 , channel2 = fc.load_input_table()
regions, regions_plot, distances, filenames, filenames_continuum = fc.load_regions_table()
cores, number, x_pix, y_pix, core_label = fc.load_cores_table()

def determine_noise_continuum():
	
	noise_continuum = np.zeros(regions.size)
	
	for i in range(regions.size):
		
		#Open continuum fits file
		hdu = fits.open(data_directory + filenames_continuum[i])[0]
		
		#Extract flux 
		flux = hdu.data[0,:,:]*1000.0 #mJy/beam
		
		#compute standard deviation of continuum image
		noise_continuum[i] = np.std(flux)
		
	return noise_continuum
	
	
def plot_continuum():
	
	#plotting parameters
	ticklength = 7.0 #for 2d maps
	plt.rcParams.update(params)
	
	noise_continuum = determine_noise_continuum()
	
	for i in range(regions.size):
		
		#Open continuum fits file
		hdu = fits.open(data_directory + filenames_continuum[i])[0]
		
		#Extract flux and coordinate axis
		flux = hdu.data[0,:,:]*1000.0 # in Jy/beam
		wcs = WCS(hdu.header, naxis=['longitude', 'latitude'])
		delta = hdu.header['CDELT2'] * 3600.0
		
		#Extract beam properties
		bmin = hdu.header['BMIN'] * 3600.0
		bmaj = hdu.header['BMAJ']* 3600.0
		bpa = hdu.header['BPA']
		
		#compute 3 sigma contour levels
		sigma_cont = noise_continuum[i]
		cont_levels = np.array([3*sigma_cont])
		
		#plot continuum
		fig = plt.figure(1)
		ax = plt.subplot(1, 1, 1, projection=wcs)
		
		#2D continuum image
		im = ax.imshow(flux, origin='lower', interpolation='nearest', cmap='viridis')
		
		#3sigma contours
		ax.contour(flux, colors='white', alpha=1.0, levels=cont_levels,linewidths=0.5)
		
		#choose image x and y range
		if regions[i] != 'S87IRS1':
			cut1 = 384
			cut2 = 640
		else:
			cut1 = 334
			cut2 = 690
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
			
			#error if core_label is not correct	
			else: 
			
				print(core_label_mask[k])
				print('Tag can only be C or E!')
				
			#plot marker at core position	
			ax.plot([x_pix_mask[k]], [y_pix_mask[k]], '^', color=labelcol, markersize=2)
			#annotate core number at core position
			ax.annotate(str(number_mask[k]), xy=(x_pix_mask[k]+1, y_pix_mask[k]+1),xycoords='data', color=labelcol, fontsize=8)
			
			#lon, lat = wcs.all_pix2world(x_pix_mask[k], y_pix_mask[k],0)
				
		#save plot	
		plt.savefig('PLOTS/CONTINUUM/Continuum_color_' + regions[i] + '.pdf', format='pdf', bbox_inches='tight')
		plt.close()
