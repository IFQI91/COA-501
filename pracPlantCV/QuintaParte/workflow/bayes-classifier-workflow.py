#!/usr/bin/env python
import os
import argparse
from plantcv import plantcv as pcv
import numpy as np

# Parse command-line arguments
def options():
	parser = argparse.ArgumentParser(description="Imaging processing with opencv")
	parser.add_argument("-i", "--image", help="Input image file.", required=True)
	parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
	parser.add_argument("-r", "--result", help="result file.", required=False)
	parser.add_argument("-w", "--writeimg", help="write out images.", default=False, action="store_true")
	parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default=None)
	parser.add_argument("-p", "--pdfs", help="Naive Bayes PDF file.", required=True)
	args = parser.parse_args()
	return args


def main():
	# Get options
	args = options()

	# Initialize device counter
	pcv.params.debug = args.debug
	pcv.params.debug_outdir = args.outdir  # set output directory

	# Read in the input image
	img, path, filename = pcv.readimage(filename=args.image)

	# Aplicamos el filtro gaussiano
	gaussian_img = pcv.gaussian_blur(img=img, ksize=(5, 5), sigma_x=1, sigma_y=1)

	# Use the output file from `plantcv-train.py` to run the multiclass 
	# naive bayes classification on the image. The function below will 
	# print out 4 masks (lesson,prelesson, green, background)
	mask = pcv.naive_bayes_classifier(rgb_img=gaussian_img, 
                                  pdf_file=args.pdfs)

	#fill_holes para eliminar valores muy pequeños 
	flor_img=pcv.fill_holes(bin_img=mask['flor'])

	#Eliminar todas las particulas menores a 200
	flor_fill=pcv.fill(bin_img=flor_img, size=200)

	#holes removal
	flor_senescente_img=pcv.fill_holes(bin_img=mask['florsenescente'])

	#Eliminar todas las particulas menores a 200 en flor_senescente
	flor_senescente_fill=pcv.fill(bin_img=flor_senescente_img, size=200)

	#le coloca el procesado a las mascaras de flor y florscenescente
	mask['flor']=flor_fill
	mask['florsenescente']=flor_senescente_fill

	# We can apply each mask to the original image to more accurately 
	# see what got masked
	flor_img = pcv.apply_mask(mask=(mask['flor']), img=img, mask_color='white')
	florsenescente_img = pcv.apply_mask(mask=(mask['florsenescente']), img=img, mask_color='white')
	planta_img = pcv.apply_mask(mask=(mask['hoja']), img=img, mask_color='black')
	background_img = pcv.apply_mask(mask=(mask['fondo']), img=img, mask_color='black')

	# Write image and mask with the same name to the path 
	# specified (creates two folders within the path if they do not exist).
	lesson_maskpath, plant_analysis_images = pcv.output_mask(img=img, mask=mask['flor'], 
                                                        filename='flor.png', mask_only=True)
	predamage_maskpath, pust_analysis_images = pcv.output_mask(img=img, mask=mask['florsenescente'], 
                                                      filename='florsenescente.png', mask_only=True)
	green_maskpath, chlor_analysis_images = pcv.output_mask(img=img, mask=mask['hoja'], 
                                                        filename='hoja.png', mask_only=True)
	bkgrd_maskpath, bkgrd_analysis_images = pcv.output_mask(img=img, mask=mask['fondo'], 
                                                        filename='fondo.png', mask_only=True)


	# To see all of these masks together we can plot them with plant set to green,
	# senescent flower set to gold, and flower set to red.
	classified_img = pcv.visualize.colorize_masks(masks=[mask['flor'], mask['florsenescente'], 
                                                     mask['hoja'], mask['fondo']], 
                                              colors=['red','gold', 'dark green', 'black'])

	#calcular la superficie, porcentaje 
	planta_total = np.count_nonzero(mask['hoja']) + np.count_nonzero(mask['flor']) + np.count_nonzero(mask['florsenescente'])
	flor = np.count_nonzero(mask['flor'])
	flor_senescente= np.count_nonzero(mask['florsenescente'])
                            
	porciento_flor = flor*100 / (flor + planta_total+flor_senescente)
	porciento_flor_senescente = flor_senescente*100 / (flor + planta_total+flor_senescente)                                  
	porciento_planta = planta_total*100 / (img.shape[0]*img.shape[1])

	# Create a new measurement (gets saved to the outputs class) 
	pcv.outputs.clear()
	pcv.outputs.add_observation(sample='default',variable='porciento_flor', trait='porciento de superficie de flor',
                            method='relación de píxeles', scale='porciento', datatype=float,
                            value=porciento_flor, label='porciento_flor')

	# The print_results function will take the measurements stored when running any (or all) of these functions, format, 
	# and print an output text file for data analysis. The Outputs class stores data whenever any of the following functions
	# are ran: analyze_bound_horizontal, analyze_bound_vertical, analyze_color, analyze_nir_intensity, analyze_object, 
	# fluor_fvfm, report_size_marker_area, watershed. If no functions have been run, it will print an empty text file 
	# pcv.outputs.save_results(filename=args.result,outformat='csv')

	outfile = False
	if args.writeimg == True:
		outfile = os.path.join(args.outdir, filename)

	# Print out the colorized figure that got created
	pcv.print_image(classified_img, os.path.join(args.outdir, filename))
	pcv.outputs.save_results(filename=args.result,outformat="csv" )

if __name__ == '__main__':
	main()
