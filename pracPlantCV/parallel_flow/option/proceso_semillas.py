#!/usr/bin/env python
#Sección 1. Importación de librerías e imagen
import os
import argparse
from plantcv import plantcv as pcv
import matplotlib
import numpy as np
import cv2


### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w", "--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug",
                        help="can be set to 'print' or None (or 'plot' if in jupyter) prints intermediate images.",
                        default=None)
    args = parser.parse_args()
    return args

#### Start of the Main/Customizable portion of the workflow.

### Main workflow
def main():
    # Get options
    args = options()

    pcv.params.debug = args.debug  # set debug mode
    pcv.params.debug_outdir = args.outdir  # set output directory

    #Set the plotting size (default =100)
    pcv.params.dpi = 100

    # Read image plantcv
    img, path, filename = pcv.readimage(filename=args.image)

    #Select file name only
    justname=os.path.splitext(filename)

    #Sección 2. Segmentación e identificación de objetos
    #Conversión de la imagen a escala de grises
    b_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')

    #Se definen umbrales
    b_thresh = pcv.threshold.binary(gray_img=b_img, threshold=139, max_value=150, object_type='light')

    #Eliminación de ruido del fondo, es decir se eliminan los objetos
    #de tamaño pequeño
    b_fill = pcv.fill(bin_img=b_thresh, size=20)

    #Sección 3. Conteo y análisis de semillas

    #Identificación simple de objetos
    objects = cv2.findContours(b_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #Cuenta de semillas
    number_seeds = len(objects[0])

    # Create a new measurement
    pcv.outputs.add_observation(sample='default', variable='nsemillas',
                            trait='Number of seeds with cv2',
                            method='cuenta de semillas con cv2', scale='number', datatype=int,
                            value=number_seeds, label='number_of_seeds')


    #Identificación de semillas para el análisis de forma y color
    objects2, obj_hierarchy = pcv.find_objects(img=img, mask=b_fill)

    #Mediciones en cada semilla
    # Se crea una copia de la imagen RGB para el análisis de forma
    # Entradas:
    #   img = image
    shape_img = np.copy(img)

    # Se elimina la opción de presentar gráficas o imágenes
    pcv.params.debug = None

    # Hace iteraciones sobre todos los objetos en objects2 y se hace el análisis de forma y color
    # for i in range(0, len(objects2)):
    # El bucle anterior consume demasiada memoria, pero idealmente se hace el bucle por cada semilla


    # Con un propósito demostrativo, se hace el bucle solo con los primeros 15 objetos
    for i in range(0, number_seeds):
        # Se verifica si el objeto tiene una rama en la jerarquía
        if obj_hierarchy[0][i][3] == -1:
            # Crea un objeto y una máscara para un objeto (semilla)
            #
            # Entradas:
            #   img - rgb image
            #   contours - list entry i in objects2
            #   hierarchy - np.array of obj_hierarchy[0][1]
            seed, seed_mask = pcv.object_composition(img=img, contours=[objects2[i]], hierarchy=np.array([[obj_hierarchy[0][i]]]))

            # Analiza la forma de cada semilla
            #
            # Entradas:
            #   img - rgb image
            #   obj - seed
            #   mask - mask created of single seed
            #   label - label for each seed in image
            shape_img = pcv.analyze_object(img=shape_img, obj=seed, mask=seed_mask, label=f"seed{i}")

            # Analiza el color de cada semilla
            #
            # Entradas:
            #   img - rgb image
            #   obj - seed
            #   hist_plot_type - 'all', or None for no histogram plot
            #   label - 'default'
            color_img = pcv.analyze_color(rgb_img=img, mask=b_fill, hist_plot_type=None, label="default")

    #Visualización del análisis de forma de las semillas
    #pcv.plot_image(img=shape_img)

    #pcv.plot_image(img=color_img)


    #Envía los resultados a un archivo
    pcv.outputs.save_results(filename=args.result,outformat='csv')


if __name__ == '__main__':
    main()
