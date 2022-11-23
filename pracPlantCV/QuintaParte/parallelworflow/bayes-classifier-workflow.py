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




    outfile = False
    if args.writeimg == True:
        outfile = os.path.join(args.outdir, filename)

    # Print out the colorized figure that got created
    pcv.print_image(classified_img, os.path.join(args.outdir, filename))
    pcv.outputs.save_results(filename=args.result,outformat="csv" )

if __name__ == '__main__':
    main()
