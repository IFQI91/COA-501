usage: plantcv-train.py [-h] {naive_bayes,naive_bayes_multiclass}

Subcommands:
    naive_bayes
        usage: plantcv-train.py naive_bayes [-h] -i IMGDIR -b MASKDIR -o OUTFILE [-p]

        optional arguments:
            -h, --help                       Show this help message and exit
            -i IMGDIR, --imgdir IMGDIR       Input directory containing images.
            -b MASKDIR, --maskdir MASKDIR    Input directory containing black/white masks.
            -o OUTFILE, --outfile OUTFILE    Trained classifier output filename.
            -p, --plots                      Make output plots.

    naive_bayes_multiclass
        usage: plantcv-train.py naive_bayes_multiclass [-h] -f FILE -o OUTFILE [-p]

        optional arguments:
            -h, --help                       Show this help message and exit
            -f FILE, --file FILE             Input file containing a table of pixel RGB values sampled for each input
                                             class.
            -o OUTFILE, --outfile OUTFILE    Trained classifier output filename.
            -p, --plots                      Make output plots.


# Set global debug behavior to None (default), "print" (to file),
# or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = "print"

# Read in a color image
img, path, filename = pcv.readimage("color_image.png")

# Classify the pixels as plant or background
masks = pcv.naive_bayes_classifier(img, pdf_file="naive_bayes_pdfs.txt")
