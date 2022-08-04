from functions import *

# To test the results of some functions
nuclei, nps, nrows, ncols = stackloader(
    "1.2%_1Hour_A-Tubulin.lif - Series001.tif",
    dir_in="tifs_unanalysed/1.2%_1Hour_A-Tubulin/tifs/",
    plot=True,
)
plotpic(
    "1.2%_1Hour_A-Tubulin.lif - Series001.tif",
    dir_in="tifs_unanalysed/1.2%_1Hour_A-Tubulin/tifs/",
    plot=True,
)
filefolder(dirname="tifs_unanalysed/", plot=True)
improve_reso(4)
