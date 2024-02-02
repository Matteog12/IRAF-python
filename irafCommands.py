from astropy.nddata import CCDData
from matplotlib import pyplot as plt
from astropy.nddata.blocks import block_reduce
from astropy import visualization as aviz
import numpy as np
import os

home = os.getcwd()

def imstat(dataList, viewValue = True):

    NPIX = np.zeros(len(dataList), dtype=float)
    MEAN = np.zeros(len(dataList), dtype=float)
    STDDEV = np.zeros(len(dataList), dtype=float)
    MIN = np.zeros(len(dataList), dtype=float)
    MAX = np.zeros(len(dataList), dtype=float)

    for i, imDir in enumerate(dataList):
        imData = CCDData.read(imDir, unit='adu').data

        NPIX[i] = len(imData)*len(imData[0])
        MEAN[i] = np.mean(imData)
        STDDEV[i] = np.std(imData)
        MIN[i] = np.min(imData)
        MAX[i] = np.max(imData)

    if viewValue:
        print("\nIMAGE\t\t\tNPIX\tMEAN\tSTDDEV\tMIN\tMAX")

        for i, imDir in enumerate(dataList):
            if MEAN[i] < 1000:
                rMean = round(MEAN[i], 2)
            else:
                rMean = round(MEAN[i])

            if STDDEV[i] < 100:
                rSTD = round(STDDEV[i], 2)
            else:
                rSTD = round(STDDEV[i])


            print(imDir.split(os.sep)[-1] + "\t" + str(int(NPIX[i])) + "\t" + str(rMean) + "\t" + str(rSTD) + "\t" + str(int(MIN[i])) + "\t" + str(int(MAX[i])))
        
        print()

    return NPIX, MEAN, STDDEV, MIN, MAX

def disp(data, cMap, cbar, folderPath, filename, viewValue=False, log=False, z1=None, z2=None):

    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    percu = 99
    percl = 100 - percu

    pxSize = fig.get_size_inches() * fig.dpi

    ratio = (data.shape // pxSize).max()
    ratio = max(ratio, 1)

    reduced_data = block_reduce(data, ratio)
    reduced_data = reduced_data / ratio**2
    extent = [0, data.shape[1], 0, data.shape[0]]

    if log:
        stretch = aviz.LogStretch()
    else:
        stretch = aviz.LinearStretch()

    norm = aviz.ImageNormalize(reduced_data, stretch=stretch,
                               interval=aviz.AsymmetricPercentileInterval(percl, percu),
                               vmin=z1, vmax=z2, clip=True)
    scale_args = dict(norm=norm)

    im = ax.imshow(reduced_data, origin='lower', cmap=cMap, extent=extent, aspect='equal', **scale_args)
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    ax.tick_params(which = 'both', size = 0, labelsize = 0)

    if cbar:
        cbarAx = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbarAx.ax.tick_params(labelsize=20) 
    

    if not os.path.isdir(folderPath): os.mkdir(folderPath)
    plt.savefig(folderPath + os.sep + filename, dpi=300)
    
    if viewValue: plt.show()
    plt.close()
