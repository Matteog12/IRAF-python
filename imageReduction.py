from astropy.stats import mad_std
from astropy.nddata import CCDData
from itertools import combinations
import ccdproc as ccdp
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import gridspec

home = os.getcwd()

def chooseIndeces(arr, numbers, func, mode):
    comb = list(combinations(range(len(arr)), numbers))
    listValues = np.zeros(len(comb), dtype=float)
    for pos, i in enumerate(comb):
        listValues[pos] = func(arr[list(i)])

    selVal = mode(listValues)
    pos = np.argwhere(listValues == selVal)[0][0]
    selInd = list(comb[pos])

    return selInd

def chooseFiles(names, selInd):
    selNames = []
    for i in selInd:
        selNames.append(names[i])

    return selNames

def inv_median(arr):
    return 1/np.median(arr)

def checkOverscan(dataList, folderPath, keywords, numberOfPixels=50):
    # keywords[0] = lightKeyword
    # keywords[1] = flatKeyword
    # keywords[2] = biasKeyword

    legendName = {
        keywords[0]:'Science image',
        keywords[1]:'Flat image',
        keywords[2]:'Bias image'
    }

    print("Controllo dell'eventuale presenza di overscan in corso...")
    imageArr = []
    for filePath in dataList:
        imageArr.append(CCDData.read(filePath, unit='adu'))

    inst = imageArr[0].meta['INSTRUME']

    gs1 = gridspec.GridSpec(1,2)
    gs2 = gridspec.GridSpec(2,2)

    fig1 = plt.figure(1, figsize=(15,10))
    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs1[1])

    fig2 = plt.figure(2, figsize=(15,10))
    ax11 = plt.subplot(gs2[0, 0])
    ax12 = plt.subplot(gs2[0, 1])
    ax21 = plt.subplot(gs2[1, 0])
    ax22 = plt.subplot(gs2[1, 1])
    ax = [ax1, ax11, ax21, ax2, ax12, ax22]
    titles = [
        'Media lungo tutte le colonne',
        'Media lungo le prime ' + str(numberOfPixels) + ' colonne',
        'Media lungo le ultime ' + str(numberOfPixels) + ' colonne',
        'Media lungo tutte le righe',
        'Media lungo le prime ' + str(numberOfPixels) + ' righe',
        'Media lungo le ultime ' + str(numberOfPixels) + ' righe'
    ]

    if not os.path.isdir(folderPath): os.mkdir(folderPath)

    for image in imageArr:
        meanAx1 = image.data.mean(axis=0) #media lungo le colonne
        meanAx2 = image.data.mean(axis=1) #media lungo le righe

        for n, axes in enumerate(ax):
            if n < 3:
                axes.plot(meanAx1, label=legendName[image.meta['imagetyp']])
            else:
                axes.plot(meanAx2, label=legendName[image.meta['imagetyp']])


    for n, axes in enumerate(ax):
        axes.legend(loc='upper right')
        axes.set_xlabel('pixel')
        axes.set_ylabel('ADU')
        axes.set_title(titles[n])
    
    fig1.suptitle("Analisi dell'andamento dello strumento " + inst, fontsize=20)


    fig2.suptitle('Possibili regioni di overscan con lo strumento ' + inst, fontsize=20)
    ax11.set_xlim(xmax=numberOfPixels)
    ax11.legend(loc='upper left')
    ax21.set_xlim(xmin=len(meanAx1) - numberOfPixels)
    ax12.set_xlim(xmax=numberOfPixels)
    ax12.legend(loc='upper left')
    ax22.set_xlim(xmin=len(meanAx2) - numberOfPixels)

    fig1.savefig(folderPath + os.sep + inst + '_totale.png', dpi=300)
    fig2.savefig(folderPath + os.sep + inst + '_parziale.png', dpi=300)
    plt.close('all')

    print("Procedura completata con successo.")
    print("I file sono stati salvato correttamente nella cartella:\n" + folderPath)
    print()

def biasProcessing(dataList, folderPath, combinedName, trim='[:,:]', OW=True):
    print("Inizio della procedura per la riduzione dei bias:")
    
    if not os.path.isdir(folderPath): os.mkdir(folderPath)

    calibrated_biases = []
    for filePath in dataList:
        bias = CCDData.read(filePath, unit='adu')
        trimmedBias = ccdp.trim_image(bias, fits_section=trim)

        newFilePath = folderPath + os.sep + filePath.split(os.sep)[-1]
        trimmedBias.write(newFilePath, overwrite=OW)
        calibrated_biases.append(newFilePath)

    combined_bias = ccdp.combine(calibrated_biases,
                                 method='average',
                                 sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                 sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                                 mem_limit=350e6, unit='adu'
                                )

    combined_bias.meta['combined'] = True
    combined_bias.write(folderPath + os.sep + combinedName, overwrite=OW)
    print("Procedura completata con successo.")
    print("Il file " + combinedName + " è stato salvato correttamente nella cartella:\n" + folderPath)
    print()

def flatProcessing(dataList, folderPath, combinedName, masterBias, trim='[:,:]', OW=True):
    print("Inizio della procedura per la riduzione dei flat:")
    
    if not os.path.isdir(folderPath): os.mkdir(folderPath)

    calibrated_flats = []
    for filePath in dataList:
        flat = CCDData.read(filePath, unit='adu')
        trimmedFlat = ccdp.trim_image(flat, fits_section=trim)

        trimmedFlat = ccdp.subtract_bias(trimmedFlat, masterBias)

        newFilePath = folderPath + os.sep + filePath.split(os.sep)[-1]
        trimmedFlat.write(newFilePath, overwrite=OW)
        calibrated_flats.append(newFilePath)

    combined_flat = ccdp.combine(calibrated_flats,
                                 method='average', scale=inv_median,
                                 sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                 sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                                 mem_limit=350e6, unit='adu'
                                )

    combined_flat.meta['combined'] = True
    combined_flat.write(folderPath + os.sep + combinedName, overwrite=OW)
    print("Procedura completata con successo.")
    print("Il file " + combinedName + " è stato salvato correttamente nella cartella:\n" + folderPath)
    print()

def darkProcessing(dataList, folderPath, combinedName, masterBias, trim='[:,:]', OW=True):
    print("Inizio della procedura per la riduzione dei dark:")
    
    if not os.path.isdir(folderPath): os.mkdir(folderPath)

    calibrated_darks = []
    for filePath in dataList:
        dark = CCDData.read(filePath, unit='adu')
        trimmedDark = ccdp.trim_image(dark, fits_section=trim)

        trimmedFlat = ccdp.subtract_bias(trimmedDark, masterBias)

        newFilePath = folderPath + os.sep + filePath.split(os.sep)[-1]
        trimmedDark.write(newFilePath, overwrite=OW)
        calibrated_darks.append(newFilePath)

    combined_darks = ccdp.combine(calibrated_darks,
                                 method='average',
                                 sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                 sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                                 mem_limit=350e6, unit='adu'
                                )

    combined_darks.meta['combined'] = True
    combined_darks.write(folderPath + os.sep + combinedName, overwrite=OW)
    print("Procedura completata con successo.")
    print("Il file " + combinedName + " è stato salvato correttamente nella cartella:\n" + folderPath)
    print()

def lightProcessing(dataList, reducedList, folderPath, combinedName, trim='[:,:]', OW=True, 
                    bias=True, dark=True, flat=True):
    print("Inizio della procedura per la riduzione dei light:")

    if not os.path.isdir(folderPath): os.mkdir(folderPath)

    calibrated_lights = []
    for filePath in dataList:
        light = CCDData.read(filePath, unit='adu')
        trimmedLight = ccdp.trim_image(light, fits_section=trim)
        
        biasDone = False
        flatDone = False
        darkDone = False

        for process in reducedList:
            if bias and not biasDone:
                trimmedLight = ccdp.subtract_bias(trimmedLight, process)
                biasDone = True

            elif dark and not darkDone:
                trimmedLight = ccdp.subtract_dark(trimmedLight, process)
                darkDone = True

            elif flat and not flatDone:
                trimmedLight = ccdp.flat_correct(trimmedLight, process)
                flatDone = True

        newFilePath = folderPath + os.sep + filePath.split(os.sep)[-1]
        trimmedLight.write(newFilePath, overwrite=OW)
        calibrated_lights.append(newFilePath)

    if len(calibrated_lights) > 1:
        combined_light = ccdp.combine(calibrated_lights,
                                     method='average',
                                     sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                     sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                                     mem_limit=350e6, unit='adu'
                                    )
    else:
        combined_light = trimmedLight

    combined_light.meta['combined'] = True
    combined_light.write(folderPath + os.sep + combinedName, overwrite=OW)
    print("Procedura completata con successo.")
    print("Il file " + combinedName + " è stato salvato correttamente nella cartella:\n" + folderPath)
    print()