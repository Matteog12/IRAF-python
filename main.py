from astropy.nddata import CCDData
from ccdproc import ImageFileCollection
import numpy as np
import os

import irafCommands as iraf
import imageReduction as reduction

########PARAMETRI########
blueFilter = 3
redFilter = 5

lightKeyword = 'object'
darkKeyword = 'dark'
flatKeyword = 'flat'
biasKeyword = 'zero'
#########################

#################CARTELLE###################
home = os.getcwd()
imagePath = home + os.sep + 'ser'
irafPath = home + os.sep + 'iraf_res'
graphPath = home + os.sep + 'graph'
darkPath = home + os.sep + 'reduced_darks'
flatPath = home + os.sep + 'reduced_flats'
biasPath = home + os.sep + 'reduced_biases'
lightPath = home + os.sep + 'reduced_lights'
############################################

irafCompare = ImageFileCollection(irafPath)
irafRes = irafCompare.files_filtered(imagetyp='*', include_path=True)


rawImages = ImageFileCollection(imagePath)

bLight = rawImages.files_filtered(imagetyp=lightKeyword, filters=blueFilter, include_path=True)
bDark = rawImages.files_filtered(imagetyp=darkKeyword, filters=blueFilter, include_path=True)
bFlat = rawImages.files_filtered(imagetyp=flatKeyword, filters=blueFilter, include_path=True)
bBias = rawImages.files_filtered(imagetyp=biasKeyword, filters=blueFilter, include_path=True)

rLight = rawImages.files_filtered(imagetyp=lightKeyword, filters=redFilter, include_path=True)
rDark = rawImages.files_filtered(imagetyp=darkKeyword, filters=redFilter, include_path=True)
rFlat = rawImages.files_filtered(imagetyp=flatKeyword, filters=redFilter, include_path=True)
rBias = rawImages.files_filtered(imagetyp=biasKeyword, filters=redFilter, include_path=True)

sampleArr = [bLight[0], bFlat[0], bBias[0]]
reduction.checkOverscan(sampleArr, graphPath, [lightKeyword, flatKeyword, biasKeyword])
trimRange = '[:, 2:1300]'

def reductionProcess(filterBand, nChooseBias, nChooseFlat, biasImageCollection, flatImageCollection, lightImageCollection, viewOpt, trim='[:,:]'):
    combinedNamePrefix = 'combined_'
    if trim != '[:,:]': combinedNamePrefix += 'trimmed_'
    if type(viewOpt) == type(True): viewOpt = [viewOpt]*3

    #BIAS REDUCTION
    print("I bias disponibili per il filtro " + filterBand + " hanno le seguenti statistiche:")
    NPIX, MEAN, STDDEV, MIN, MAX = iraf.imstat(biasImageCollection)
    if nChooseBias < len(STDDEV):
        indeces = reduction.chooseIndeces(STDDEV, nChooseBias, np.std, np.min)
        biasImageCollection = reduction.chooseFiles(biasImageCollection, indeces)
        print("Sono stati selezionati", str(nChooseBias) + "/" + str(len(STDDEV)), "file disponibili.")
        print("Ecco le statistiche dei file selezionati:")
        iraf.imstat(biasImageCollection)

    biasName = combinedNamePrefix + filterBand + '_bias'
    reduction.biasProcessing(biasImageCollection, biasPath, biasName + '.fit')
    masterBias = CCDData.read(biasPath + os.sep + biasName + '.fit', unit='adu')
    if viewOpt[0]: print("Questo il master bias ridotto e combinato:")
    iraf.disp(masterBias.data, cMap = 'gray', folderPath=biasPath, filename=biasName + '.png', viewValue=viewOpt[0], cbar=True)
    print("Il file " + biasName + '.png' + " è stato salvato correttamente nella cartella:\n" + biasPath)

    #FLAT REDUCTION
    print("I flat disponibili per il filtro desiderato hanno le seguenti statistiche:")
    NPIX, MEAN, STDDEV, MIN, MAX = iraf.imstat(flatImageCollection)
    if nChooseFlat < len(STDDEV):
        indeces = reduction.chooseIndeces(STDDEV, nChooseFlat, np.std, np.min)
        flatImageCollection = reduction.chooseFiles(flatImageCollection, indeces)
        print("Sono stati selezionati", str(nChooseFlat) + "/" + str(len(STDDEV)), "file disponibili.")
        print("Ecco le statistiche dei file selezionati:")
        iraf.imstat(flatImageCollection)

    flatName = combinedNamePrefix + filterBand + '_flat'
    reduction.flatProcessing(flatImageCollection, flatPath, flatName + '.fit', masterBias)
    masterFlat = CCDData.read(flatPath + os.sep + flatName + '.fit', unit='adu')
    if viewOpt[1]: print("Questo il master flat ridotto e combinato:")
    iraf.disp(masterFlat.data, cMap = 'gray', folderPath=flatPath, filename=flatName + '.png', viewValue=viewOpt[1], cbar=True)
    print("Il file " + flatName + '.png' + " è stato salvato correttamente nella cartella:\n" + lightPath)


    #LIGHT REDUCTION
    lightName = combinedNamePrefix + filterBand + '_light'
    reduced = [masterBias, masterFlat] #il primo deve essere il bias, seguito e dal dark e dal flat
    reduction.lightProcessing(lightImageCollection, reduced, lightPath, lightName + '.fit', dark=False)
    image = CCDData.read(lightPath + os.sep + lightName + '.fit', unit='adu')
    if viewOpt[2]: print("Questo il light ridotto e combinato:")
    iraf.disp(image.data, cMap = 'gray', folderPath=lightPath, filename= lightName + '.png', viewValue=viewOpt[2], cbar=True)
    print("Il file " + lightName + '.png' + " è stato salvato correttamente nella cartella:\n" + lightPath)


reductionProcess('B', 7, 3, bBias, bFlat, bLight, viewOpt=False)
print("\nRiduzione con il taglio dell'immagine per il filtro Blu:")
reductionProcess('B', 7, 3, bBias, bFlat, bLight, viewOpt=False, trim=trimRange)
print("\nRiduzione con il taglio dell'immagine per il filtro Rosso:")
reductionProcess('R', 7, 4, bBias, rFlat, rLight, viewOpt=False, trim=trimRange)

print("Generazione delle immagini di confronto in corso...")
comparison = [bBias[0], bFlat[0], bLight[0], rLight[0]]
for i, path in enumerate(comparison):
    name = path.split(os.sep)[-1][:-5]
    image = CCDData.read(path, unit='adu').data
    iraf.disp(image.data, cMap = 'gray', folderPath=lightPath, filename= name + '.png', viewValue=False, cbar=True)

for path in irafRes:
    name = path.split(os.sep)[-1][:-5]
    image = CCDData.read(path, unit='adu').data
    iraf.disp(image, cMap='gray', folderPath=irafPath, filename= name + '.png', viewValue=False, cbar=True)
print("Operazione completata con successo.")