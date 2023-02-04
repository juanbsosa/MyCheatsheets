# !!! Se copian los archivos de los subdirectorios, pero todos los archivos se copian en la misma carpeta, no se crean los subdirectorios, se deberian crear.


import os
import shutil
data_in = r''
data_out = r''
extension = '.'

for folders, subfolders, filenames in os.walk(data_in):
    for filename in filenames:
        if filename.endswith('{}'.format(extension)):
            # print(filename)
            shutil.copy(os.path.join(folders, filename), data_out)