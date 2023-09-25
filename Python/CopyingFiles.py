# !!! Se copian los archivos de los subdirectorios, pero todos los archivos se copian en la misma carpeta, no se crean los subdirectorios, se deberian crear.

import os
import shutil

# data_in = r'E:\Backup\Music\iTunes\iTunes Media\Music'
# data_out = r'C:\Users\Usuario\Music\iTunes\iTunes Media\Automatically Add to iTunes'
# extension = '.mp3'

def copy_files(data_in, data_out, extension, print_message=False):

    for folders, subfolders, filenames in os.walk(data_in):
        for filename in filenames:
            if filename.endswith('{}'.format(extension)):
                if print_message==True:
                    print("Copying " + filename "...")
                shutil.copy(os.path.join(folders, filename), data_out)


if __name__ == '__main__':

    data_in = input("Enter input directory: ")
    data_out = input("Enter output directory: ")
    extension = input("Enter extension: ")
    print_message = input("Print progress message in console (True/False): ")

    copy_files(data_in=data_in, data_out=data_out, extension=extension)