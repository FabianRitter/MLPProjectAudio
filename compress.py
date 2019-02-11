import os
import zipfile
path_to_zip_file = 'FSDnoisy18k.meta.zip'
directory_to_extract_to = 'audio'
zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
zip_ref.extractall(directory_to_extract_to)
zip_ref.close()