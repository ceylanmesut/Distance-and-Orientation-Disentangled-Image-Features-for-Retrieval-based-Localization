from fnmatch import fnmatch
import os
import re
import numpy as np


def get_recursive_file_list(root, pattern):
    print(root)
    print(pattern)
    matched_files = []
    for path, subdirs, files in os.walk(root):
        for file in files:
            full = os.path.join(path, file)
            if fnmatch(full, pattern):
                matched_files.append(full)
    print(len(matched_files))
    return matched_files


def get_file_list(path, pattern):
    print(path)
    print(pattern)
    matched_files = []
    for file in os.listdir(path):
        full = os.path.join(path, file)
        if fnmatch(full, pattern):
                matched_files.append(full)
    print(len(matched_files))
    return matched_files


def get_t_x_y_a_from_path(path):
    name = os.path.basename(path)

    t = re.findall(r'(?<=t)-?\d+[.]?\d*', name)
    x = re.findall(r'(?<=x)-?\d+[.]?\d*', name)
    y = re.findall(r'(?<=y)-?\d+[.]?\d*', name)
    a = re.findall(r'(?<=a)-?\d+[.]?\d*', name)

    if not (len(t)== 1 and len(x)==1 and len(y)==1 and len(a) ==1):
        return False, 0, 0, 0, 0
    else:
        return True, float(t[0]), float(x[0]), float(y[0]), float(a[0])
    
def get_weather_name(path):
    name = path
    
    w_fullname = name.split('/')[-3]
    w_name = w_fullname[5:-1]
    
    return w_name
    


def parse_file_list(FILES):
    valid_files = []
    TXYA = []
    w_names = []
    for i in range(len(FILES)):
        is_valid, t, x, y, a = get_t_x_y_a_from_path(FILES[i])
        w_name = get_weather_name(FILES[i])
        if is_valid:
            valid_files.append(FILES[i])
            TXYA.append([t, x, y, a])
            w_names.append(w_name)
    return valid_files, np.array(TXYA), w_names

