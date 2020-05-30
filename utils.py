import os
import uuid
import sqlite3
from PIL import Image
import numpy as np
import base64
import psutil
import json

from config import (
    PERSON_DATA_PATH, 
    FACE_IMAGE_PATH
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def verify_data(data, *keys):
    '''
    data: json data in form of dictionary.
    *keys: keys to be checked in data.
    Returns: True if all keys are found in data else False.
    '''
    data_keys = list(data.keys())
    for key in keys:
        if key not in data_keys:
            return False
    return True

def read_image(img_path):
    '''
    img_path: path of image to be read.
    Returns: a numpy array of shape HxWx3 and data type uint8.
    '''
    img = np.array(Image.open(img_path))
    img = img.astype(np.uint8)
    return img


def encode_to_base64(file_path):
    '''
    file_path: path of file whose contents are to be encoded.
    Returns: base64 encoded file data that has been decoded as UTF-8 string.
    '''
    with open(file_path, "rb") as f:
        data = f.read()
    data = base64.b64encode(data)
    return data.decode("UTF-8")


def decode_from_base64(data, file_path):
    '''
    data: base64 encoded data that has been decoded as UTF-8 string.
    file_path: path of file to which the decoded data is to be written.
    Returns: True and writes decoded data to specified file.
    '''
    data = data.encode("UTF-8")
    data = base64.b64decode(data)
    with open(file_path, "wb") as f:
        f.write(data)
    return True


def verify_file_extension(filename):
    '''
    filename: str.
    Returns: True if extension of file is in allowed extensions else False.
    '''
    splt = filename.split(".")
    if len(splt) > 1 and splt[-1] in ALLOWED_EXTENSIONS:
        return True
    return False

def get_unique_id():
    '''
    Returns: a unique string.
    '''
    return str(uuid.uuid4())


def get_system_info():
    '''
    Returns a dictionary containing system info.
    '''
    cpu_percent = psutil.cpu_percent()
    cpu_freq = psutil.cpu_freq().current
    cpu_count = psutil.cpu_count()
    _temp = psutil.virtual_memory()
    total_memory = _temp.total/(1024*1024)
    available_memory = _temp.available/(1024*1024)
    _temp = psutil.swap_memory()
    total_swap = _temp.total/(1024*1024)
    available_swap = _temp.free/(1024*1024)
    _temp = psutil.disk_usage("/")
    total_disk = _temp.total/(1024*1024*1024)
    available_disk = _temp.free/(1024*1024*1024)
    info = {
        "cpu_percent": cpu_percent, 
        "cpu_frequency": cpu_freq, 
        "cpu_count": cpu_count, 
        "total_memory": total_memory, 
        "available_memory": available_memory, 
        "total_swap": total_swap, 
        "available_swap": available_swap, 
        "total_disk": total_disk, 
        "available_disk": available_disk
    }
    return info


def get_person_data(id):
    '''
    id: id of person to get data for.
    Returns a dictionary of person data.
    '''
    data_dir = os.path.join(PERSON_DATA_PATH, id+".json")
    with open(data_dir, "r") as f:
        d_json = json.load(f)
    del d_json["face_encoding"]
    return d_json
