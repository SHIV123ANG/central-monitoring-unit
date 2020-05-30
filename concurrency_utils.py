import os
import math
from multiprocessing import Pool, Manager, TimeoutError

from utils import get_unique_id

pool = Pool()
manager = Manager()
mp_progress_dict = manager.dict()
mp_result_dict = dict()

def get_progress(token):
    '''
    token: unique process token generated when creating process.
    Returns: float value representing the progress of process in [0, 1] 
            or None if token is not valid.
    '''
    try:
        return mp_progress_dict[token]
    except KeyError:
        return None

def get_result(token):
    '''
    token: unique process token generated when creating process.
    Returns: Result of process if process is completed else False 
            or None if token is not valid.
    '''
    try:
        return mp_result_dict[token].get(timeout=0.1)
    except TimeoutError:
        return False
    except KeyError:
        return None

def delete_task(token):
    '''
    token: unique process token generated when creating process.
    Returns: True if task is deleted else False.
    '''
    try:
        del mp_progress_dict[token]
        del mp_result_dict[token]
        return True
    except KeyError:
        return False

def get_total_tasks():
    '''
    Returns: A tuple of (total_tasks, completed_tasks)
    '''
    total_tasks = len(mp_progress_dict)
    completed_tasks = 0
    for k, v in mp_progress_dict.items():
        if v == 1.0:
            completed_tasks += 1
    return (total_tasks, completed_tasks)
