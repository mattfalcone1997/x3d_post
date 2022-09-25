import os
from tabnanny import check

def check_path(path,statistics=False,data=False):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path {path} does not exist")
    if 'parameters.json' not in os.listdir(path):
        raise FileNotFoundError(f"parameters.json file not found in {path}")
    
    stat_path = os.path.join(path,'statistics')

    if statistics and not os.path.isdir(stat_path):
        raise FileNotFoundError(f"Path {stat_path} does not exist")

    data_path = os.path.join(path,'data')
    if data and not os.path.isdir(data_path):
        raise FileNotFoundError(f"Path {data_path} does not exist")

def max_iteration(path):
    check_path(path,statistics=True)

    stat_path = os.path.join(path,'statistics')

    files = os.listdir(stat_path)
    its = sorted(set([int(x[-7:]) for x in files]))
    return its[-1]