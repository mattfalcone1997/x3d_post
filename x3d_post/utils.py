import os

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

def get_iterations(path,statistics=False):
    data = ~statistics
    check_path(path,statistics=statistics,data=data)
    
    if statistics:
        stat_path = os.path.join(path,'statistics')
        files = os.listdir(stat_path)
        return sorted(set([int(x[-7:]) for x in files]))
    else:
        stat_path = os.path.join(path,'data')
        files = [f for f in os.listdir(stat_path) if 'snapshot' in f]
        return sorted(set([int(x[-12:-5]) for x in files]))
    
def max_iteration(path,statistics=False):
    return get_iterations(path,statistics=statistics)[-1]