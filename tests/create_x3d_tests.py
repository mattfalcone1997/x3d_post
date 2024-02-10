import subprocess
import os
from shutil import copytree, which, rmtree
import json
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)

logger.addHandler(ch)

logger.debug("Running with log level debug")

def get_x3d(git_url, branch):

    rmtree('Incompact3d',ignore_errors=True)
    cmd = ['git','clone','-b',branch,git_url]
    subprocess.run(cmd)

    return os.path.join(os.getcwd(),'Incompact3d')

def build_x3d():
    os.chdir('Incompact3d')
    os.mkdir('build')
    os.chdir('build')

    # use cmake to build incompact3d

    cmd = ['cmake','-B','.', '-S','../']
    subprocess.run(cmd)

    np = str(len(os.sched_getaffinity(0)) // 2)
    cmd = ['make',f'-j{np}']
    subprocess.run(cmd)

    return os.path.join(os.getcwd(),'bin')

def run_channel_test(x3d_bin,root,directory):
    rmtree(directory,ignore_errors=True)
    copytree(os.path.join(root,'examples','Channel-Flow'),directory)
    
    # modifying test to create short run time
    os.chdir(directory)
    with open('input.i3d','r') as f:
        lines = f.readlines()
    
        for i, line in enumerate(lines):
            if 'ilast' in line:
                logger.debug("ilast replaced")
                lines[i] = line.replace('100000','25')

            if 'ioutput' in line:
                logger.debug("ioutput replaced")
                lines[i] = line.replace('10000','5')

            if 'icheckpoint' in line:
                logger.debug('icheckpoint replaced')
                lines[i] = line.replace('5000','5')
    
            if 'initstat' in line:
                logger.debug('initstat replaced')
                lines[i] = line.replace('40000','0')     
    
    with open('input.i3d','w') as f:
        f.writelines(lines)
   
    logger.debug("Finished updating lines")
    
    if which('mpirun'):
        np = str(len(os.sched_getaffinity(0)) // 2)
        cmd = ['mpirun','-n',np,x3d_bin]
        subprocess.run(cmd)
    else:
        raise RuntimeError('mpirun command is required '
                            'to run this test')
    os.chdir('..')

def run_blayer_test(x3d_bin,root,directory):
    rmtree(directory,ignore_errors=True)
    copytree(os.path.join(root,'examples','TBL-recycle'),directory)
    
    # modifying test to create short run time
    os.chdir(directory)
    with open('input.i3d','r') as f:
        lines = f.readlines()
    
        for i, line in enumerate(lines):
            if 'ilast' in line:
                logger.debug("ilast replaced")
                lines[i] = line.replace('100000','25')

            if 'ioutput' in line:
                logger.debug("ioutput replaced")
                lines[i] = line.replace('5000','5')

            if 'icheckpoint' in line:
                logger.debug('icheckpoint replaced')
                lines[i] = line.replace('5000','5')
    
            if 'initstat' in line:
                logger.debug('initstat replaced')
                lines[i] = line.replace('50000','0')     
    
    with open('input.i3d','w') as f:
        f.writelines(lines)
   
    logger.debug("Finished updating lines")
    
    if which('mpirun'):
        np = str(len(os.sched_getaffinity(0)) // 2)
        cmd = ['mpirun','-n',np,x3d_bin]
        subprocess.run(cmd)
    else:
        raise RuntimeError('mpirun command is required '
                            'to run this test')  

    os.chdir('..')

def get_git_sha(branch,origin=False):
    if origin:
        cmd = ['git','fetch', 'origin']
        out = subprocess.run(cmd,stdout=subprocess.DEVNULL)

        cmd = ['git','rev-parse', 'origin/'+branch]
    else:
        cmd = ['git','rev-parse', branch]
    out = subprocess.run(cmd,capture_output=True)
    return out.stdout.decode('utf-8')

def check_rebuild(branch):
    
    if not os.path.isfile('build.json'):
        logger.debug('build.json not found returning true')
        return True

    with open('build.json','r') as f:
        params = json.load(f)
    
    if not os.path.isfile(params['x3d_bin']):
        logger.debug(f"{params['x3d_bin']} not found returning True")
        return True

    if not os.path.isdir('Incompact3d'):
        logger.debug('Incompact3d directory not found returning True')
        return True
    
    os.chdir('Incompact3d')
    git_sha = get_git_sha(branch,origin=True)
    
    logger.debug(f"Checking git sha: {git_sha} {params['git_sha']}")
    os.chdir('..')
    return git_sha != params['git_sha']
    


def main():
    git_url =  'https://github.com/mattfalcone1997/Incompact3d.git'
    branch = 'tbl_recycle'
    
    if check_rebuild(branch):
        x3d = get_x3d(git_url,branch)
        x3d_bin = build_x3d()
    
        x3d_bin = os.path.join(x3d_bin,'xcompact3d')
    
        build_dict = dict(x3d_bin=x3d_bin,
                          x3d=x3d,
                          git_sha=get_git_sha(branch),
                          git_branch=branch,
                          git_url=git_url)

        os.chdir(os.path.join('..','..'))

        with open('build.json','w') as f:
            json.dump(build_dict,f)

    else:
        with open('build.json','r') as f:
            build_dict = json.load(f)

        x3d_bin = build_dict['x3d_bin']
        x3d = build_dict['x3d']

    run_channel_test(x3d_bin, x3d,'x3d_test_channel')
    run_blayer_test(x3d_bin,x3d,'x3d_test_blayer')

if __name__ == '__main__':
    main()
