'''
this script 
- uses argparse to determine the jobs to run 
    a either runs the job 
    b1 writes the .sh file for each job
    b2 submits the jobs
    b3 deletes the .sh files
'''
import os
import time
from copy import deepcopy

import argparse
parser = argparse.ArgumentParser(description='run jobs')

# sim
parser.add_argument('--sim_sif', '-ss', type=str, default='evogym_numba_networkx.sif', 
                    choices=['evogym_numba_networkx.sif'], help='simulation sif')
# task 
parser.add_argument('--task', '-t', help='specify the task',
                    choices=['BasicEnv-v0', 'LogicEnv-v0'], default='BasicEnv-v0')
# experiment related arguments
parser.add_argument('-ros', '--run_or_slurm', type=str,
                    help='run the job directly or submit it to slurm', choices=['slurm', 'run'])
parser.add_argument('-sq', '--slurm_queue', type=str,
                    help='choose the job queue for the slurm submissions', choices=['short', 'week', 'bluemoon'])
parser.add_argument('-cpu', '--cpu', type=int, 
                    help='number of cpu cores requested for slurm job')
parser.add_argument('-sp', '--saving_path', help='path to save the experiment')
parser.add_argument('-n', '--n_jobs', type=int, default=1,
                    help='number of jobs to submit')
# evolutionary algorithm related arguments
parser.add_argument('--evolutionary_algorithm', '-ea', type=str,
                    choices=['qd'], help='choose the evolutionary algorithm')
parser.add_argument('-nrp', '--nr_parents', type=int,
                     help='number of parents')
parser.add_argument('-nrg', '--nr_generations', type=int,
                     help='number of generations')
parser.add_argument('--nr_random_individual', '-nri', type=int, 
                    help='Number of random individuals to insert each generation')
# softrobot related arguments
parser.add_argument('--bounding_box', '-bb', nargs='+', type=int,
                    help='Bounding box dimensions (x,y). e.g.IND_SIZE=(6, 6)->workspace is a rectangle of 6x6 voxels') # trying to get rid of this

parser.add_argument('--gif_every', '-ge', type=int, default=50)

parser.add_argument('-r', '--repetition', type=int,
                    help='repetition number, dont specify this if you want it to be determined automatically', nargs='+')
parser.add_argument('-id', '--id', type=str,
                    help='id of the job, dont specify this if you want no specific id for the jobs', nargs='+')

# internal use only
parser.add_argument('--local', '-l', action='store_true')

args = parser.parse_args()

def run(args):
    import random
    import numpy as np
    import multiprocessing
    import time

    from utils import prepare_rundir
    from population import ARCHIVE
    from algorithms import MAP_ELITES
    from make_gif import MAKEGIF
    import settings

    multiprocessing.set_start_method('spawn')

    # save the start time
    settings.START_TIME = time.time()

    # run the job directly
    if args.repetition is None:
        args.repetition = [1]
    rundir = prepare_rundir(args)
    args.rundir = rundir
    print('rundir', rundir)

    # if this experiment is currently running or has finished, we don't want to run it again
    if os.path.exists(args.rundir + '/RUNNING') and not args.local:
        print('Experiment is already running')
        exit()
    if os.path.exists(args.rundir + '/FINISHED'):
        print('Experiment has already finished')
        exit()

    # Initializing the random number generator for reproducibility
    SEED = args.repetition[0]
    random.seed(SEED)
    np.random.seed(SEED)

    # Setting up the optimization algorithm and runnning
    if args.evolutionary_algorithm == 'qd':
        map_elites = MAP_ELITES(args=args, map=ARCHIVE(args=args))
        map_elites.optimize()
    else:
        raise ValueError('unknown evolutionary algorithm')

    # delete running file in any case
    if os.path.isfile(args.rundir + '/RUNNING'):
        os.remove(args.rundir + '/RUNNING')

    # if the job is finished, create a finished file
    if not settings.STOP:
        # write a file to indicate that the job finished successfully
        with open(args.rundir + '/FINISHED', 'w') as f:
            pass
    else:
        # if the job is not finished, resubmit it
        print('resubmitting job')
        submit_slurm(args, resubmit=True)

def submit_slurm(args, resubmit=False):
    # submit the job to slurm
    base_string = '#!/bin/sh\n\n'
    base_string += '#SBATCH --partition=' + args.slurm_queue + ' # Specify a partition \n\n'
    base_string += '#SBATCH --nodes=1  # Request nodes \n\n'
    if args.cpu is None:
        if args.nr_parents < 50:
            base_string += '#SBATCH --ntasks=' + str(args.nr_parents*2) + '  # Request some processor cores \n\n'
        else:
            base_string += '#SBATCH --ntasks=100 # Request some processor cores \n\n'
    else:
        base_string += '#SBATCH --ntasks=' + str(args.cpu) + '  # Request some processor cores \n\n'
    base_string += '#SBATCH --job-name=evogym  # Name job \n\n'
    base_string += '#SBATCH --signal=B:USR1@600  # signal the bash \n\n'
    base_string += '#SBATCH --output=outs/%x_%j.out  # Name output file \n\n'
    base_string += '#SBATCH --mail-user=alican.mertan@uvm.edu  # Set email address (for user with email "usr1234@uvm.edu") \n\n'
    base_string += '#SBATCH --mail-type=FAIL   # Request email to be sent at begin and end, and if fails \n\n'
    base_string += '#SBATCH --mem-per-cpu=10GB  # Request 16 GB of memory per core \n\n'
    if args.slurm_queue == 'short':
        base_string += '#SBATCH --time=0-02:30:00  # Request 2 hours and 20 minutes of runtime \n\n'
    elif args.slurm_queue == 'week':
        base_string += '#SBATCH --time=6-00:00:00  # Request 6 days of runtime \n\n'
    elif args.slurm_queue == 'bluemoon':
        base_string += '#SBATCH --time=0-28:00:00  # Request 28 hours of runtime \n\n'

    base_string += 'cd /users/a/m/amertan/workspace/EVOGYM/evogym-sab/ \n'
    base_string += 'spack load singularity@3.7.1\n'
    base_string += 'trap \'kill -INT "${PID}"; wait "${PID}"; handler\' USR1 SIGINT SIGTERM \n'
    base_string += 'singularity exec --bind ../../../scratch/evogym_experiments:/scratch_evogym_experiments ' + args.sim_sif + ' xvfb-run -a python3 '

    # for each job
    for i in range(args.n_jobs):
        # create a string to write to the .sh file
        string_to_write = base_string
        string_to_write += 'main.py '
        # iterate over all of the arguments
        dict_args = deepcopy(vars(args))
        # handle certain arguments differently
        if 'run_or_slurm' in dict_args:
            dict_args['run_or_slurm'] = 'run'
        if 'n_jobs' in dict_args:
            dict_args['n_jobs'] = 1
        if 'repetition' in dict_args:
            if dict_args['repetition'] is None:
                dict_args['repetition'] = i+1
            else:
                dict_args['repetition'] = dict_args['repetition'][i]
        if 'rundir' in dict_args: # rundir might be in, delete it
            del dict_args['rundir']
        # write the arguments
        for key in dict_args:
            # key can be None, skip it in that case
            if dict_args[key] is None:
                continue
            # if the key is a list, we need to iterate over the list
            if isinstance(dict_args[key], list) or isinstance(dict_args[key], tuple):
                string_to_write += '--' + key + ' '
                for item in dict_args[key]:
                    string_to_write += str(item) + ' '
            elif isinstance(dict_args[key], bool):
                if dict_args[key]:
                    string_to_write += '--' + key + ' '
            else:
                string_to_write += '--' + key + ' ' + str(dict_args[key]) + ' '
        # job submission or resubmission
        if resubmit == False: # this process can call sbatch since it is not in container
            # write to the file
            with open('job.sh', 'w') as f:
                f.write(string_to_write + '&\nPID="$!"\nwait "${PID}"\n')
            # submit the job
            os.system('sbatch job.sh')
            # sleep for a second
            time.sleep(0.1)
            # remove the job file
            os.remove('job.sh')
            # sleep for a second
            time.sleep(0.1)
        else: # this process is in container, so it cannot call sbatch. there is shell script running that will check for sh files and submit them
            import random
            import string
            # write to the file
            with open('resubmit_'+''.join(random.choices(string.ascii_lowercase, k=5))+'.sh', 'w') as f:
                f.write(string_to_write + '&\nPID="$!"\nwait "${PID}"\n')


if __name__ == '__main__':

    if args.run_or_slurm == 'run':
        # run the job
        run(args)
    elif args.run_or_slurm == 'slurm':
        # submit the job
        submit_slurm(args)
    else:
        raise ValueError('run_or_slurm must be either run or slurm')





