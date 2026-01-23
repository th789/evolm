# Run experiments on the Harvard FAS grid  
# Run this script from lit-trainer folder: cd learn-better/evolm/pretrain/lit-trainer


import os
from itertools import product


def dict2options(single_args: dict) -> tuple[list[str], list[str]]:

    keys = [] #argument names, list of strings
    opts = [] #argument values, list of lists
    for k in single_args:
        keys.append(k)
        opts.append(single_args[k])

    exp_list = list(product(*opts)) #list of tuples, each tuple is one combination of arguments

    options_str = []
    name_str = []
    for option in exp_list:
        temp_str = '' #string of command line arguments
        temp_name_str = '' #string for naming the experiment
        for (k,i) in zip(keys, option):
            ###format of argument for full command line string
            temp_str+=f'--{k}={i} '
            
            ###format of argument name for full experiment name
            if k not in ['config']: #for most arguments, just add their values to the name string
                temp_name_str += f'{i}_'
            if k == 'config':  #for config argument, argument is a path, so only keep the name of the yaml file (which contains info about the model) not the full path
                config_file_name = os.path.splitext(os.path.basename(i))[0]
                temp_name_str += f'{config_file_name}_'
        
        options_str.append(temp_str) #full command line argument string
        name_str.append(temp_name_str[0:-1]) #full experiment name string

    return options_str, name_str


### Note on functions for experiments
# exp00_setup_XXX functions: define main command + sbatch options
# run_exp00_XXX functions: define experiment arguments + run each experiment using exp00_setup_XXX


def run_bash_script(bash_script: str, 
                    job_name: str,
                    log_file: str,
                    device: str,
                    n_nodes: str = '1',
                    n_tasks: str = '1',
                    n_gpus: str = '2',
                    n_tasks_per_node: str = '2',
                    cpus_per_task: str = '24',
                    time_hrs: str = '2',
                    time_days: str = '7',
                    memory_gb: str = '64',
                    ):
    # remove existing log file
    os.system('rm -f ' + log_file)

    #if log file has a folder, check if folder exists (if not, create folder)
    log_folder = os.path.split(log_file)[0]
    if log_folder != '':
        if not os.path.isdir(f'logs/{log_folder}'):
            os.makedirs(f'logs/{log_folder}')
    
    # bash script is defined in each experiment's run_expXX_XXX() function

    # sbatch options
    if device=='n_gpus_a100_sxm4_80gb':
        sbatch_options = (
            f'sbatch '
            f'--job-name={job_name} '
            f'--partition=seas_gpu,serial_requeue,gpu_requeue '
            f'--nodes={n_nodes} '
            f'--gres=gpu:nvidia_a100-sxm4-80gb:{n_gpus} '
            f'--ntasks-per-node={n_tasks_per_node} '
            f'--cpus-per-task={cpus_per_task} '
            f'--time=0-0{time_hrs}:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'--exclude=holygpu8a31505,holygpu8a25306,holygpu8a25104 '
            f'$FILENAME'
        )
    if device=='n_gpus_a100':
        #NOTE: small mistake, technically "n_gpus_a100" corresponds to the setting below, but leaving it as is for now for reproducibility
            # f'--gres=gpu:{n_gpus} '
            # f'--constraint=a100 '
        sbatch_options = (
            f'sbatch '
            f'--job-name={job_name} '
            f'--partition=seas_gpu,gpu '
            f'--nodes={n_nodes} '
            f'--gres=gpu:nvidia_a100-sxm4-80gb:{n_gpus} '
            f'--ntasks-per-node={n_tasks_per_node} '
            f'--cpus-per-task={cpus_per_task} '
            f'--time={time_days}-00:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'--exclude=holygpu8a31505,holygpu8a25306,holygpu8a25104 '
            f'$FILENAME'
        )
    if device=='seas_cpu':
        sbatch_options = (
            f'sbatch '
            f'--job-name={job_name} '
            f'--partition=seas_compute '
            f'--nodes={n_nodes} '
            f'--ntasks={n_tasks} '
            f'--cpus-per-task={cpus_per_task} '
            f'--time=0-0{time_hrs}:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'$FILENAME'
        )

    # run two commands above in order; remove temp file later
    full_cmd = f'{bash_script} ; {sbatch_options}'
    
    os.system(full_cmd)




def run_bash_script_simplified(bash_script: str, 
                               job_name: str,
                               log_file: str,
                               partition: str,
                               n_nodes: str = None,
                               n_tasks: str = None,
                               n_tasks_per_node: str = None,
                               cpus_per_task: str = None,
                               n_gpus_any: str = None,
                               n_gpus_a100: str = None,
                               n_gpus_a100_80gb: str = None,
                               time_mins: str = None,
                               time_hrs: str = None,
                               time_days: str = None,
                               memory_gb: str = None,
                               dependency_type_and_job_id: str = None,
                               ):
    # remove existing log file
    os.system('rm -f ' + log_file)

    #if log file has a folder, check if folder exists (if not, create folder)
    log_folder = os.path.split(log_file)[0]
    if log_folder != '':
        if not os.path.isdir(f'logs/{log_folder}'):
            os.makedirs(f'logs/{log_folder}')
    
    # bash script is defined in each experiment's run_expXX_XXX() function -- writes a file called FILENAME
    # sbatch options
    # sbatch_options = (
    #     f'sbatch '
    #     f'--job-name={job_name} '
    #     f'--partition={partition} '
    #     #nodes and tasks
    #     f'--nodes={n_nodes} ' if n_nodes is not None else ''
    #     f'--ntasks={n_tasks} ' if n_tasks is not None else ''
    #     f'--ntasks-per-node={n_tasks_per_node} ' if n_tasks_per_node is not None else ''
    #     f'--cpus-per-task={cpus_per_task} ' if cpus_per_task is not None else ''
    #     #specify gpus
    #     f'--gres=gpu:{n_gpus_any} ' if n_gpus_any is not None else '' #any gpu
    #     f'--gres=gpu:{n_gpus_a100} --constraint=a100 ' if n_gpus_a100 is not None else '' #a100 gpu
    #     f'--gres=gpu:nvidia_a100-sxm4-80gb:{n_gpus_a100_80gb} ' if n_gpus_a100_80gb is not None else '' #a100-sxm4-80gb gpu
    #     #time: specify only minutes, only days, only hours, or days+hours
    #     f'--time=0-00:0{time_mins} ' if time_mins is not None else ''
    #     f'--time={time_days}-00:00 ' if time_days is not None and time_hrs is None else ''
    #     f'--time=0-0{time_hrs}:00 ' if time_days is None and time_hrs is not None else ''
    #     f'--time={time_days}-0{time_hrs}:00 ' if time_days is not None and time_hrs is not None else ''
    #     #memory
    #     f'--mem={memory_gb}gb ' if memory_gb is not None else ''
    #     #log files
    #     f'--error=logs/{log_file}_jobid%j '
    #     f'--output=logs/{log_file}_jobid%j '
    #     f'--exclude=holygpu8a31505,holygpu8a25306,holygpu8a25104 '
    #     f'$FILENAME'
    #     )

    flags = [
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--error=logs/{log_file}_jobid%j",
        f"--output=logs/{log_file}_jobid%j",
    ]

    #nodes and tasks
    flags.append(f"--nodes={n_nodes}") if n_nodes else None
    flags.append(f"--ntasks={n_tasks}") if n_tasks else None
    flags.append(f"--ntasks-per-node={n_tasks_per_node}") if n_tasks_per_node else None
    flags.append(f"--cpus-per-task={cpus_per_task}") if cpus_per_task else None
    #specify gpus
    flags.append(f"--gres=gpu:{n_gpus_any}") if n_gpus_any else None  #any gpu
    flags.append(f"--gres=gpu:{n_gpus_a100} --constraint=a100") if n_gpus_a100 else None  #a100 gpu
    flags.append(f"--gres=gpu:nvidia_a100-sxm4-80gb:{n_gpus_a100_80gb}") if n_gpus_a100_80gb else None #a100-sxm4-80gb gpu
    flags.append(f"--exclude=holygpu8a31505,holygpu8a25306,holygpu8a25104") if n_gpus_any or n_gpus_a100 or n_gpus_a100_80gb else None
    #time: can specify only minutes, only days, only hours, or any combo of the three
    time_days = time_days if time_days is not None else '0'
    time_hrs = time_hrs if time_hrs is not None else '0'
    time_mins = time_mins if time_mins is not None else '0'
    flags.append(f"--time={time_days}-0{time_hrs}:0{time_mins}")
    #memory
    flags.append(f"--mem={memory_gb}gb") if memory_gb else None
    #dependency
    flags.append(f"--dependency={dependency_type_and_job_id}") if dependency_type_and_job_id else None

    sbatch_options = "sbatch " + " ".join(flags) + " " + "$FILENAME"

    # run two commands above in order; remove temp file later
    # OLD does not work with heredoc, see example bash scriptin run_exp05...() -- (causes EOF ; sbatch syntax error)
    # full_cmd = f'{bash_script} ; {sbatch_options}'

    # NEW: run sbatch on a new line after the heredoc
    full_cmd = f"{bash_script}\n{sbatch_options}"

    
    os.system(full_cmd)


#exp00 -- demo to pretrain small pythia model -- 14m params, 370m tokens (tiny stories dataset)
    #note for 2 GPUs
    #bash script: must use srun + don't set CUDA_VISIBLE_DEVICES inside
    #sbatch options: when nodes=1 (using <= 4GPUs), must have --ntasks-per-node equal to # of gpus in --gres=gpu

def run_exp00_pretrain_pythia():

    def create_bash_script(bash_script_args):
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nexport WANDB_API_KEY=b10df87569c5fdcef6d7b86acf29819b378fe28d'
        f'\nexport WANDB_ENTITY=th789-harvard'
        f'\nexport WANDB_PROJECT=overtraining'
        f'\nmodule load python'
        f'\nmamba activate litgpt-e'
        f'\nsource /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh' 
        f'\nexport FINEFINEWEB_FOLDER_PATH_PV="/n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset20BT/mathematics0.01_litgpt/pretrain" ' #specify any path for ffw -- since this experiment does not actually use ffw
        f'\nsrun litgpt pretrain {bash_script_args}" > $FILENAME' #note ending quote
        )
        return bash_script_complete

    single_args = {
        'config': ['config_hub/custom_configs/pretrain/pythia-14m.yaml']
    }

    list_arg_combos, list_names = dict2options(single_args)

    #run each experiment, which has a different combination of arguments from single_args
    for (arg_combo, name) in zip(list_arg_combos, list_names):
        bash_script = create_bash_script(arg_combo)
        
        run_bash_script(bash_script=bash_script, 
                        job_name=name, 
                        log_file=f'exp00_pretrain_pythia/log_{name}', 
                        #2 GPUs
                        # device='n_gpus_a100_sxm4_80gb', n_nodes='1', n_gpus='2', n_tasks_per_node='2', cpus_per_task='24', time_hrs='1', memory_gb='64'
                        #1 GPU
                        device='n_gpus_a100_sxm4_80gb', n_nodes='1', n_gpus='1', n_tasks_per_node='1', cpus_per_task='12', time_hrs='1', memory_gb='32'
                        )

        print(f'job_name = {name}, options = {arg_combo}')  






#exp01 -- convert downloaded data (fineweb) to litgpt format
#no arg combos, just define main_command_in_bash_script
def run_exp01_prepare_data_fineweb():
    
    # this is the bash script: evolm/pretrain/lit-trainer/jobs/prepare_data/prepare_fineweb.sh or prepare_finemath.sh
    def create_bash_script(main_command_in_bash_script):
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nexport TMPDIR="/n/netscratch/doshi-velez_lab/Everyone/tmp"'
        f'\nexport DATA_OPTIMIZER_CACHE_FOLDER="/n/netscratch/doshi-velez_lab/Everyone/tmp_data_optimizer_cache"'
        f'\nmodule load python'
        f'\nmamba activate litgpt-e'
        f'\n{main_command_in_bash_script}" > $FILENAME' #note ending quote
        )
        return bash_script_complete

    #prepare fineweb data 350BT
    main_command_in_bash_script = f"""python litgpt/scripts/prepare_fineweb.py \
        /n/netscratch/doshi-velez_lab/Everyone/fineweb/sample/350BT \
        /n/netscratch/doshi-velez_lab/Everyone/fineweb_litgpt/pretrain/train \
        /n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/meta-llama/Llama-2-7b-hf """
    job_name='prepare_fineweb_350BT'
    log_file='exp01a_prepare_fineweb/log_prepare_fineweb_350BT'

    #prepare fineweb data 100BT


    #run experiment
    bash_script = create_bash_script(main_command_in_bash_script)
    run_bash_script(bash_script=bash_script,
                    job_name=job_name,
                    log_file=log_file,
                    device='seas_cpu', n_nodes='1', n_tasks='1', cpus_per_task='48', time_hrs='8', memory_gb='64')

    print(f'job_name = {job_name}')  


    

#exp02 -- pretrain llama models -- 0.5B-10BT and and 1B-20BT (fineweb dataset)  
#0.5B-10BT models -- 4 GPUs, 15 hours --> request more than this
#1B-20BT models -- uses 8 GPUs, 1 day --> request more than this
#note
#bash script: must use srun + don't set CUDA_VISIBLE_DEVICES inside (slurm does this on its own when using srun)
#sbatch: 4 GPUs (1 node): wnen nodes=1, ntasks-per-node must equal n_gpus --> so n_gpus = ntasks-per-node = 4
#sbatch: 8 GPUs (2 nodes) wnen nodes=2, --ntasks-per-node can equal 1 --> so  n_gpus=4, ntasks-per-node=1

def run_exp02_pretrain_llama():
    
    # bash script
    def create_bash_script(bash_script_args):
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nexport WANDB_API_KEY=b10df87569c5fdcef6d7b86acf29819b378fe28d'
        f'\nexport WANDB_ENTITY=th789-harvard'
        f'\nexport WANDB_PROJECT=overtraining'
        f'\nmodule load python'
        f'\nmamba activate litgpt-e'
        f'\nsource /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh' 
        f'\nexport FINEFINEWEB_FOLDER_PATH_PV="/n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset20BT/mathematics0.01_litgpt/pretrain" ' #specify any path for ffw -- since this experiment does not actually use ffw
        f'\nsrun litgpt pretrain {bash_script_args}" > $FILENAME' #note ending quote
        )
        return bash_script_complete

    ### ------------------- pretrain models on FineWeb ------------------------
    #0.5B models
    # single_args = {
    #     'config': [
    #         # 'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay0.0001-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay0.001-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay0.01-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay0.1-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay0.5-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay1.0-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay1.5-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay3.0-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay10.0-seed42.yaml',
    #                ]
    # }
    # #1B models
    # single_args = {
    #     'config': [
    #         # 'config_hub/custom_configs/pretrain/llama-1B-20BT-weightdecay0.0001-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-1B-20BT-weightdecay0.001-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-1B-20BT-weightdecay0.01-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-1B-20BT-weightdecay0.1-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-1B-20BT-weightdecay0.5-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-1B-20BT-weightdecay1.0-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-1B-20BT-weightdecay1.5-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-1B-20BT-weightdecay3.0-seed42.yaml',
    #         # 'config_hub/custom_configs/pretrain/llama-1B-20BT-weightdecay10.0-seed42.yaml',
    #                ]
    # }
    ### ----------------------------------------------------------------------


    list_arg_combos, list_names = dict2options(single_args)

    #run each experiment, which has a different combination of arguments from single_args
    for (arg_combo, name) in zip(list_arg_combos, list_names):
        bash_script = create_bash_script(arg_combo)

        run_bash_script(bash_script=bash_script, 
                        job_name=name,
                        log_file=f'exp02_pretrain_llama/log_{name}',
                        device='n_gpus_a100',
                        # n_nodes='1', n_gpus='4', n_tasks_per_node='4', cpus_per_task='16', time_days='1', memory_gb='64' #0.5B models
                        n_nodes='2', n_gpus='4', n_tasks_per_node='4', cpus_per_task='16', time_days='2', memory_gb='128' #1B models
                        )

        print(f'job_name = {name}, options = {arg_combo}')  




def run_exp06_compute_val_loss():

    # bash script
    def create_bash_script(config_file_path: str) -> str:
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\necho "Start time: $(date +"%Y-%m-%d %H:%M:%S")"'
        f'\nexport WANDB_API_KEY=b10df87569c5fdcef6d7b86acf29819b378fe28d'
        f'\nexport WANDB_ENTITY=th789-harvard'
        f'\nexport WANDB_PROJECT=overtraining'
        f'\nmodule load python'
        f'\nmamba activate litgpt-e'
        f'\nsource /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh' 
        f'\nexport FINEFINEWEB_FOLDER_PATH_PV="/n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset20BT/mathematics0.01_litgpt/pretrain" ' #specify any path for ffw -- since this experiment does not actually use ffw
        f'\nsrun litgpt pretrain --config {config_file_path}'
        f'\necho "End time: $(date +"%Y-%m-%d %H:%M:%S")" " > $FILENAME' #note ending quote
        )
        return bash_script_complete

    ### ------------------- compute val loss ------------------------

    ### start here : add pythia
    # config_lst = [
    #     # 'config_hub/custom_configs/val/pythia-14m-val.yaml', #pythia does not work due to data imcompatibilty
    #     'config_hub/custom_configs/val/llama-0.5B-10BT-weightdecay0.1-seed42-val.yaml',
    #     ]

    # # # #0.5B models    
    # config_lst = [
    #     # 'config_hub/custom_configs/val/llama-0.5B-10BT-weightdecay0.0001-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-0.5B-10BT-weightdecay0.001-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-0.5B-10BT-weightdecay0.01-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-0.5B-10BT-weightdecay0.1-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-0.5B-10BT-weightdecay0.5-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-0.5B-10BT-weightdecay1.0-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-0.5B-10BT-weightdecay1.5-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-0.5B-10BT-weightdecay3.0-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-0.5B-10BT-weightdecay10.0-seed42-val.yaml',
    #     ]

    # # #1B models
    # config_lst = [
    #     # 'config_hub/custom_configs/val/llama-1B-20BT-weightdecay0.0001-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-1B-20BT-weightdecay0.001-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-1B-20BT-weightdecay0.01-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-1B-20BT-weightdecay0.1-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-1B-20BT-weightdecay0.5-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-1B-20BT-weightdecay1.0-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-1B-20BT-weightdecay1.5-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-1B-20BT-weightdecay3.0-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-1B-20BT-weightdecay10.0-seed42-val.yaml',
    #     ]

    # #4B models
    # config_lst = [
    #     'config_hub/custom_configs/val/llama-4B-80BT-weightdecay0.1-seed42-val.yaml',
    #     # 'config_hub/custom_configs/val/llama-4B-80BT-weightdecay1.0-seed42-val.yaml',
    #     ]
    
 

    ### ----------------------------------------------------------------------



    #run each experiment, which has a different combination of arguments from single_args
    for config in config_lst:

        bash_script = create_bash_script(config)

        job_name = config.split('/')[-1].replace('.yaml', '') #get string after last / and remove .yaml

        run_bash_script_simplified(
            bash_script=bash_script,
            job_name=job_name,
            log_file=f'exp06_val/log_{job_name}',
            partition='seas_gpu,gpu,gpu_requeue,serial_requeue',
            # n_gpus_any='1', time_hrs='1', memory_gb='32' #pythia demo -- DOES NOT WORK
            # n_gpus_any='1', time_hrs='2', memory_gb='64', # 0.5B + 1B models
            n_gpus_any='1', time_hrs='6', memory_gb='64', # 4B models
            # dependency_type_and_job_id='afterok:56451233'
            )

        print(f'job_name = {job_name}')




def run_exp03_prepare_data_finefineweb():
    
    #change argument
    percent_doi = 0.001 #options: [0.1, 0.05, 0.01, 0.005, 0.001]

    # this is the bash script: evolm/pretrain/lit-trainer/jobs/prepare_data/prepare_finefineweb.sh
    def create_bash_script(main_command_in_bash_script):
        # bash_script_complete = (
        # f'FILENAME=$(mktemp) ; '
        # f'echo "#!/bin/sh' #note starting quote
        # f'\nexport TMPDIR="/n/netscratch/doshi-velez_lab/Everyone/tmp{percent_doi}" '
        # f'\nexport DATA_OPTIMIZER_CACHE_FOLDER="/n/netscratch/doshi-velez_lab/Everyone/tmp_data_optimizer_cache{percent_doi}" '
        # f'\n[ ! -d "$TMPDIR" ] && mkdir -p "$TMPDIR" ' #create tmp directory if it doesn't exist
        # f'\n[ ! -d "$DATA_OPTIMIZER_CACHE_FOLDER" ] && mkdir -p "$DATA_OPTIMIZER_CACHE_FOLDER" ' #create data optimizer cache folder if it doesn't exist
        # f'\nmodule load python'
        # f'\nmamba activate litgpt-e'
        # f'\n{main_command_in_bash_script} '
        # f'\nrm -rf "$TMPDIR" '
        # f'\nrm -rf "$DATA_OPTIMIZER_CACHE_FOLDER" '
        # f'\n" > $FILENAME' #note ending quote
        # )

        # Make sure TMPDIR and DATA_OPTIMIZER_CACHE_FOLDER variables (folders) exist AND are empty!
        # best to remake them each time
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nexport TMPDIR="/n/netscratch/doshi-velez_lab/Everyone/tmp" '
        f'\nexport DATA_OPTIMIZER_CACHE_FOLDER="/n/netscratch/doshi-velez_lab/Everyone/tmp_data_optimizer_cache" '
        #create these folders -- somehow compute nodes cannot see them
        f'\nmodule load python'
        f'\nmamba activate litgpt-e'
        f'\n{main_command_in_bash_script} " > $FILENAME' #note ending quote
        )
        return bash_script_complete

    #prepare finefineweb data 20BT
    main_command_in_bash_script = f"""python litgpt/scripts/prepare_finefineweb.py \
        /n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset20BT/mathematics{percent_doi} \
        /n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset20BT/mathematics{percent_doi}_litgpt/pretrain/train \
        /n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/meta-llama/Llama-2-7b-hf """
    
    #run experiment
    bash_script = create_bash_script(main_command_in_bash_script)
    job_name=f'prep_ffw_mysubset20BT_mathematics{percent_doi}'
    log_file=f'exp03_prepare_finefineweb/log_prepare_mysubset20BT_mathematics{percent_doi}'

    run_bash_script(bash_script=bash_script,
                    job_name=job_name,
                    log_file=log_file,
                    device='seas_cpu', n_nodes='1', n_tasks='1', cpus_per_task='48', time_hrs='8', memory_gb='64')

    print(f'job_name = {job_name}')  






#for exp04, need to copy and edit config file

import shutil
def copy_and_edit_config(og_config_file_path: str,
                         new_config_file_path: str,
                         percent_doi) -> None:
    """
    Copy a YAML config file and, in the copy, replace all occurrences of
    'mathematics0.1' with 'mathematics{percent_doi}'.

    Parameters
    ----------
    og_config_file_path : str
        Path to the original YAML config file.
    new_config_file_path : str
        Path where the modified config file will be written.
    percent_doi : float
        The value to insert after 'mathematics'. Can be a string or a number.
    """
    
    # 1. Copy original to new location
    shutil.copyfile(og_config_file_path, new_config_file_path)
    # 2. Read contents of the new file
    with open(new_config_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # 3. Replace occurrences of 'mathematics0.1' with 'mathematics{percent_doi}'
    text = text.replace("mathematics0.1", f"mathematics{str(percent_doi)}")
    # 4. Write back the modified contents
    with open(new_config_file_path, "w", encoding="utf-8") as f:
        f.write(text)


#create config files for exp04
def create_config_files_for_exp04():
    percent_dois = [0.05, 0.01, 0.005, 0.001] #use percent_doi=0.1 as reference config file
    weight_decays = [0.0001, 0.001, 0.01, 0.1, 1.0]

    for percent_doi in percent_dois:
        for weight_decay in weight_decays:
            og_config_file_path = f'config_hub/custom_configs/pretrain/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.1-weightdecay{weight_decay}-seed42.yaml'
            new_config_file_path = f'config_hub/custom_configs/pretrain/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay{weight_decay}-seed42.yaml'
            copy_and_edit_config(og_config_file_path, new_config_file_path, percent_doi)



def run_exp04_pretrain_llama_on_finefineweb():
    
    #set argument
    percent_doi = 0.001 #options: [0.1, 0.05, 0.01, 0.005, 0.001]

    # bash script
    def create_bash_script(bash_script_args):
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nmodule load python'
        f'\nmamba activate litgpt-e'
        #load wandb variables + FINEFINEWEB_FOLDER_PATH_PV (needed because litgpt.data.__init__.py imports both FineWeb and FineFineWeb unconditionally)
        f'\nsource /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh' 
        #***load training data based on argument
        f'\nexport FINEFINEWEB_FOLDER_PATH_PV="/n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset20BT/mathematics{percent_doi}_litgpt/pretrain" ' #specify path for pretraining data
        # f'\necho "*********** DATA FOLDER: $FINEFINEWEB_FOLDER_PATH_PV" ' #using double quotes -- does not print correctly
        f'\necho \'*********** DATA FOLDER: \'$FINEFINEWEB_FOLDER_PATH_PV ' #using single quotes -- need to test if it prints correctly
        f'\nsrun litgpt pretrain {bash_script_args}" > $FILENAME' #note ending quote
        )
        return bash_script_complete


    ### ----------- pretrain models on FineFineWeb, my subset 20BT -----------
    #1B models
    single_args = {
        'config': [
            f'config_hub/custom_configs/pretrain/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.0001-seed42.yaml',
            f'config_hub/custom_configs/pretrain/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.001-seed42.yaml',
            f'config_hub/custom_configs/pretrain/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.01-seed42.yaml',
            f'config_hub/custom_configs/pretrain/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.1-seed42.yaml',
            f'config_hub/custom_configs/pretrain/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay1.0-seed42.yaml'
                   ]
    }
    ### ----------------------------------------------------------------------


    list_arg_combos, list_names = dict2options(single_args)

    #run each experiment, which has a different combination of arguments from single_args
    for (arg_combo, name) in zip(list_arg_combos, list_names):
        bash_script = create_bash_script(arg_combo)

        run_bash_script(bash_script=bash_script, 
                        job_name=name,
                        log_file=f'exp04_pretrain_llama_on_finefineweb/log_{name}',
                        device='n_gpus_a100',
                        n_nodes='2', n_gpus='4', n_tasks_per_node='4', cpus_per_task='16', time_days='3', memory_gb='128' #1B models
                        )

        print(f'job_name = {name}, options = {arg_combo}')  

    


def run_exp05_convert_pretrained_models_to_hf_format():

    def create_bash_script(percent_doi: float, weight_decay: float):
        #script from evolm/pretrain/lit-trainer/jobs/convert_ckpts/convert_from_lit.sh
        # bash_script_complete = (
        # f'FILENAME=$(mktemp) ; '
        # f'echo \"#!/bin/sh' #note starting quote
        # f'\nmodule load python'
        # f'\nmamba activate litgpt-e'
        # f'\nmodel_dir=\"models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay{weight_decay}-seed42\" '
        # #***load training data based on argument
        # f'\nsource /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh ' 
        # f'\nexport FINEFINEWEB_FOLDER_PATH_PV=\"/n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset20BT/mathematics{percent_doi}_litgpt/pretrain\" '
        # # f'\necho \"-------------> DATA FOLDER: $FINEFINEWEB_FOLDER_PATH_PV <-------------\" ' #using single quotes -- need to test if it prints correctly
        # f'\nname=final '
        # f'\nlit_ckpt_dir=$model_dir/$name '
        # f'\nlit_convert_out_dir=$model_dir/$name-converted '
        # f'\nhf_ckpt_dir=$model_dir/$name-hf '
        # f'\nlitgpt convert_from_litgpt $lit_ckpt_dir $lit_convert_out_dir '
        # f'\npython scripts/make_hf_model.py --lit_convert_out_dir $lit_convert_out_dir --save_dir $hf_ckpt_dir --disable_test_vllm " > $FILENAME' #note ending quote
        # )

        # return bash_script_complete

        bash_script_complete =  f"""FILENAME=$(mktemp) ;
cat > "$FILENAME" << 'EOF'
#!/bin/sh
module load python
mamba activate litgpt-e

# model_dir="models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay{weight_decay}-seed42"
model_dir="models/pretrained/llama-0.5B-10BT-weightdecay10.0-seed42"
# model_dir="models/pretrained/llama-1B-20BT-weightdecay10.0-seed42"

source /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh
export FINEFINEWEB_FOLDER_PATH_PV="/n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset20BT/mathematics{percent_doi}_litgpt/pretrain"

name=final
lit_ckpt_dir="$model_dir/$name"
lit_convert_out_dir="$model_dir/${{name}}-converted"
hf_ckpt_dir="$model_dir/${{name}}-hf"

litgpt convert_from_litgpt "$lit_ckpt_dir" "$lit_convert_out_dir"

python scripts/make_hf_model.py \\
  --lit_convert_out_dir "$lit_convert_out_dir" \\
  --save_dir "$hf_ckpt_dir" \\
  --disable_test_vllm
EOF
"""     
        return bash_script_complete

    percent_dois = [0.005] #options: [0.1, 0.05, 0.01, 0.005, 0.001]
    weight_decays = [0.0001] #options: [0.0001, 0.001, 0.01, 0.1, 1.0]

    for percent_doi, weight_decay in product(percent_dois, weight_decays):
        bash_script = create_bash_script(percent_doi, weight_decay)
        
        job_name = f'convert_model_mathematics{percent_doi}_weightdecay{weight_decay}'
        run_bash_script_simplified(
            bash_script=bash_script,
            job_name=job_name,
            log_file=f'exp05_convert_pretrained_models_to_hf_format/log_{job_name}',
            partition='gpu_test',
            n_gpus_any='1', time_mins='15', memory_gb='64' #1B models
            )

        print(f'job_name = {job_name}')  


if __name__ == "__main__":
    # run_exp00_pretrain_pythia()
    # run_exp01_prepare_data_fineweb()
    # run_exp02_pretrain_llama()
    run_exp06_compute_val_loss()


    # run_exp03_prepare_data_finefineweb() 
        #!!!NOTE for run_exp03_prepare_data_finefineweb
        # run one job at a time -- they share TMPDIR and DATA_OPTIMIZER_CACHE_FOLDER variables/folders, so need to be careful not to overwrite each other
        # before running each job, best to remake TMPDIR and DATA_OPTIMIZER_CACHE_FOLDER folders each time


    # create_config_files_for_exp04() # create config files, does not submit jobs
    # run_exp04_pretrain_llama_on_finefineweb()


    #convert pretrained models to hf format for finetuning
    # run_exp05_convert_pretrained_models_to_hf_format() 
    



