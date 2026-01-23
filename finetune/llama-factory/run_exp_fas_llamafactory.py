# Run experiments on the Harvard FAS grid  
# Run this script from lit-trainer folder: cd learn-better/evolm/finetune/llamafactory


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
    # nodes_to_exclude = 'holygpu8a31505,holygpu8a25306,holygpu8a25104'  #nodes to exclude from job scheduling, ran into errors    
    nodes_to_exclude = 'holygpu8a27301,holygpu8a29201,holygpu8a31305,holygpu8a22405'  #nodes to exclude from job scheduling, ran into errors    

    
    if device=='n_gpus_a100_sxm4_80gb':
        sbatch_options = (
            f'sbatch '
            f'--job-name={job_name} '
            # f'--partition=seas_gpu,gpu,serial_requeue,gpu_requeue '
            f'--partition=seas_gpu '
            f'--nodes={n_nodes} '
            f'--gres=gpu:nvidia_a100-sxm4-80gb:{n_gpus} '
            f'--ntasks-per-node={n_tasks_per_node} '
            f'--cpus-per-task={cpus_per_task} '
            f'--time={time_days}-0{time_hrs}:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'--exclude={nodes_to_exclude} '
            f'$FILENAME'
        )
    if device=='n_gpus_a100':
        sbatch_options = (
            f'sbatch '
            f'--job-name={job_name} '
            f'--partition=seas_gpu,serial_requeue,gpu_requeue,gpu '
            f'--nodes={n_nodes} '
            f'--gres=gpu:{n_gpus} '
            f'--constraint=a100 '
            f'--ntasks-per-node={n_tasks_per_node} '
            f'--cpus-per-task={cpus_per_task} '
            f'--time={time_days}-0{time_hrs}:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'--exclude={nodes_to_exclude} '
            f'$FILENAME'
        )
    if device=='n_gpus_test':
        sbatch_options = (
            f'sbatch '
            f'--job-name={job_name} '
            f'--partition=gpu_test '
            f'--nodes={n_nodes} '
            f'--gres=gpu:{n_gpus} '
            f'--ntasks-per-node={n_tasks_per_node} '
            f'--cpus-per-task={cpus_per_task} '
            f'--time={time_days}-0{time_hrs}:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'--exclude={nodes_to_exclude} '
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
            f'--exclude=h{nodes_to_exclude} '
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
    

#exp01 -- finetune llama models -- finetune 0.5B-10BT and 1B-20BT (pretrained on fineweb dataset) on metamathqa dataset
#note
#sbatch: 4 GPUs (1 node): wnen nodes=1, ntasks-per-node must equal n_gpus --> so n_gpus = ntasks-per-node = 4
def run_exp01_finetune_llama():
    
    # bash script
    def create_bash_script(config_file_path: str) -> str:
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nmodule load python'
        f'\nmodule load cuda/12.9.1-fasrc01'
        f'\nmamba activate llamafactory'
        f'\nsource /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh' 
        f'\nFORCE_TORCHRUN=1 llamafactory-cli train {config_file_path}" > $FILENAME' #note ending quote
        )
        return bash_script_complete


    #demo w. small alpaca dataset (provided by llamafactory)
    # config_file_paths = ['config_hub/custom_configs/ft_metamathqa/llama-0.5B-10BT-weightdecay0.1-seed42-alpacaendemo.yaml']

    sft_dataset = 'simplescaling' #options: ['metamathqa', 'hellaswag','medmcqa', pubmedqa', 'mmluprocot', 'race', 'simplescaling']
    
    # # #0.5B models, llama
    # config_file_paths = [
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay0.0001-seed42-{sft_dataset}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay0.001-seed42-{sft_dataset}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay0.01-seed42-{sft_dataset}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay0.1-seed42-{sft_dataset}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay0.5-seed42-{sft_dataset}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay1.0-seed42-{sft_dataset}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay1.5-seed42-{sft_dataset}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay3.0-seed42-{sft_dataset}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay10.0-seed42-{sft_dataset}.yaml',
    #    ]

    # 1B models, llama
    # config_file_paths = [
    #     f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay0.0001-seed42-{sft_dataset}.yaml',
    #     f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay0.001-seed42-{sft_dataset}.yaml',
    #     f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay0.01-seed42-{sft_dataset}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay0.1-seed42-{sft_dataset}.yaml',
    #     f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay0.5-seed42-{sft_dataset}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay1.0-seed42-{sft_dataset}.yaml',
    #     f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay1.5-seed42-{sft_dataset}.yaml',
    #     f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay3.0-seed42-{sft_dataset}.yaml',
        # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay10.0-seed42-{sft_dataset}.yaml',
    # ]

    # olmo models
    config_file_paths = [
        #30BT (1x chinchilla)
        # f'config_hub/custom_configs/ft_{sft_dataset}/olmo-1B-30BT-weightdecay0.1-{sft_dataset}.yaml',
        # f'config_hub/custom_configs/ft_{sft_dataset}/olmo-1B-30BT-weightdecay0.3-{sft_dataset}.yaml',
        # f'config_hub/custom_configs/ft_{sft_dataset}/olmo-1B-30BT-weightdecay1.0-{sft_dataset}.yaml',
        #210BT (7x chinchilla)
        # f'config_hub/custom_configs/ft_{sft_dataset}/olmo-1B-210BT-weightdecay0.1-{sft_dataset}.yaml',
        f'config_hub/custom_configs/ft_{sft_dataset}/olmo-1B-210BT-weightdecay0.3-{sft_dataset}.yaml',
        # f'config_hub/custom_configs/ft_{sft_dataset}/olmo-1B-210BT-weightdecay1.0-{sft_dataset}.yaml',
    ]

    # 4B llama models -- eval-single-model--run-exp.sh: use vllm option 
    # config_file_paths = [
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-4B-80BT-weightdecay0.1-seed42-{sft_dataset}.yaml',
    #     f'config_hub/custom_configs/ft_{sft_dataset}/llama-4B-80BT-weightdecay1.0-seed42-{sft_dataset}.yaml',
    # ]



    #run each experiment, which has a different combination of arguments from single_args
    for config_file_path in config_file_paths:
        bash_script = create_bash_script(config_file_path)
        name = os.path.splitext(os.path.basename(config_file_path))[0]

        # run_bash_script(bash_script=bash_script, 
        #                 job_name=name,
        #                 log_file=f'exp01_finetune_llama/log_{name}',
        #                 device='n_gpus_a100_sxm4_80gb',
        #                 #note for sbatch: 4 GPUs (1 node): wnen nodes=1, ntasks-per-node must equal n_gpus --> so n_gpus = ntasks-per-node = 4 or 2
        #                 # n_nodes='1', n_gpus='2', n_tasks_per_node='2', cpus_per_task='16', time_days='0', time_hrs='1', memory_gb='64' #demo models (0.5B model)
        #                 n_nodes='1', n_gpus='4', n_tasks_per_node='4', cpus_per_task='16', time_days='0', time_hrs='6', memory_gb='64' #0.5B and 1B models
        #                 )
        
        run_bash_script_simplified(bash_script=bash_script, 
                        job_name=name,
                        log_file=f'exp01_finetune_llama/log_{name}',
                        
                        #note for sbatch: 4 GPUs (1 node): wnen nodes=1, ntasks-per-node must equal n_gpus --> so n_gpus = ntasks-per-node = 4 or 2
                        #model: 0.5B llama, 1B llama, 1B olmo
                        partition='seas_gpu,gpu,serial_requeue,gpu_requeue',
                        n_nodes='1', n_gpus_a100_80gb='4', n_tasks_per_node='4', cpus_per_task='12', time_hrs='6', memory_gb='64', 
                        #model: 4B llama
                        # partition='seas_gpu,gpu',
                        # n_nodes='1', n_gpus_a100_80gb='4', n_tasks_per_node='4', cpus_per_task='12', time_hrs='14', memory_gb='64',
                        # dependency_type_and_job_id='after:52127215'
                        )
        #actual run times
        #demo (0.5B-10BT, FT on alpacaendemo) -- 2 GPUs, 1 minute
        #0.5B-10BT models, FT on metamathqa for 3 epochs -- 4 GPUs, 3 hours 1min
        #1B-20BT models, FT on metamathqa for 3 epochs -- 4 GPUs, 4.5 hours

        print(f'job_name = {name}, options = {name}')  



def run_exp01_finetune_llama_additional_seeds():
    
    # bash script
    def create_bash_script(config_file_path: str) -> str:
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nmodule load python'
        f'\nmodule load cuda/12.9.1-fasrc01'
        f'\nmamba activate llamafactory'
        f'\nsource /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh' 
        f'\nFORCE_TORCHRUN=1 llamafactory-cli train {config_file_path}" > $FILENAME' #note ending quote
        )
        return bash_script_complete


    sft_dataset = 'medmcqa' #options: ['metamathqa', 'hellaswag','medmcqa', pubmedqa', 'mmluprocot', 'race']
    ft_seed = 5701 #[8107, 5701]

    #0.5B models, llama
    # config_file_paths = [
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay0.0001-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay0.001-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay0.01-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay0.1-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay0.5-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay1.0-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay1.5-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay3.0-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-0.5B-10BT-weightdecay10.0-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #    ]

    # # 1B models, llama
    # config_file_paths = [
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay0.0001-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay0.001-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay0.01-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay0.1-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay0.5-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay1.0-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay1.5-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay3.0-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    #     # f'config_hub/custom_configs/ft_{sft_dataset}/llama-1B-20BT-weightdecay10.0-seed42-{sft_dataset}-ftseed{ft_seed}.yaml',
    # ]


    #run each experiment, which has a different combination of arguments from single_args
    for config_file_path in config_file_paths:
        bash_script = create_bash_script(config_file_path)
        name = os.path.splitext(os.path.basename(config_file_path))[0]
        
        run_bash_script_simplified(bash_script=bash_script, 
                        job_name=name,
                        log_file=f'exp01_finetune_llama/log_{name}',
                        partition='seas_gpu,gpu,serial_requeue,gpu_requeue',
                        #note for sbatch: 4 GPUs (1 node): wnen nodes=1, ntasks-per-node must equal n_gpus --> so n_gpus = ntasks-per-node = 4 or 2
                        n_nodes='1', n_gpus_a100_80gb='4', n_tasks_per_node='4', cpus_per_task='12', time_hrs='6', memory_gb='64', #0.5B and 1B models
                        # dependency_type_and_job_id='after:52127215'
                        )

        print(f'job_name = {name}, options = {name}')  





#for exp02, need to copy and edit config file to finetune models pretrained on finefineweb

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
def create_config_files_for_exp02():
    percent_dois = [0.05, 0.01, 0.005, 0.001] #use percent_doi=0.1 as reference config file
    weight_decays = [0.0001, 0.001, 0.01, 0.1, 1.0]

    for percent_doi in percent_dois:
        for weight_decay in weight_decays:
            og_config_file_path = f'config_hub/custom_configs/ft_metamathqa/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.1-weightdecay{weight_decay}-seed42-metamathqa.yaml'
            new_config_file_path = f'config_hub/custom_configs/ft_metamathqa/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay{weight_decay}-seed42-metamathqa.yaml'
            copy_and_edit_config(og_config_file_path, new_config_file_path, percent_doi)




def run_exp02_finetune_models_pretrained_on_finefineweb():
    
    # bash script
    def create_bash_script(config_file_path: str) -> str:
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nmodule load python'
        f'\nmodule load cuda/12.9.1-fasrc01'
        f'\nmamba activate llamafactory'
        f'\nsource /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh' 
        f'\nFORCE_TORCHRUN=1 llamafactory-cli train {config_file_path}" > $FILENAME' #note ending quote
        )
        return bash_script_complete

    # #1B models
    #argument to change
    percent_doi = 0.001 #options: [0.1, 0.05, 0.01, 0.005, 0.001]
    
    config_file_paths = [
        # f'config_hub/custom_configs/ft_metamathqa/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.0001-seed42-metamathqa.yaml',
        f'config_hub/custom_configs/ft_metamathqa/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.001-seed42-metamathqa.yaml',
        f'config_hub/custom_configs/ft_metamathqa/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.01-seed42-metamathqa.yaml',
        f'config_hub/custom_configs/ft_metamathqa/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.1-seed42-metamathqa.yaml',
        f'config_hub/custom_configs/ft_metamathqa/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay1.0-seed42-metamathqa.yaml',
    ]


    #run each experiment, which has a different combination of arguments from single_args
    for config_file_path in config_file_paths:
        bash_script = create_bash_script(config_file_path)
        name = os.path.splitext(os.path.basename(config_file_path))[0]

        run_bash_script(bash_script=bash_script, 
                        job_name=name,
                        log_file=f'exp02_finetune_models_pretrained_on_finefineweb/log_{name}',
                        device='n_gpus_a100_sxm4_80gb',
                        #note for sbatch: 4 GPUs (1 node): wnen nodes=1, ntasks-per-node must equal n_gpus --> so n_gpus = ntasks-per-node = 4 or 2
                        n_nodes='1', n_gpus='4', n_tasks_per_node='4', cpus_per_task='16', time_days='0', time_hrs='8', memory_gb='64' #0.5B and 1B models
                        )
        #actual run times
        #1B-20BT models, FT on metamathqa for 3 epochs -- 4 GPUs, 4.5 hours

        print(f'job_name = {name}, options = {name}') 




def run_exp03_vary_wd_during_ft():
    
    # bash script
    def create_bash_script(config_file_path: str) -> str:
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nmodule load python'
        f'\nmodule load cuda/12.9.1-fasrc01'
        f'\nmamba activate llamafactory'
        f'\nsource /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh' 
        f'\nFORCE_TORCHRUN=1 llamafactory-cli train {config_file_path}" > $FILENAME' #note ending quote
        )
        return bash_script_complete
    

    # #1B models
    wd_pt_lst = [10.0] #options[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 0.5, 1.5]
    wd_ft_lst = [0.1] #options: [0.01, 0.1, 1.0]
    sft_dataset = "hellaswag" #options: ["metamathqa", "hellaswag"]

    # config_file_paths = [
    #     f'config_hub/custom_configs/ft_metamathqa/vary_wd_during_ft/llama-1B-20BT-weightdecay{wd_pt}-seed42-{sft_dataset}-ftweightdecay{wd_ft}.yaml' for wd_pt, wd_ft in product(wd_pt_lst, wd_ft_lst)
    # ]


    #run each experiment, which has a different combination of arguments from single_args
    for wd_pt, wd_ft in product(wd_pt_lst, wd_ft_lst):
        config_file_path = f'config_hub/custom_configs/ft_{sft_dataset}/vary_wd_during_ft/llama-1B-20BT-weightdecay{wd_pt}-seed42-{sft_dataset}-ftweightdecay{wd_ft}.yaml'
        bash_script = create_bash_script(config_file_path)
        name = os.path.splitext(os.path.basename(config_file_path))[0]
        arg_combo_str = f"wd_pt{wd_pt}_wd_ft{wd_ft}"

        run_bash_script_simplified(bash_script=bash_script, 
                        job_name=arg_combo_str,
                        log_file=f'exp03_vary_wd_during_ft/log_{name}',
                        partition='seas_gpu,gpu,serial_requeue,gpu_requeue',
                        #note for sbatch: 4 GPUs (1 node): wnen nodes=1, ntasks-per-node must equal n_gpus --> so n_gpus = ntasks-per-node = 4 or 2
                        n_nodes='1', n_gpus_a100_80gb='4', n_tasks_per_node='4', cpus_per_task='16', time_hrs='6', memory_gb='64', #0.5B and 1B models
                        # dependency_type_and_job_id='after:50575021'
                        )
        #actual run times
        #1B-20BT models, FT on metamathqa for 3 epochs -- 4 GPUs, 4.5 hours

        print(f'job_name = {arg_combo_str}, full_model_name = {name}')  



if __name__ == "__main__":
    run_exp01_finetune_llama()
    # run_exp01_finetune_llama_additional_seeds()


    # create_config_files_for_exp02() # create config files, does not submit jobs
    # run_exp02_finetune_models_pretrained_on_finefineweb()

    #create config files for exp03 -- write_config_files__vary_wd_during_ft.ipynb
    # run_exp03_vary_wd_during_ft()