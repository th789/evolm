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


#exp00 -- 2 gpus
#demo to pretrain small pythia model -- 14m params, 370m tokens (tiny stories dataset)
def setup_exp00_pretrain_pythia(options: str,
                                job_name: str,
                                log_file: str,
                                device: str,
                                nodes: str = '1',
                                ntasks_per_node: str = '2',
                                cpus_per_task: str = '24',
                                time_hrs: str = '2',
                                memory_gb: str = '64') -> None:

    # remove existing log file
    os.system('rm -f ' + log_file)

    #if log file has a folder, check if folder exists (if not, create folder)
    log_folder = os.path.split(log_file)[0]
    if log_folder != '':
        if not os.path.isdir(f'logs/{log_folder}'):
            os.makedirs(f'logs/{log_folder}')
    


    ### note
    #bash script: must use srun + don't set CUDA_VISIBLE_DEVICES inside
    #sbatch: must have --ntasks-per-node equal to # of gpus in --gres=gpu

    if device=='gpu_two_a100_80gb':
        # this is the bash script
        main_command = (
            f'FILENAME=$(mktemp) ; '
            f'echo "#!/bin/sh' #note starting quote
            f'\nexport WANDB_API_KEY=b10df87569c5fdcef6d7b86acf29819b378fe28d'
            f'\nexport WANDB_ENTITY=th789-harvard'
            f'\nexport WANDB_PROJECT=overtraining'
            f'\nmodule load python'
            f'\nmamba activate litgpt-e'
            f'\nsrun litgpt pretrain {options}" > $FILENAME' #note ending quote
        )

        sbatch_options = (
            f'sbatch '
            f'--job-name={job_name} '
            f'--partition=seas_gpu,serial_requeue,gpu_requeue '
            f'--nodes={nodes} '
            f'--gres=gpu:nvidia_a100-sxm4-80gb:2 '
            f'--ntasks-per-node={ntasks_per_node} '
            f'--cpus-per-task={cpus_per_task} '
            f'--time=0-0{time_hrs}:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'--exclude=holygpu8a31505,holygpu8a25306,holygpu8a25104 '
            f'$FILENAME'
        )

    # run two commands above in order; remove temp file later
    full_cmd = f'{main_command} ; {sbatch_options}'
    
    os.system(full_cmd)



def run_exp00_pretrain_pythia():

    single_args = {
        'config': ['config_hub/custom_configs/pretrain/pythia-14m.yaml']
    }

    options_str, name_str = dict2options(single_args)

    for (opt, name) in zip(options_str, name_str):

        setup_exp00_pretrain_pythia(options=opt, 
                                    job_name=name, 
                                    log_file=f'exp00_pretrain_pythia/log_{name}', 
                                    device='gpu_two_a100_80gb') #use default values defined by exp00_setup() for other sbatch options

        print(f'job_name = {name}, options = {opt}')  




def setup_exp01_prepare_data(line_in_main_command: str,
                             job_name: str,
                             log_file: str,
                             device: str,
                             nodes: str = '1',
                             ntasks: str = '1',
                             cpus_per_task: str = '48',
                             time_hrs: str = '8',
                             memory_gb: str = '64') -> None:

    # remove existing log file
    os.system('rm -f ' + log_file)

    #if log file has a folder, check if folder exists (if not, create folder)
    log_folder = os.path.split(log_file)[0]
    if log_folder != '':
        if not os.path.isdir(f'logs/{log_folder}'):
            os.makedirs(f'logs/{log_folder}')
    

    if device=='seas_cpu':
        # this is the bash script: evolm/pretrain/lit-trainer/jobs/prepare_data/prepare_fineweb.sh or prepare_finemath.sh
        main_command = f"""FILENAME=$(mktemp) ; \
            echo "#!/bin/sh
            export TMPDIR="/n/netscratch/doshi-velez_lab/Everyone/tmp"
            export DATA_OPTIMIZER_CACHE_FOLDER="/n/netscratch/doshi-velez_lab/Everyone/tmp_data_optimizer_cache"
            module load python
            mamba activate litgpt-e
            {line_in_main_command}" > $FILENAME"""
            #note starting quote in echo line
            #note ending quote in last line

        sbatch_options = (
            f'sbatch '
            f'--job-name={job_name} '
            f'--partition=seas_compute '
            f'--nodes={nodes} '
            f'--ntasks={ntasks} '
            f'--cpus-per-task={cpus_per_task} '
            f'--time=0-0{time_hrs}:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'--exclude=holygpu8a31505,holygpu8a25306,holygpu8a25104 '
            f'$FILENAME'
        )

    # run two commands above in order; remove temp file later
    full_cmd = f'{main_command} ; {sbatch_options}'
    
    os.system(full_cmd)




def run_exp01_prepare_data():

    #prepare fineweb data
    line_in_main_command = f"""python litgpt/scripts/prepare_fineweb.py \
        /n/netscratch/doshi-velez_lab/Everyone/fineweb/sample/350BT \
        /n/netscratch/doshi-velez_lab/Everyone/fineweb_litgpt/pretrain/train \
        /n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/meta-llama/Llama-2-7b-hf """
    job_name='prepare_fineweb'
    log_file='exp01a_prepare_fineweb/log_prepare_fineweb'


    #prepare finemath data
    #need to set up arguments

    setup_exp01_prepare_data(line_in_main_command=line_in_main_command,
                             job_name=job_name,
                             log_file=log_file,
                             device='seas_cpu',) #use default values defined by setup_x() function for other sbatch options

    print(f'job_name = {job_name}')  


    

#exp02 -- 4 gpus
#pretrain llama models -- 0.5B-10BT and and 1B-20BT (fineweb dataset)
def setup_exp02_pretrain_llama(options: str,
                               job_name: str,
                               log_file: str,
                               device: str = 'gpu_four_a100',
                               nodes: str = '1',
                               n_gpus: str = '4', #n_gpus must equal n_tasks_per_node when nodes=1
                               cpus_per_task: str = '16',
                               time_days: str = '7',
                               memory_gb: str = '64') -> None:

    # remove existing log file
    os.system('rm -f ' + log_file)

    #if log file has a folder, check if folder exists (if not, create folder)
    log_folder = os.path.split(log_file)[0]
    if log_folder != '':
        if not os.path.isdir(f'logs/{log_folder}'):
            os.makedirs(f'logs/{log_folder}')
    
    ### note
    #bash script: must use srun + don't set CUDA_VISIBLE_DEVICES inside (slurm does this on its own when using srun)
    #sbatch: wnen nodes=1, --ntasks-per-node equal to # of gpus in --gres=gpu

    if device=='gpu_four_a100':
        # bash script
        main_command = (
            f'FILENAME=$(mktemp) ; '
            f'echo "#!/bin/sh' #note starting quote
            f'\nexport WANDB_API_KEY=b10df87569c5fdcef6d7b86acf29819b378fe28d'
            f'\nexport WANDB_ENTITY=th789-harvard'
            f'\nexport WANDB_PROJECT=overtraining'
            f'\nmodule load python'
            f'\nmamba activate litgpt-e'
            f'\nsource /n/home07/than157/desktop/done-large_projects/learn-better/load_private_vars.sh' 
            f'\nsrun litgpt pretrain {options}" > $FILENAME' #note ending quote
        )

        #request n a100 gpus (4 or 8 gpus)
        sbatch_options = (
            f'sbatch '
            f'--job-name={job_name} '
            f'--partition=seas_gpu '
            f'--nodes={nodes} '
            f'--gres=gpu:nvidia_a100-sxm4-80gb:{n_gpus} '
            f'--ntasks-per-node={n_gpus} '
            f'--cpus-per-task={cpus_per_task} '
            f'--time={time_days}-00:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'--exclude=holygpu8a31505,holygpu8a25306,holygpu8a25104 '
            f'$FILENAME'
        )

    # run two commands above in order; remove temp file later
    full_cmd = f'{main_command} ; {sbatch_options}'
    
    os.system(full_cmd)



def run_exp02_pretrain_llama():

    # #0.5B models
    # single_args = {
    #     'config': [
    #         'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay0.1-seed42.yaml',
    #         'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay0.01-seed42.yaml',
    #         'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay0.001-seed42.yaml',
    #         'config_hub/custom_configs/pretrain/llama-0.5B-10BT-weightdecay0.0001-seed42.yaml'
    #                ]
    # }

    #1B models
    single_args = {
        'config': [
            'config_hub/custom_configs/pretrain/llama-1B-10BT-weightdecay0.1-seed42.yaml',
            'config_hub/custom_configs/pretrain/llama-1B-10BT-weightdecay0.01-seed42.yaml',
            'config_hub/custom_configs/pretrain/llama-1B-10BT-weightdecay0.001-seed42.yaml',
            'config_hub/custom_configs/pretrain/llama-1B-10BT-weightdecay0.0001-seed42.yaml'
                   ]
    }

    options_str, name_str = dict2options(single_args)

    for (opt, name) in zip(options_str, name_str):

        setup_exp02_pretrain_llama(options=opt, 
                                   job_name=name, 
                                   log_file=f'exp02_pretrain_llama/log_{name}',
                                #    nodes='1', n_gpus='4', time_days='7', memory_gb='64' #0.5B models
                                   nodes='2', n_gpus='4', time_days='7', memory_gb='128' #1B models
                                   ) #use default values defined by exp00_setup() for other sbatch options

        print(f'job_name = {name}, options = {opt}')  





if __name__ == "__main__":
    # run_exp00_pretrain_pythia()
    # run_exp01_prepare_data()
    run_exp02_pretrain_llama()

    ##OLD from med=llm
    # run_exp01_prompt_models()
    # run_exp04_finetuning()
    # run_exp06_prompt_ft_models() #uses exp01 script










