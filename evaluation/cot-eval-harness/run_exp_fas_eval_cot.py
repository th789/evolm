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
    nodes_to_exclude = 'holygpu8a27301'  #nodes to exclude from job scheduling, ran into errors    

    
    if device=='n_gpus_a100_sxm4_80gb':
        sbatch_options = (
            f'sbatch '
            f'--job-name={job_name} '
            f'--partition=seas_gpu,serial_requeue,gpu_requeue '
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



    

#exp01 -- finetune llama models -- finetune 0.5B-10BT and 1B-20BT (pretrained on fineweb dataset) on metamathqa dataset
#note
#sbatch: 4 GPUs (1 node): wnen nodes=1, ntasks-per-node must equal n_gpus --> so n_gpus = ntasks-per-node = 4
def run_exp01_eval_cot():
    
    # 0.5B models
    # model_dirs=[
    #     "/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.0001-seed42-metamathqa",
    #     "/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.001-seed42-metamathqa",
    #     "/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.01-seed42-metamathqa",
    #     "/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.1-seed42-metamathqa",
    #     "/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay1.0-seed42-metamathqa",
    # ]


    # 1B models
    model_dirs=[
        "/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.0001-seed42-metamathqa",
        # "/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.001-seed42-metamathqa",
        # "/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.01-seed42-metamathqa",
        # "/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.1-seed42-metamathqa",
        # "/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay1.0-seed42-metamathqa",
    ]

    datasets = [
        # "MATHLevel1",
        # "MATHLevel2",
        # "MATHLevel3",
        # "MATHLevel4",
        "MATHHard",
        # "CRUXEval",
        # "BoardgameQA500",
        # "TabMWP",
        # "StrategyQA500",
    ]


    combinations = product(model_dirs, datasets)

    for model_dir, dataset in combinations:

        bash_script = f"jobs/custom/myllama/eval-single-model--run-exp.sh {model_dir} {dataset}"

        job_name = os.path.basename(model_dir) + "_" + dataset
        log_file = f"exp01_eval_ft_models/log_{job_name}"
        n_gpus = "1"
        time_days = "0"
        time_hrs = "2"
        memory_gb = "64"
        nodes_to_exclude = "holygpu8a27301"


        # remove existing log file
        os.system('rm -f ' + log_file)

        #if log file has a folder, check if folder exists (if not, create folder)
        log_folder = os.path.split(log_file)[0]
        if log_folder != '':
            if not os.path.isdir(f'logs/{log_folder}'):
                os.makedirs(f'logs/{log_folder}')

        sbatch_str = (
            f'sbatch '
            f'--job-name={job_name} '
            # f'--partition=seas_gpu,serial_requeue,gpu_requeue,gpu '
            # f'--gres=gpu:nvidia_a100-sxm4-80gb:{n_gpus} '
            f'--partition=gpu_test '
            f'--gres=gpu:{n_gpus} '
            f'--time={time_days}-0{time_hrs}:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'--exclude={nodes_to_exclude} '
            f'{bash_script}'
        )

        # full_cmd = f'{bash_script} ; {sbatch_options}'
        
        os.system(sbatch_str)

        print(f'job_name = {job_name}')  









if __name__ == "__main__":
    
    run_exp01_eval_cot()







