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



def run_bash_script_written_in_terminal(bash_script: str, 
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
                               ):


    """
    Notes
    - bash_script is written in the terminal, provided full bash script as a string
    - when writing bash_script in the terminal
        - use $FILENAME to indicate the temporary file that will be created by the bash script
        - use heredoc to contain the file --> sbatch runs on a new line after the heredoc,  see example bash script in run_exp_fas_lit_traininer.py, run_exp05...() -- (causes EOF ; sbatch syntax error)
    """

    ### handle log file
    # remove existing log file
    os.system('rm -f ' + log_file)

    #if log file has a folder, check if folder exists (if not, create folder)
    log_folder = os.path.split(log_file)[0]
    if log_folder != '':
        if not os.path.isdir(f'logs/{log_folder}'):
            os.makedirs(f'logs/{log_folder}')
    
    ### bash script is written in each experiment's run_expXX_XXX() function -- writes a file called FILENAME
    
    ### sbatch options
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

    sbatch_options = "sbatch " + " ".join(flags)

    ### run full command, submit job with sbatch on a new line after the heredoc
    full_cmd = f"{bash_script}\n{sbatch_options} $FILENAME"
    os.system(full_cmd)




def run_bash_script_provided(bash_script: str, 
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
    """
    Notes
    - bash_script is already written, provide path to .sh file as a string
    """

    ### handle log file
    # remove existing log file
    os.system('rm -f ' + log_file)

    #if log file has a folder, check if folder exists (if not, create folder)
    log_folder = os.path.split(log_file)[0]
    if log_folder != '':
        if not os.path.isdir(f'logs/{log_folder}'):
            os.makedirs(f'logs/{log_folder}')
    
    ### bash script path is provided
    
    ### sbatch options
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

    sbatch_options = "sbatch " + " ".join(flags)

    ### run full command, submit job with sbatch on a new line after the heredoc
    full_cmd = f"{sbatch_options} {bash_script}"
    os.system(full_cmd)




#exp01 -- evaluate 0.5B-10BT and 1B-20BT llama models that were pretrained on fineweb, then finetuned on metamathqa
def run_exp01_eval_cot():
    sft_dataset = "medmcqa" #options: ["metamathqa", "hellaswag", "medmcqa"]
    # 0.5B models
    # model_dirs=[
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.0001-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.001-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.01-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.1-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.5-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay1.0-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay1.5-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay3.0-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay10.0-seed42-{sft_dataset}",
    # ]

    # 1B models
    # model_dirs=[
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.0001-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.001-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.01-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.1-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.5-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay1.0-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay1.5-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay3.0-seed42-{sft_dataset}",
    #     # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay10.0-seed42-{sft_dataset}",
    # ]

    datasets = [
        "medmcqa",
    ]

    # datasets = [
    #     "GSM8KPlatinum",
    #     "MATHLevel1",
    #     "MATHLevel2",
    #     "MATHLevel3",
    #     "MATHLevel4",
    #     "MATHHard",
    #     "CRUXEval",
    #     "BoardgameQA500",
    #     "TabMWP",
    #     "StrategyQA500",
    # ]



    for model_dir, dataset in product(model_dirs, datasets):

        out_root = "eval_output"
        bash_script = f"jobs/custom/myllama/eval-single-model--run-exp.sh {model_dir} {dataset} {out_root}"

        job_name = "eval_" + os.path.basename(model_dir) + "_" + dataset

        run_bash_script_provided(
            bash_script=bash_script,
            job_name=job_name,
            log_file=f"exp01_eval_ft_models/log_{job_name}",
            partition='seas_gpu,gpu,gpu_requeue,serial_requeue',
            n_gpus_a100_80gb='1', time_hrs='20', memory_gb='64', #!!! change settings in bash script depending on model size
            # partition='gpu_test',
            # n_gpus_any='1', time_mins='30', memory_gb='64',
            # dependency_type_and_job_id='afterok:52306109',
            )

        print(f'job_name = {job_name}')  





def run_exp02_eval_cot_ffw_models():

    ##### expB: models pretrained on finefineweb dataset, my subsets
    percent_doi = 0.1 #options: [0.1, 0.05, 0.01, 0.005, 0.001]

    model_dirs=[
        f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.0001-seed42-metamathqa",
        f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.001-seed42-metamathqa",
        # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.01-seed42-metamathqa",
        f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay0.1-seed42-metamathqa", 
        f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay1.0-seed42-metamathqa",        
        ]


    ##### datasets -- used for both expA and expB
    datasets = [
        "GSM8KPlatinum",
        "MATHLevel1",
        "MATHLevel2",
        "MATHLevel3",
        "MATHLevel4",
        "MATHHard",
        "CRUXEval",
        "BoardgameQA500",
        "TabMWP",
        "StrategyQA500",
    ]


    for model_dir, dataset in product(model_dirs, datasets):

        out_root = "eval_output/ffw"
        bash_script = f"jobs/custom/myllama/eval-single-model--run-exp.sh {model_dir} {dataset} {out_root}"

        job_name = "eval_" + os.path.basename(model_dir) + "_" + dataset

        run_bash_script_provided(
            bash_script=bash_script,
            job_name=job_name,
            log_file=f"exp02_eval_ft_models_pretrained_on_ffw/log_{job_name}",
            partition='seas_gpu,gpu,gpu_requeue,serial_requeue',
            n_gpus_a100_80gb='1', time_hrs='2', memory_gb='64'
            )

        print(f'job_name = {job_name}')  



def run_exp03_eval_cot_models_vary_wd_during_ft():

    ##### expC: start with pretrained models from exp01, vary wd during ft
    wd_during_ft = 0.01 #options: [1.0, 0.1, 0.01]

    model_dirs=[
        # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/vary_wd_during_ft/llama-1B-20BT-weightdecay0.0001-seed42-metamathqa-ftweightdecay{wd_during_ft}",      
        # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/vary_wd_during_ft/llama-1B-20BT-weightdecay0.001-seed42-metamathqa-ftweightdecay{wd_during_ft}",      
        # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/vary_wd_during_ft/llama-1B-20BT-weightdecay0.01-seed42-metamathqa-ftweightdecay{wd_during_ft}",      
        # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/vary_wd_during_ft/llama-1B-20BT-weightdecay0.1-seed42-metamathqa-ftweightdecay{wd_during_ft}",      
        # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/vary_wd_during_ft/llama-1B-20BT-weightdecay1.0-seed42-metamathqa-ftweightdecay{wd_during_ft}",      
        # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/vary_wd_during_ft/llama-1B-20BT-weightdecay3.0-seed42-metamathqa-ftweightdecay{wd_during_ft}",      
        # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/vary_wd_during_ft/llama-1B-20BT-weightdecay10.0-seed42-metamathqa-ftweightdecay{wd_during_ft}",      
        # f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/vary_wd_during_ft/llama-1B-20BT-weightdecay1.5-seed42-metamathqa-ftweightdecay{wd_during_ft}",     
        f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/vary_wd_during_ft/llama-1B-20BT-weightdecay0.5-seed42-metamathqa-ftweightdecay{wd_during_ft}",       
        ]


    ##### datasets -- used for both expA and expB
    datasets = [
        "GSM8KPlatinum",
        "MATHLevel1",
        "MATHLevel2",
        "MATHLevel3",
        "MATHLevel4",
        "MATHHard",
        "CRUXEval",
        "BoardgameQA500",
        "TabMWP",
        "StrategyQA500",
    ]


    for model_dir, dataset in product(model_dirs, datasets):

        out_root = "eval_output/vary_wd_during_ft"
        bash_script = f"jobs/custom/myllama/eval-single-model--run-exp.sh {model_dir} {dataset} {out_root}"

        job_name = "eval_" + os.path.basename(model_dir) + "_" + dataset

        run_bash_script_provided(
            bash_script=bash_script,
            job_name=job_name,
            log_file=f"exp03_eval_cot_models_vary_wd_during_ft/log_{job_name}",
            partition='seas_gpu,gpu,gpu_requeue,serial_requeue',
            n_gpus_a100_80gb='1', time_hrs='2', memory_gb='64' #change settings in bash script for 1B model size
            )

        print(f'job_name = {job_name}')  


if __name__ == "__main__":
    
    run_exp01_eval_cot()
    # run_exp02_eval_cot_ffw_models()
    # run_exp03_eval_cot_models_vary_wd_during_ft()







