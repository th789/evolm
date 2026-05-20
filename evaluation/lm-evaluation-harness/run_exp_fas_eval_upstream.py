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









def run_exp01_eval_many_models_all_tasks():

    ###change model names in the bash script

    bash_script = f"jobs/eval-multiple--run-exp.sh"

    ###expo01
    # job_name = "eval_model_multiple_exp01_pt_1B" #pretrained models -- change model name in the bash script
    # job_name = "eval_model_multiple_exp01_ft_1B" #finetuned models -- change model name in the bash script

    ###exp03
    #pretrained models -- change model name in the bash script
    job_name = "eval_model_multiple_exp02_pt_1B_percentdoi0.001" 
    # job_name = "eval_model_multiple_exp02_pt_1B_percentdoi0.005" 
    # job_name = "eval_model_multiple_exp02_pt_1B_percentdoi0.01" 
    # job_name = "eval_model_multiple_exp02_pt_1B_percentdoi0.05" 
    # job_name = "eval_model_multiple_exp02_pt_1B_percentdoi0.1"
    #finetuned models -- change model name in the bash script
    # job_name = "eval_model_multiple_exp02_ft_1B_percentdoi0.001" 
    # job_name = "eval_model_multiple_exp02_ft_1B_percentdoi0.005" 
    # job_name = "eval_model_multiple_exp02_ft_1B_percentdoi0.01" 
    # job_name = "eval_model_multiple_exp02_ft_1B_percentdoi0.05" 
    # job_name = "eval_model_multiple_exp02_ft_1B_percentdoi0.1" 

    run_bash_script_provided(
        bash_script=bash_script,
        job_name=job_name,
        log_file=f"exp01_eval_models_multiple/log_{job_name}",
        # partition='gpu_test', n_gpus_any='1', time_hrs='1', memory_gb='32', #1B models
        partition='seas_gpu,gpu,gpu_requeue,serial_requeue', n_gpus_a100_80gb='1', time_hrs='2', memory_gb='32', #1B models      
        )
        
        #1B models: run in under 10 min

    print(f'job_name = {job_name}')  


import time


def run_exp01_eval_single_model_single_task():

    ##### expA: models pretrained on fineweb dataset

    # model_size = "0.5B-10BT"
    # wd_lst = [0.5, 1.5] #full: [0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 0.5, 1.5]
    # task_lst = ["hellaswag", "winogrande", "piqa", "openbookqa", "arc_easy", "arc_challenge", "mathqa"]  #full: ["hellaswag", "winogrande", "piqa", "openbookqa", "arc_easy", "arc_challenge", "mathqa"]

    counter = 0
    for wd, task in product(wd_lst, task_lst):
        ### select model directory
        #pretrained
        # model_dir = f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/llama-{model_size}-weightdecay{wd}-seed42/final-hf"
        #finetuned on metamathqa
        # model_dir = f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-{model_size}-weightdecay{wd}-seed42-metamathqa"
        # #finetuned on hellaswag
        # model_dir = f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-{model_size}-weightdecay{wd}-seed42-hellaswag"

        ### define bash script arguments
        bash_script = f"jobs/eval-single-model-and-task--run-exp.sh {model_dir} {task}"

        ### create job name
        if "evolm/pretrain/lit-trainer/models" in model_dir:
            job_name = "eval_pt_" + os.path.basename(os.path.dirname(model_dir)) + "_" + task #model name is second to last folder in model_dir
        elif "evolm/finetune/llama-factory/llamafactory_out" in model_dir:
            job_name = "eval_ft_" + os.path.basename(model_dir) + "_" + task #model name is last folder in model_dir

        run_bash_script_provided(
            bash_script=bash_script,
            job_name=job_name,
            log_file=f"exp01_eval_single_model_single_task/log_{job_name}",
            partition='seas_gpu,gpu,gpu_requeue,serial_requeue',
            n_gpus_any='1', time_hrs='1', memory_gb='32'
            )

        print(f'job_name = {job_name}')  
        
        ### if using gpu_test -- can only submit 2 jobs at a time
        # counter += 1
        # if counter % 2 == 0:
        #     #wait for 5min
        #     print(f'Counter = {counter}, waiting for 5min...')
        #     time.sleep(300)



#exp01 -- evaluate 0.5B-10BT and 1B-20BT llama models that were pretrained on fineweb, then finetuned on metamathqa
def run_exp01_eval_single_model_all_tasks():

    #--------------------------- OLD RUNS --------------------------------

    # ##### exp01: models pretrained on fineweb dataset
    # model_sizes = ["1B-10BT"] #["0.5B-10BT", "1B-20BT"]
    # wd_lst = [1.5] #[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0]
    

    #pretrained models
    # model_dirs=[
    #     f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/llama-{model_size}-weightdecay{wd}-seed42/final-hf" for model_size, wd in product(model_sizes, wd_lst)
    # ]

    # # #finetuned models
    # model_dirs=[
    #     f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-{model_size}-weightdecay{wd}-seed42-metamathqa" for model_size, wd in product(model_sizes, wd_lst)
    # ]


    ##### exp02: models pretrained on finefineweb dataset, my subsets
    # wd_lst = [0.0001, 0.001, 0.01, 0.1, 1.0]
    # percent_doi_lst = [0.001, 0.005, 0.01, 0.05, 0.1]
    # wd_lst = [1.0]
    # percent_doi_lst = [0.005]

    #pretrained models
    # model_dirs=[
    #     f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay{wd}-seed42/final-hf" for wd, percent_doi in product(wd_lst, percent_doi_lst)
    # ]

    # #finetuned models
    # model_dirs=[
    #     f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics{percent_doi}-weightdecay{wd}-seed42-metamathqa" for wd, percent_doi in product(wd_lst, percent_doi_lst)
    # ]


    # ##### exp03: models finetuned with different wd -- only ft models
    # wd_ft_lst = [1.0] #[1.0, 0.1, 0.01]
    # wd_pt_lst = [1.5] #[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0]
    # model_dirs=[
    #     f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/vary_wd_during_ft/llama-1B-20BT-weightdecay{wd_pt}-seed42-metamathqa-ftweightdecay{wd_ft}" for wd_pt, wd_ft in product(wd_pt_lst, wd_ft_lst)
    # ]

    #---------------------------------------------------------------------------------

    ### plasticity-stability experiments

    model_dict = {
        # "llama-4B-80BT": [0.1],
        # "olmo-1B-210BT": [0.1],
        #full wds below
        # "llama-0.5B-10BT": [0.0], #[0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 3.0, 10.0], #[0.1, 1.0, 3.0, 10.0], #[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 3.0, 10.0],
        # "llama-1B-20BT": [0.0], #[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 3.0, 10.0],
        # "llama-4B-80BT": [0.0], #[0.1, 1.0],
        # "olmo-1B-30BT": [0.6], #[0.1, 0.3, 0.6, 1.0],
        "olmo-1B-210BT": [0.0], #[0.1, 0.3, 1.0],
    }
    

    # ## pretrained models
    for model_name, wd_lst in model_dict.items():
        if model_name in ["llama-0.5B-10BT", "llama-1B-20BT"]:
            model_dirs = [f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/{model_name}-weightdecay{wd}-seed42/final-hf" for wd in wd_lst]
        elif model_name in ["llama-4B-80BT"]:
            model_dirs = []
            for wd in wd_lst:
                if wd == 0.1:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/zhenting/myllama-4B-80BT")
                elif wd == 1.0:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/hlzhang109/llama-4B-80BT-weightdecay1.0-seed42")
                elif wd == 0.0:
                    model_dirs.append("/n/netscratch/doshi-velez_lab/Everyone/models/pretrained/llama-4B-80BT-weightdecay0.0-seed42/final-hf")
        elif model_name in ["olmo-1B-30BT"]:
            model_dirs = []
            for wd in wd_lst:
                if wd == 0.1:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/sbordt/OLMo-2-1B-1x")
                elif wd == 0.3:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/sbordt/OLMo-2-1B-1x-WD03")
                elif wd == 0.6:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/sbordt/OLMo-2-1B-1x-WD06")
                elif wd == 1.0:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/sbordt/OLMo-2-1B-1x-WD1")
                elif wd == 0.0:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/sbordt/OLMo-2-1B-1x-WD0")
        elif model_name in ["olmo-1B-210BT"]:
            model_dirs = []
            for wd in wd_lst:
                if wd == 0.0:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/sbordt/OLMo-2-1B-7x-WD0")
                elif wd == 0.1:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/sbordt/OLMo-2-1B-Decayed-Early")
                    # model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/sbordt/OLMo-2-1B-7x")
                elif wd == 0.3:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/sbordt/OLMo-2-1B-7x-WD03")
                elif wd == 1.0:
                    model_dirs.append("/n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/sbordt/OLMo-2-1B-WD1")
        else:
            raise ValueError(f"Model name {model_name} not found")

    ### finetuned models
    # sft_dataset = "simplescaling" #["metamathqa", "medmcqa", "pubmedqa", "mmluprocot", "race", "simplescaling"]
    # for model_name, wd_lst in model_dict.items():
    #     if model_name in ["llama-0.5B-10BT", "llama-1B-20BT", "llama-4B-80BT"]:
    #         model_dirs = [f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/{model_name}-weightdecay{wd}-seed42-{sft_dataset}" for wd in wd_lst]
    #     elif model_name in ["olmo-1B-30BT", "olmo-1B-210BT"]:
    #         model_dirs = [f"/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/{model_name}-weightdecay{wd}-{sft_dataset}" for wd in wd_lst]


    for model_dir in model_dirs:

        bash_script = f"jobs/eval-single--run-exp.sh {model_dir}"

        #pretrained models
        if "evolm/pretrain/lit-trainer/models" in model_dir:
            job_name = "eval_pt_" + os.path.basename(os.path.dirname(model_dir)) #model name is second to last folder in model_dir
        elif "/n/netscratch/" in model_dir:
            job_name = "eval_pt_" + os.path.basename(os.path.dirname(model_dir)) #model name is second to last folder in model_dir
        elif "evolm/models/hf_ckpts" in model_dir:
            job_name = "eval_pt_" + os.path.basename(model_dir) #model name is second to last folder in model_dir
        
        #finetuned models
        elif "evolm/finetune/llama-factory/llamafactory_out" in model_dir:
            job_name = "eval_ft_" + os.path.basename(model_dir) #model name is last folder in model_dir

        run_bash_script_provided(
            bash_script=bash_script,
            job_name=job_name,
            log_file=f"exp01_eval_single_model_all_tasks/log_{job_name}",
            partition='seas_gpu,gpu,gpu_requeue,gpu_h200', n_gpus_any='1', time_hrs='1', memory_gb='32' #all PT + FT models
            # partition='gpu_test', n_gpus_any='1', time_hrs='1', memory_gb='32' #all PT + FT models
            )
        
        #1B models: run in under 10 min

        print(f'job_name = {job_name}')  



def run_exp02_eval_single_all_tasks__new_ft_tasks():

    ##### models
    model_lst = [
        # "llama-0.5B-10BT-weightdecay0.0-seed42",
        # "llama-0.5B-10BT-weightdecay0.0001-seed42",
        # "llama-0.5B-10BT-weightdecay0.001-seed42",
        # "llama-0.5B-10BT-weightdecay0.01-seed42",
        # "llama-0.5B-10BT-weightdecay0.1-seed42",
        # "llama-0.5B-10BT-weightdecay0.5-seed42",
        # "llama-0.5B-10BT-weightdecay1.0-seed42",
        # "llama-0.5B-10BT-weightdecay1.5-seed42",
        # "llama-0.5B-10BT-weightdecay3.0-seed42",
        # "llama-0.5B-10BT-weightdecay10.0-seed42",
        
        # "llama-1B-20BT-weightdecay0.0-seed42",
        # "llama-1B-20BT-weightdecay0.0001-seed42",
        # "llama-1B-20BT-weightdecay0.001-seed42",
        # "llama-1B-20BT-weightdecay0.01-seed42",
        # "llama-1B-20BT-weightdecay0.1-seed42",
        # "llama-1B-20BT-weightdecay0.5-seed42",
        # "llama-1B-20BT-weightdecay1.0-seed42",
        # "llama-1B-20BT-weightdecay1.5-seed42",
        # "llama-1B-20BT-weightdecay3.0-seed42",
        # "llama-1B-20BT-weightdecay10.0-seed42",

        # "llama-4B-80BT-weightdecay0.0-seed42",
        # "llama-4B-80BT-weightdecay0.1-seed42",
        # "llama-4B-80BT-weightdecay1.0-seed42",

        # "olmo-1B-30BT-weightdecay0.0",
        # "olmo-1B-30BT-weightdecay0.1",
        # "olmo-1B-30BT-weightdecay0.3",
        # "olmo-1B-30BT-weightdecay0.6",
        # "olmo-1B-30BT-weightdecay1.0",

        "olmo-1B-210BT-weightdecay0.0",
        # "olmo-1B-210BT-weightdecay0.1",
        # "olmo-1B-210BT-weightdecay0.1",
        # "olmo-1B-210BT-weightdecay0.3",
        # "olmo-1B-210BT-weightdecay1.0",
    ]

    #dataset/task that model was FT'd on
    task_lst = ['arc_easy'] #options: ['hellaswag', 'winogrande', 'piqa', 'arc_easy', 'arc_challenge']   #tasks are used to specify model (not eval tasks) -- for each model, evaluate on all upstream tasks (even though only need one, it's just easier)

    for model, task in product(model_lst, task_lst):        
        #define bash script arguments
        model_dir = f"/n/netscratch/doshi-velez_lab/Everyone/models/ft_new_tasks/{model}-{task}"
        bash_script = f"jobs/eval-single--run-exp.sh {model_dir}"
        
        job_name = f"eval_up_{model}_{task}"
        
        run_bash_script_provided(
            bash_script=bash_script,
            job_name=job_name,
            log_file=f"exp02_eval_single_model_all_tasks__new_ft_tasks/log_{job_name}",
            partition='seas_gpu,gpu,gpu_requeue,gpu_h200', n_gpus_any='1', time_hrs='1', memory_gb='32', #all ft models
            # partition='gpu_test', n_gpus_any='1', time_mins='30', memory_gb='32', #all ft models
            # dependency_type_and_job_id='afterok:2262241,2262242,2262243,2262255'
            )

        print(f'job_name = {job_name}')  


if __name__ == "__main__":
    #exp01
    # run_exp01_eval_single_model_single_task() #for exp01, 0.5B or 1B models -- one job: one model + one task
    # run_exp01_eval_many_models_all_tasks() #for exp01, 1B models -- one job: multiple models + all tasks

    # wait for 5min
    # time.sleep(700)

    #exp02 -- pick one
    # run_exp01_eval_many_models_all_tasks() #for exp02, 1B models -- one job: multiple models + all tasks -- not good because some evals fail and some succedd in same run
    # run_exp01_eval_single_model_all_tasks() #PT models (old: for exp02 + exp03, which only has 1B models)


    # exp02 -- FT model on hellaswag/piqa/winogrande/arc_easy/arc_challenge and eval on these respective datasets
    run_exp02_eval_single_all_tasks__new_ft_tasks()
