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
            f'--partition=seas_gpu '
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
            f'--time={time_days}-0{time_hrs}:00 '
            f'--mem={memory_gb}gb '
            f'--error=logs/{log_file}_jobid%j '
            f'--output=logs/{log_file}_jobid%j '
            f'$FILENAME'
        )

    # run two commands above in order; remove temp file later
    full_cmd = f'{bash_script} ; {sbatch_options}'
    
    os.system(full_cmd)





def run_exp01_subset_finefineweb():
    
    # bash script
    def create_bash_script():
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nmodule load python'
        f'\nmamba activate process_ffw'
        f'\npython -u -m evolm.common.scripts.ffw_subset_dataset" > $FILENAME' #note ending quote
        )
        return bash_script_complete

    #run script single_args
    bash_script = create_bash_script()

    job_name = 'subset_ffw'
    run_bash_script(bash_script=bash_script, 
                    job_name=job_name,
                    log_file=f'exp01_subset_ffw/log_{job_name}',
                    device='seas_cpu',
                    n_nodes='1', time_days='2', memory_gb='128'
                    )

    print(f'job_name = {job_name}')  



def run_exp02_count_tokens_for_ffwsample_domains():
    
    # bash script
    def create_bash_script(domain):
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nmodule load python'
        f'\nmamba activate process_ffw'
        f'\npython -u -m evolm.common.scripts.subset_finefineweb.prepffwa_count_domain_llama2tokens --domain {domain}" > $FILENAME' #note ending quote
        )
        return bash_script_complete
    
    #single args
    # domains_all = [
    #     'aerospace', 'agronomy', 'artistic', 'astronomy', 'atmospheric_science', 
    #     'automotive', 'beauty', 'biology', 'celebrity', 'chemistry', 
    #     'christianity', 'civil_engineering', 'communication_engineering', 'computer_science_and_technology', 'design',
    #     'drama_and_film', 'economics', 'electronic_science', 'entertainment', 'environmental_science',
    #     'fashion', 'finance', 'food', 'gamble', 'game',
    #     'geography', 'health', 'history', 'hobby', 'hydraulic_engineering',
    #     'instrument_science', 'journalism_and_media_communication', 'landscape_architecture', 'law', 'library',
    #     'literature', 'materials_science', 'mathematics', 'mechanical_engineering', 'medical',
    #     'mining_engineering', 'movie', 'music_and_dance', 'news', 'nuclear_science', 
    #     'ocean_science', 'optical_engineering', 'painting', 'pet', 'petroleum_and_natural_gas_engineering',
    #     'philosophy', 'photo', 'physics', 'politics', 'psychology',
    #     'public_administration', 'relationship', 'sociology', 'sports', 'statistics',
    #     'systems_science', 'textile_science', 'topicality', 'transportation_engineering', 'travel',
    #     'urban_planning', 'weapons_science'
    #     ]
    domains = ['mathematics']

    #run each experiment, which has a different value for the domain argument
    for domain in domains:
        bash_script = create_bash_script(domain)

        job_name = f'count_tokens_{domain}'
        run_bash_script(bash_script=bash_script, 
                        job_name=job_name,
                        log_file=f'exp02_count_tokens_for_finefinewebsample_domains/log_{job_name}',
                        device='seas_cpu',
                        n_nodes='1', time_days='0', time_hrs='10', memory_gb='32'
                        )

        print(f'job_name = {job_name}')  



def run_exp03_create_my_subset():
    
    # bash script
    def create_bash_script(domain, percent_doi):
        bash_script_complete = (
        f'FILENAME=$(mktemp) ; '
        f'echo "#!/bin/sh' #note starting quote
        f'\nmodule load python'
        f'\nmamba activate process_ffw'
        f'\npython -u -m evolm.common.scripts.subset_finefineweb.prepffwc_subset_domain_random_files --domain {domain} --percent_doi {percent_doi}" > $FILENAME' #note ending quote
        )
        return bash_script_complete
    
    #single args -- full
    # domains_all = [
    #     'aerospace', 'agronomy', 'artistic', 'astronomy', 'atmospheric_science', 
    #     'automotive', 'beauty', 'biology', 'celebrity', 'chemistry', 
    #     'christianity', 'civil_engineering', 'communication_engineering', 'computer_science_and_technology', 'design',
    #     'drama_and_film', 'economics', 'electronic_science', 'entertainment', 'environmental_science',
    #     'fashion', 'finance', 'food', 'gamble', 'game',
    #     'geography', 'health', 'history', 'hobby', 'hydraulic_engineering',
    #     'instrument_science', 'journalism_and_media_communication', 'landscape_architecture', 'law', 'library',
    #     'literature', 'materials_science', 'mathematics', 'mechanical_engineering', 'medical',
    #     'mining_engineering', 'movie', 'music_and_dance', 'news', 'nuclear_science', 
    #     'ocean_science', 'optical_engineering', 'painting', 'pet', 'petroleum_and_natural_gas_engineering',
    #     'philosophy', 'photo', 'physics', 'politics', 'psychology',
    #     'public_administration', 'relationship', 'sociology', 'sports', 'statistics',
    #     'systems_science', 'textile_science', 'topicality', 'transportation_engineering', 'travel',
    #     'urban_planning', 'weapons_science'
    #     ]
    # percent_dois_all = [0.1, 0.05, 0.01, 0.005, 0.001]


    #subset of args
    domains = [
        'aerospace', 'agronomy', 'artistic', 'astronomy', 'atmospheric_science', 
        'automotive', 'beauty', 'biology', 'celebrity', 'chemistry', 
        'christianity', 'civil_engineering', 'communication_engineering', 'computer_science_and_technology', 'design',
        'drama_and_film', 'economics', 'electronic_science', 'entertainment', 'environmental_science',
        'fashion', 'finance', 'food', 'gamble', 'game',
        'geography', 'health', 'history', 'hobby', 'hydraulic_engineering',
        'instrument_science', 'journalism_and_media_communication', 'landscape_architecture', 'law', 'library',
        'literature', 'materials_science', 'mathematics', 'mechanical_engineering', 'medical',
        'mining_engineering', 'movie', 'music_and_dance', 'news', 'nuclear_science', 
        'ocean_science', 'optical_engineering', 'painting', 'pet', 'petroleum_and_natural_gas_engineering',
        'philosophy', 'photo', 'physics', 'politics', 'psychology',
        'public_administration', 'relationship', 'sociology', 'sports', 'statistics',
        'systems_science', 'textile_science', 'topicality', 'transportation_engineering', 'travel',
        'urban_planning', 'weapons_science'
        ]

    percent_dois = [0.01, 0.005, 0.001] #full list: [0.1, 0.05, 0.01, 0.005, 0.001]


    #arg combinations
    arg_combos = list[tuple[str, float, int, str, str]](product(domains, percent_dois))
    #run each experiment, which has a different value for the domain argument
    for arg_combo in arg_combos:
        domain, percent_doi = arg_combo

        bash_script = create_bash_script(domain, percent_doi)

        job_name = f'subset_{domain}_mathematics{percent_doi}'
        run_bash_script(bash_script=bash_script, 
                        job_name=job_name,
                        log_file=f'run_exp03_create_my_subset/mathematics{percent_doi}/log_{job_name}',
                        device='seas_cpu',
                        n_nodes='1', time_days='0', time_hrs='1', memory_gb='32'
                        )

        print(f'job_name = {job_name}')  



if __name__ == "__main__":
    # run_exp01_subset_finefineweb() --> bad, takes too long to run + based on gpt2 tokens
    # run_exp02_count_tokens_for_ffwsample_domains()
    # run_exp03_create_my_subset()









