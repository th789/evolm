import shutil
from pathlib import Path
import subprocess

from datetime import datetime

import argparse


### parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()

model_path = args.model_path


now = datetime.now()
print('Start time: ', now.strftime("%Y-%m-%d %H:%M:%S"))
print(f'model_path = {model_path}')

#function to check size of folder -- run du -sh in terminal
def du_sh(path):
    result = subprocess.run(
        ["du", "-sh", path],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.split()[0]



def get_checkpoint_folders(base_path):
    return [p for p in Path(base_path).iterdir()
            if p.is_dir() and p.name.startswith("checkpoint-")]



ft_models_folder = "/n/netscratch/doshi-velez_lab/Everyone/models/sweep"
print('FT models folder, before: ', du_sh(ft_models_folder))
print('')



model_path = Path(model_path)
model_folder_name_short = model_path.parts[-1:][0]

#if model folder does not exist, stop this iteration here and continue to next iteration
if not model_path.exists():
    print(f"{model_folder_name_short} -- model folder does not exist")

#if model folder exists
elif model_path.exists():
    #get list of checkpoint folders
    ckpt_folder_lst = get_checkpoint_folders(model_path)

    # #if there are no checkpoint folders, stop this iteration here and continue to next iteration
    if len(ckpt_folder_lst) == 0:
        print(f"{model_folder_name_short} -- model folder exists, but there are no ckpt folders")
    
    #if there are checkpoint folders, delete them
    elif len(ckpt_folder_lst) >= 1:
        for ckpt_folder in ckpt_folder_lst:
            ckpt_folder_name_short = ckpt_folder.parts[-1:][0]
            #delete ckpt folder
            shutil.rmtree(ckpt_folder)
            print(f"{model_folder_name_short} -- {ckpt_folder_name_short} deleted")


print('\nFT models folder, after: ', du_sh(ft_models_folder))
print('Complete!')

now = datetime.now()
print('End time: ', now.strftime("%Y-%m-%d %H:%M:%S"))