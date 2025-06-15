# Supervised Fine-tuning

We use `LLaMA-Factory` to fine-tune models.

## Installation

First create a new virtual environment and activate it:

```bash
conda create -n llamafactory python=3.10
conda activate llamafactory
```

Then install the package with all dependencies:

```bash
cd llama-factory
pip install -e ".[torch,metrics,deepspeed,bitsandbytes,vllm,wandb]"
```

These two steps should be enough to install the package and all dependencies. If there is any issue, please refer to the original `LLaMA-Factory` project at https://github.com/hiyouga/LLaMA-Factory.

## Step 1: Prepare Training Data

Modify `scripts/make_dataset/make_sft_dataset.py`  according to your needs. Then, run the script to format the dataset:
```bash
python scripts/make_dataset/make_sft_dataset.py
```

Make sure that the `json` file contains a list of dictionaries, where each dictionary has the following keys:
```json
{
    "instruction": ...,
    "input": ...,
    "output": ...,
}
```

After this, you should have the dataset ready for training. Next, in `data/dataset_info.json`, add a JSON dictionary as the configuration of this new dataset. For example, if you added "MetaMathQA", you should add the following configuration:
```json
"metamath": {
    "file_name": "/path/to/your/MetaMathQA.json"
},
```

## Step 2: Start Training

### Prepare `yaml` Configuration File
For example, if you are going to train `myllama-1B-160BT--cpt-full-bs1m-lr2e-4`, go to `overtraining/finetune/llama-factory/config_hub/custom_configs/myllama-1B-160BT--cpt-full-bs1m-lr2e-4.yaml`, and modify:
- `dataset`. This is the dataset you want to train on. The name you should put here is the key you added in `data/dataset_info.json`.
- `output_dir` according to your designated directory. This is the directory where the model checkpoints will be saved.
- (optional) `template`. This is the chat template according to which query and response will be placed. You can modify this according to your needs. All tempalte definitions are in `src/llamafactory/data/template.py`.

In terms of `wandb` logging, please make sure to set up correct key and project name beforehand on your local machine. For debug purpose, you can turn off `wandb` by setting `report_to` to `none` in the configuration file.

### Launch Training Job
Launch the training job by running:
```bash
FORCE_TORCHRUN=1 llamafactory-cli train config_hub/custom_configs/myllama-1B-160BT--cpt-full-bs1m-lr2e-4.yaml
```

> Note: If you encounter `RuntimeError: CUDA out of memory`, you can try reducing `per_device_train_batch_size` in the configuration file, while also increasing `gradient_accumulation_steps` accordingly to make sure that the total batch size remains the same.
