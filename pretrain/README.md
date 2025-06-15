# Pre-training / Continued Pre-training

We use PyTorch Lightning's `litgpt` to do pre-training and continued pre-training.

## Installation

First create a new virtual environment and activate it:

```bash
conda create -n litgpt python=3.10
conda activate litgpt
```

Then install the package with all dependencies:

```bash
cd lit-trainer
pip install -e '.[all]'
```

These two steps should be enough to install the package and all dependencies. If there is any issue, please refer to the original `litgpt` project at https://github.com/Lightning-AI/litgpt.


## Step 1: Prepare Tokenizer 

In `overtraining/common/scripts/download_hf_model.py`, modify `model_id` to `"meta-llama/Llama-2-7b-hf"`, and `download_path` to your designated directory. Go to `overtraining/common/scripts`, and run:
```bash
python download_hf_model.py
```

## Step 2: Prepare Training Data 

As an example, we use `fineweb` for pre-training and `finemath` for continued pre-training.

First, download `fineweb` and `finemath` to local disk. Modify `local_dir` in `overtraining/common/scripts/download_hf_dataset.py` to your designated directory, uncomment the code for `fineweb` and `finemath`. Go to `overtraining/common/scripts`, and run:
```bash
python download_hf_dataset.py
```

Then, turn the data into `litgpt` format. Modify `ROOT_DIR` in `jobs/prepare_data/prepare_fineweb.sh` to your designated directory, and run:
```bash
bash jobs/prepare_data/prepare_fineweb.sh
bash jobs/prepare_data/prepare_finemath.sh
```

Next, in `litgpt/data/fineweb.py` and `litgpt/data/finemath.py`, modify `data_path` to your designated directory so that `litgpt` knows where to find the training data.

## Step 3: Start Training

### (Continued Pre-training Only) Prepare Pre-trained Model
For continued pretraining, you should first prepare a pre-trained model. For pretraining, you can skip this step.

Download pre-trained models from HuggingFace to local disk. In `jobs/download/download_myllama.sh`, modify `ckpt_dir` to your designated directory, and run:
```bash
bash jobs/download/download_myllama.sh
```

### Prepare `yaml` Configuration File
Go to `config_hub`and create your `xxx.yaml`, and modify the following three fields: `out_dir`, `initial_checkpoint_dir`, and `tokenizer_dir` according to your designated directories (see the comments in the file for more details). Please refer to provided `yaml` files under `config_hub/custom_configs` for examples.

In terms of `wandb` logging, please make sure to set up correct key and project name beforehand on your local machine. For debug purpose, you can turn off `wandb` by setting `logger_offline` to `true` in the configuration file.

### Launch Training Job
Launch the training job by running:
```bash
litgpt pretrain --config /path/to/your/xxx.yaml
```

Both pre-training and continued pre-training can be launched in this way.

## (Optional) Convert Lit Model to HuggingFace Format
In `jobs/convert_ckpts/convert_from_lit.sh`, modify `model_dir` to your designated directory, and run:
```bash
bash jobs/convert_ckpts/convert_from_lit.sh
```

This script will convert the model to HuggingFace format, and save it to `${model_dir}/final-hf`.
