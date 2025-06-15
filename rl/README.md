# Reinforcement Learning

We use `verl` to RL fine-tune models.

## Installation

Requirements:
- Python: Version >= 3.9
- CUDA: Version >= 12.1

First create a new virtual environment and activate it:

```bash
conda create -n verl python==3.10
conda activate verl
```

Then install the package with all dependencies:

```bash
cd verl
pip3 install -e .
pip3 install vllm==0.8.2
pip3 install flash-attn --no-build-isolation
```

These two steps should be enough to install the package and all dependencies. If there is any issue, please refer to the original `verl` project at https://github.com/volcengine/verl.

