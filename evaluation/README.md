# Evaluation

We use `lm-eval-harness` and `cot-eval-harness` to evaluate models.

## Evaluation of Pre-trained Models

### Installation of `lm-eval-harness`

**Step 1:** First create a new virtual environment and activate it

```bash
conda create -n lmeval python=3.10
conda activate lmeval
```

**Step 2:** Then install the package with all dependencies

```bash
cd lm-evaluation-harness
pip install -e ".[vllm,wandb]"
```

These two steps should be enough to install the package and all dependencies. If there is any issue, please refer to the original `lm-evaluation-harness` project at https://github.com/EleutherAI/lm-evaluation-harness.