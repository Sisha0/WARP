# WARP: On the Benefits of Weight Averaged Rewarded Policies

Original paper: https://arxiv.org/abs/2406.16768.

Report with rexperiments results: [link](./report/report.pdf).

Checkpoints folder: https://drive.google.com/drive/folders/1iZ7S603cg9yZ6q1kW0EgSuYKafsG-uCY.

The goal is to verify whether the algorithm works for [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) reviews dataset. We will fine-tune the model to generate more positive reviews.

# How to run

## Environment

Python version: **3.12.4** (3.12.* versions should be fine). Dependencies can be install with the command:

```shell
pip install -r requirements.txt
```

To run the experiments, you need to have **at least one GPU**. All the scripts used for training log into [WANDB](https://wandb.ai/), so either set the mode to offline (using environment variable `WANDB_MODE = 'offline'`) or log in locally (run `wandb login` and provide the [API key](https://wandb.ai/authorize)).

## Reward model

[distilbert-base-cased](https://huggingface.co/distilbert/distilbert-base-cased) is used as a reward model. Details about reward modeling can be found in the report.
To train a reward model, run the script from the project's root folder:

```shell
python -m scripts.train_reward
```

To see the full list of supported arguments, run:

```shell
python -m scripts.train_reward --help
```

During the later experiments, we did use the reward model from this [run](https://wandb.ai/sisha/huggingface/runs/x6le85qi).

## SFT model

[lvwerra/gpt2-imdb](https://huggingface.co/lvwerra/gpt2-imdb) is used as a reference model. Because we are going to use [LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) for training during RLHF stage, we can't use plain lvwerra/gpt2-imdb model (because it has no LoRA layers). So before RLHF stage, we fine-tune the model with LoRA for some iterations:

```shell
python -m scripts.train_sft_model --fp16 --per_device_train_batch_size=32
```

To see the full list of supported arguments, run:

```shell
python -m scripts.train_reward --help
```

During the later experiments, we did use the SFT model from this [run](https://wandb.ai/sisha/huggingface/runs/z08v0ek8).

## RLHF

### WARP

[lvwerra/gpt2-imdb](https://huggingface.co/lvwerra/gpt2-imdb) is fine-tuned with RLHF to generate more positive reviews. To fine-tune the model, run the script:

```shell
python -m scripts.train_policy --learning_rate=0.0001 --max_new_tokens=128 --per_device_train_batch_size=16 --per_device_eval_batch_size=16
```

To see the full list of supported arguments, run:

```shell
python -m scripts.train_policy --help
```

If you did log in to WANDB, the script will log runs into the group specified with `--group_name` parameter (*warp* by default). If you run the script again with the same `--group_name`, all the new runs will go to the same group, so it's recommended to use new group name every time (or delete previous runs from the group). An example experiment is [here](https://wandb.ai/sisha/tk-alignment/groups/warp/workspace).

### WARP with RLOO

We use [RLOO](https://arxiv.org/abs/2402.14740) as an alternative to plain REINFORCE used in the original paper. This leads to a better KL/Reward trade-off (see [report](./report/report.pdf)). The run is [here](https://wandb.ai/sisha/tk-alignment/groups/warp_test_rloo).

# Results

Details about parameters choices and experiments' results are in the [report](./report/report.pdf). Plots can be reproduces with the [notebook](./notebooks/kl-reward-plot.ipynb). An example of how to run the code in Kaggle's environment (with 2x Tesla T4) is [here](./notebooks/kaggle.ipynb).
