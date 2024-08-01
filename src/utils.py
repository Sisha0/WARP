import os
import re
import torch
import wandb
from wandb.sdk.wandb_run import Run


def compute_angle(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    cos = torch.sum(v1 * v2) / torch.norm(v1) / torch.norm(v2)
    return torch.acos(cos)


def get_latest_checkpoint(dir: str | os.PathLike) -> str:
    pattern = r'checkpoint-\d+'
    checkpoints = [file for file in os.scandir(dir) if file.is_dir() and re.fullmatch(pattern, file.name)]
    if not checkpoints:
        return dir

    latest_checkpoint = max(int(checkpoint.name.split('-')[1]) for checkpoint in checkpoints)
    return os.path.join(dir, f'checkpoint-{latest_checkpoint}')


def print_wandb_run(run: Run):
    wandb_path = '/'.join(run.dir.split('/')[:-2])
    group_url = f'{run.get_project_url()}/groups/{run.group}'

    info = [
        f'Currently logged in as: {run.entity}',
        f'Tracking run with wandb version {wandb.__version__}',
        f'Run data is saved locally in {wandb_path}',
        f'View project at {run.get_project_url()}',
        f'View runs at {group_url}'
    ]

    print(*info, sep='\n')


def is_lora_layer(name: str) -> bool:
    return name.find('lora_A') != -1 or name.find('lora_B') != -1
