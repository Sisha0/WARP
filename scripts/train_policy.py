import os
import transformers
import torch.multiprocessing as tmp
from torch.utils.data import Subset
from src import configs, dataset, warp_trainer, utils


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_SILENT'] = 'true'
    tmp.set_start_method('spawn')

    parser = transformers.HfArgumentParser([
        configs.WARPArgs,
        configs.DatasetArgs,
        configs.GenerationArgs,
        configs.PolicyTrainArgs,
        configs.LoraArgs,
        configs.CheckpointsArgs
    ])
    warp_args, dataset_args, generation_args, train_args, lora_args, checkpoints_args, unknown_args =\
        parser.parse_args_into_dataclasses(return_remaining_strings=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        utils.get_latest_checkpoint(checkpoints_args.sft_checkpoint)
    )
    imdb = dataset.build_imdb_dataset(
        tokenizer,
        min_text_length=dataset_args.min_text_length,
        min_tokens=dataset_args.min_tokens,
        max_tokens=dataset_args.max_tokens,
    )

    train_dataset = imdb['train']
    test_dataset = Subset(imdb['test'], list(range(dataset_args.eval_size)))

    warp_trainer = warp_trainer.WARPTrainer(
        warp_args,
        generation_args,
        train_args,
        lora_args,
        checkpoints_args,
        train_dataset,
        test_dataset
    )

    warp_trainer.train()
