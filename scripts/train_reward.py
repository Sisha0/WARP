import torch
import transformers
import trl
from dataclasses import asdict
from src.dataset import IMDBRewardDataset
from src.configs import RewardConfigArgs


if __name__ == '__main__':
    trl.set_seed(0)
    parser = transformers.HfArgumentParser(RewardConfigArgs)
    reward_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    device = torch.device('cuda')
    model_name = 'distilbert-base-cased'

    reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, device_map=device)
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    reward_dataset = IMDBRewardDataset(reward_tokenizer, 0)

    training_args = trl.RewardConfig(
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False} if reward_args.gradient_checkpointing else None,
        report_to='wandb',
        run_name='warp_reward_train',
        **asdict(reward_args)
    )

    trainer = trl.RewardTrainer(
        model=reward_model,
        args=training_args,
        tokenizer=reward_tokenizer,
        train_dataset=reward_dataset,
    )

    trainer.train()
