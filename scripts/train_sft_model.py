import transformers
import peft
import datasets
from dataclasses import asdict
from trl import set_seed
from src import configs

if __name__ == '__main__':
    set_seed(0)

    parser = transformers.HfArgumentParser([configs.SFTArgs, configs.PolicyTrainArgs, configs.LoraArgs])
    sft_args, train_args, lora_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        r=lora_args.rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        fan_in_fan_out=True,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained('lvwerra/gpt2-imdb', device_map='cuda')
    model = peft.get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb", padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    imdb = datasets.load_dataset('stanfordnlp/imdb')
    imdb = imdb.map(lambda inp: tokenizer(inp['text'], truncation=True, max_length=512), remove_columns=['text', 'label'])

    training_args = transformers.TrainingArguments(
        output_dir=sft_args.save_folder,
        gradient_checkpointing_kwargs={"use_reentrant": False} if train_args.gradient_checkpointing else None,
        report_to='wandb',
        run_name='warp_sft_train',
        num_train_epochs=0,
        **asdict(train_args)
    )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=imdb['train'],
    )

    trainer.train()
