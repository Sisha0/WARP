from dataclasses import dataclass, field


@dataclass
class SFTArgs:
    save_folder: str = field(default='sft', metadata={'help': 'Folder to save SFT model.'})


@dataclass
class RewardConfigArgs:
    output_dir: str = field(default='reward', metadata={'help': 'The output directory where the model checkpoints will be written.'})
    per_device_train_batch_size: int = field(default=32, metadata={'help': 'The batch size for training.'})
    gradient_accumulation_steps: int = field(default=1, metadata={'help': 'Number of updates steps to accumulate the gradients for.'})
    learning_rate: float = field(default=1.41e-5, metadata={'help': 'The initial learning rate for AdamW optimizer.'})
    max_steps: int = field(default=4000, metadata={'help': 'The total number of training steps to perform.'})
    logging_steps: int = field(default=100, metadata={'help': 'Number of update steps between two logs.'})
    gradient_checkpointing: bool = field(default=True, metadata={'help': 'If True, use gradient checkpointing.'})
    fp16: bool = field(default=True, metadata={'help': 'Whether to use 16-bit (mixed) precision training.'})
    bf16: bool = field(default=False, metadata={'help': 'Whether to use bf16 16-bit (mixed) precision training.'})
    max_length: int = field(default=512, metadata={'help': 'The maximum length of the sequences in the batch.'})


@dataclass
class PolicyTrainArgs:
    per_device_train_batch_size: int = field(default=64, metadata={'help': 'The batch size for training.'})
    per_device_eval_batch_size: int = field(default=32, metadata={'help': 'The batch size for evaluation.'})
    gradient_accumulation_steps: int = field(default=1, metadata={'help': 'Number of updates steps to accumulate the gradients for.'})
    learning_rate: float = field(default=1.41e-5, metadata={'help': 'The initial learning rate for AdamW optimizer.'})
    max_steps: int = field(default=100, metadata={'help': '(T) Number of train iterations for each policy.'})
    logging_steps: int = field(default=10, metadata={'help': 'Number of update steps between two logs.'})
    gradient_checkpointing: bool = field(default=True, metadata={'help': 'If True, use gradient checkpointing.'})
    fp16: bool = field(default=False, metadata={'help': 'Whether to use fp16 16-bit (mixed) precision training.'})
    bf16: bool = field(default=False, metadata={'help': 'Whether to use bf16 16-bit (mixed) precision training.'})
    warmup_steps: int = field(default=0, metadata={'help': 'Linear warmup over warmup_steps.'})


@dataclass
class LoraArgs:
    rank: int = field(default=32, metadata={'help': 'Lora attention dimension.'})
    lora_alpha: int = field(default=32, metadata={'help': 'The alpha parameter for Lora scaling.'})
    lora_dropout: float = field(default=0.0, metadata={'help': 'The dropout probability for Lora layers.'})


@dataclass
class CheckpointsArgs:
    group_name: str = field(default='warp', metadata={'help': 'Group name for WAND runs. Should be unique every time, otherwise logs to the existing group.'})
    sft_checkpoint: str = field(default='sft', metadata={'help': 'Path to sft model/tokenizer checkpoint.'})
    reward_checkpoint: str = field(default='reward', metadata={'help': 'Path to reward model/tokenizer checkpoint.'})
    save_folder: str = field(default='warp', metadata={'help': 'Folder to save WARP output.'})


@dataclass
class DatasetArgs:
    min_text_length: int = field(default=200, metadata={'help': 'Minimum length of a review from imdb dataset.'})
    min_tokens: int = field(default=5, metadata={'help': 'Minimum number of tokens after prompt truncation.'})
    max_tokens: int = field(default=20, metadata={'help': 'Maximum number of tokens after prompt truncation.'})
    eval_size: int = field(default=100, metadata={'help': 'Evaluation subset size.'})


@dataclass
class GenerationArgs:
    max_new_tokens: int = field(default=64, metadata={'help': 'The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.'})
    do_sample: bool = field(default=True, metadata={'help': 'Whether or not to use sampling ; use greedy decoding otherwise.'})
    top_k: int | None = field(default=None, metadata={'help': 'The number of highest probability vocabulary tokens to keep for top-k-filtering.'})
    top_p: float | None = field(default=1.0, metadata={'help': 'Only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'})
    temperature: float | None = field(default=1.0, metadata={'help': 'The value used to modulate the next token probabilities.'})
    num_return_sequences: int = field(default=1, metadata={'help': 'The number of independently computed returned sequences for each element in the batch'})


@dataclass
class WARPArgs:
    num_iterations: int = field(default=2, metadata={'help': '(I) Number of WARP iterations.'})
    num_policies: int = field(default=2, metadata={'help': '(M) Number of policies to train in parallel.'})
    kl_coef: float = field(default=0.1, metadata={'help': '(beta) Weight of KL for KL-regularized reward.'})
    ema_rate: float = field(default=0.01, metadata={'help': '(mu) EMA rate for reference policies.'})
    slerp_rate: float = field(default=0.5, metadata={'help': '(lambda) Weight to SLERP trained policies. Ignored when M > 2 (lambda = 1/M).'})
    liti_rate: float = field(default=0.5, metadata={'help': '(eta) LITI rate for initial policy.'})
