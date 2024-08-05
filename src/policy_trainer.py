import torch
import torch.optim.swa_utils as swa
import transformers
from collections import defaultdict
from torch.utils.data import Dataset
from src.configs import WARPArgs


class EMAStepCallback(transformers.TrainerCallback):
    def __init__(self, model: transformers.PreTrainedModel, ema_model: swa.AveragedModel):
        self._model = model
        self._ema_model = ema_model

    def on_step_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        self._ema_model.update_parameters(self._model)


class PolicyTrainer(transformers.Trainer):
    EPS = 1e-7
    INVALID_LOGPROB = 0.0
    INVALID_REWARD = -1.0

    def __init__(
        self,
        policy: transformers.PreTrainedModel,
        policy_tokenizer: transformers.PreTrainedTokenizer,
        reward_model: transformers.PreTrainedModel,
        reward_tokenizer: transformers.PreTrainedTokenizer,
        generation_config: transformers.GenerationConfig,
        warp_args: WARPArgs,
        ref_policy: transformers.PreTrainedModel | None = None,
        training_args: transformers.TrainingArguments | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        callbacks: list[transformers.TrainerCallback] | None = None,
    ):
        super().__init__(
            model=policy,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=policy_tokenizer,
            callbacks=callbacks,
            compute_metrics=self._compute_metrics,
        )

        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.generation_config = generation_config
        self.warp_args = warp_args
        self._metrics = defaultdict(list)

        if ref_policy is None:
            self.ref_policy = swa.AveragedModel(self.model, multi_avg_fn=swa.get_ema_multi_avg_fn(1 - warp_args.ema_rate))
            self.add_callback(EMAStepCallback(self.model, self.ref_policy))
        else:
            self.ref_policy = ref_policy

    def compute_loss(self, model: transformers.PreTrainedModel, inputs: transformers.BatchEncoding, return_outputs: bool = False):
        prompt_length = inputs['input_ids'].shape[1]
        gen_tokens, gen_logprobs = self._generate(model, inputs, prompt_length)

        policy_logprobs = self._forward(model, gen_tokens, prompt_length)
        with torch.no_grad():
            ref_logprobs = self._forward(self.ref_policy, gen_tokens, prompt_length)

        kl = torch.sum(gen_logprobs - ref_logprobs, dim=1)
        reward = self._get_reward(gen_tokens)
        rlhf_reward = reward - self.warp_args.kl_coef * kl

        batch_metrics = {
            'batch_kl': kl.mean().item(),
            'batch_reward': reward.mean().item(),
            'batch_rlhf_reward': rlhf_reward.mean().item()
        }
        self.log(batch_metrics)

        if self.generation_config.num_return_sequences > 1:
            loss = self._rloo_loss(rlhf_reward, policy_logprobs.sum(dim=-1))
        else:
            loss = -torch.mean(rlhf_reward * policy_logprobs.sum(dim=-1))

        return (loss, policy_logprobs) if return_outputs else loss

    def _rloo_loss(self, rlhf_reward: torch.Tensor, policy_logprobs: torch.Tensor) -> torch.Tensor:
        k = self.generation_config.num_return_sequences
        rlhf_reward = rlhf_reward.reshape((-1, k))
        policy_logprobs = policy_logprobs.reshape((-1, k))

        baselines = (rlhf_reward.sum(dim=-1, keepdim=True) - rlhf_reward) / (k - 1)
        return -torch.mean((rlhf_reward - baselines) * policy_logprobs)

    def _compute_metrics(self, pred: transformers.EvalPrediction, compute_result: bool = False) -> dict[str, float]:
        prompt_length = pred.inputs['input_ids'].shape[1]
        gen_tokens, gen_logprobs = self._generate(self.model, pred.inputs, prompt_length)

        with torch.no_grad():
            ref_logprobs = self._forward(self.ref_policy, gen_tokens, prompt_length)

        kl = torch.sum(gen_logprobs - ref_logprobs, dim=1)
        reward = self._get_reward(gen_tokens)

        self._metrics['KL'].extend(kl.cpu().tolist())
        self._metrics['Reward'].extend(reward.cpu().tolist())
        batch_metrics = {'KL': kl.mean().item(), 'Reward': reward.mean().item()}

        if not compute_result:
            return batch_metrics

        final_metrics = {name: sum(values) / len(values) for name, values in self._metrics.items()}
        self._metrics = defaultdict(list)
        return  final_metrics

    def _generate(self, model: transformers.PreTrainedModel, inputs: transformers.BatchEncoding, prompt_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        generate_out = model.generate(
            inputs=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_tokens = generate_out.sequences
        generated_logprobs = self._process_generate_logits(
            torch.stack(generate_out.scores, dim=1),
            generated_tokens,
            prompt_length,
        )

        return generated_tokens, generated_logprobs

    def _process_generate_logits(self, logits: torch.Tensor, gen_tokens: torch.Tensor, prompt_length: int) -> torch.Tensor:
        generated_pad_mask = gen_tokens[:, prompt_length:] == self.tokenizer.pad_token_id
        all_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        logprobs = torch.gather(all_logprobs, 2, gen_tokens[:, prompt_length:].unsqueeze(-1)).squeeze(-1)
        logprobs[generated_pad_mask] = self.INVALID_LOGPROB
        return logprobs

    def _forward(self, model: transformers.PreTrainedModel, gen_tokens: torch.Tensor, prompt_length: int) -> torch.Tensor:
        attention_mask = gen_tokens != self.tokenizer.pad_token_id
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        model_out = model(input_ids=gen_tokens, attention_mask=attention_mask, position_ids=position_ids)
        logprobs = self._process_forward_logits(model_out.logits, gen_tokens, prompt_length)
        return logprobs

    def _process_forward_logits(self, policy_out: torch.Tensor, gen_tokens: torch.Tensor, prompt_length: int) -> torch.Tensor:
        policy_logits = policy_out[:, prompt_length - 1: -1]
        generated_pad_mask = gen_tokens[:, prompt_length:] == self.tokenizer.pad_token_id

        policy_logits /= (self.generation_config.temperature + self.EPS)
        all_logprobs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        logprobs = torch.gather(all_logprobs, 2, gen_tokens[:, prompt_length:].unsqueeze(-1)).squeeze(-1)
        logprobs[generated_pad_mask] = self.INVALID_LOGPROB
        return logprobs

    @torch.no_grad()
    def _get_reward(self, generated_tokens: torch.Tensor) -> torch.Tensor:
        finished_sequence_mask = generated_tokens[:, -1] == self.tokenizer.pad_token_id
        text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        reward_tokens = self.reward_tokenizer(text=text, truncation=True, padding=True, return_tensors='pt').to(self.reward_model.device)

        logits = self.reward_model(**reward_tokens).logits
        rewards = torch.nn.functional.softmax(logits, dim=-1)[:, 0]
        rewards[~finished_sequence_mask] = self.INVALID_REWARD

        return rewards
