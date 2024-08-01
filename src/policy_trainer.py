import torch
import torch.optim.swa_utils as swa
import transformers
from collections import defaultdict
from torch.utils.data import Dataset
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from src.configs import WARPArgs


class EMAStepCallback(transformers.TrainerCallback):
    def __init__(self, model: transformers.PreTrainedModel, ema_model: swa.AveragedModel):
        self._model = model
        self._ema_model = ema_model

    def on_step_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        self._ema_model.update_parameters(self._model)


class PolicyTrainer(transformers.Trainer):
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
        self.ref_policy = ref_policy or swa.AveragedModel(self.model, multi_avg_fn=swa.get_ema_multi_avg_fn(1 - warp_args.ema_rate))

        self._metrics = defaultdict(list)
        if ref_policy:
            self.add_callback(EMAStepCallback(self.model, self.ref_policy))

    def _get_logprobs(self, policy_out: CausalLMOutputWithCrossAttentions, prompt_length: int, generated_tokens: torch.Tensor) -> torch.Tensor:
        policy_logits = policy_out.logits[:, prompt_length - 1: -1]
        policy_logits /= (self.generation_config.temperature + 1e-7)
        all_logprobs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        return torch.gather(all_logprobs, 2, generated_tokens[:, prompt_length:].unsqueeze(-1)).squeeze(-1)

    def _compute_metrics(self, pred: transformers.EvalPrediction, compute_result: bool = False) -> dict[str, float]:
        prompt_length = pred.inputs['input_ids'].shape[1]
        generated_tokens = self.model.generate(
            inputs=pred.inputs['input_ids'],
            attention_mask=pred.inputs['attention_mask'],
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
        )
        attention_mask = generated_tokens != self.tokenizer.pad_token_id

        with torch.no_grad():
            policy_out = self.model(input_ids=generated_tokens, attention_mask=attention_mask)
            ref_policy_out = self.ref_policy(input_ids=generated_tokens, attention_mask=attention_mask)

        policy_logprobs = self._get_logprobs(policy_out, prompt_length, generated_tokens)
        ref_logprobs = self._get_logprobs(ref_policy_out, prompt_length, generated_tokens)

        kl = torch.sum(policy_logprobs - ref_logprobs, dim=1)
        text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        reward_tokens = self.reward_tokenizer(text=text, truncation=True, padding=True, return_tensors='pt').to(self.reward_model.device)

        with torch.no_grad():
            reward = self.reward_model(**reward_tokens).logits[:, 0]

        self._metrics['KL'].extend(kl.cpu().tolist())
        self._metrics['Reward'].extend(reward.cpu().tolist())
        batch_metrics = {'KL': kl.mean().item(), 'Reward': reward.mean().item()}

        if not compute_result:
            return batch_metrics

        final_metrics = {name: sum(values) / len(values) for name, values in self._metrics.items()}
        self._metrics = defaultdict(list)
        return  final_metrics

    def compute_loss(self, model: transformers.PreTrainedModel, inputs: transformers.BatchEncoding, return_outputs: bool = False):
        prompt_length = inputs['input_ids'].shape[1]
        generated_tokens = model.generate(
            inputs=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
        )
        attention_mask = generated_tokens != self.tokenizer.pad_token_id

        policy_out = model(input_ids=generated_tokens, attention_mask=attention_mask)
        policy_logprobs = self._get_logprobs(policy_out, prompt_length, generated_tokens)
        with torch.no_grad():
            ref_policy_out = self.ref_policy(input_ids=generated_tokens, attention_mask=attention_mask)
        ref_logprobs = self._get_logprobs(ref_policy_out, prompt_length, generated_tokens)
        kl = torch.sum(policy_logprobs - ref_logprobs, dim=1)
        text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        reward_tokens = self.reward_tokenizer(text=text, truncation=True, padding=True, return_tensors='pt').to(self.reward_model.device)

        with torch.no_grad():
            reward = self.reward_model(**reward_tokens).logits[:, 0]
        rlhf_reward = reward - self.warp_args.kl_coef * kl
        # print(kl.mean(), reward.mean(), rlhf_reward.mean(), policy_logprobs.sum(dim=-1))

        loss = -torch.mean(rlhf_reward * policy_logprobs.sum(dim=-1))
        return (loss, policy_out) if return_outputs else loss
