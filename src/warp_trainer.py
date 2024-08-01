import os
import gc
import wandb
import torch
import torch.multiprocessing as tmp
import transformers
import peft
from dataclasses import asdict
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from src import configs, utils, policy_trainer


class AsyncProgressCallback(transformers.TrainerCallback):
    def __init__(self, queue: tmp.Queue):
        self._queue = queue

    def on_step_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        self._queue.put(None)


class WARPTrainer:
    def __init__(
        self,
        warp_args: configs.WARPArgs,
        generation_args: configs.GenerationArgs,
        training_args: configs.PolicyTrainArgs,
        lora_args: configs.LoraArgs,
        checkpoints_args: configs.CheckpointsArgs,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None
    ):
        self.checkpoints_args = checkpoints_args
        self.generation_config = transformers.GenerationConfig.from_dict(asdict(generation_args))
        self.warp_args = warp_args
        self.train_args = training_args
        self.lora_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=lora_args.rank,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
        )

        self.init_policy_checkpoint = checkpoints_args.sft_checkpoint
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        manager = tmp.Manager()
        self._progress_queue = manager.Queue()
        self._device_queue = manager.Queue()
        self._print_event = manager.Event()
        self._eval_run_id = None

        device_count = torch.cuda.device_count()
        for device_idx in range(device_count):
            self._device_queue.put(device_idx)

    def train(self):
        steps_per_iteration = self.warp_args.num_policies * self.train_args.max_steps
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            utils.get_latest_checkpoint(self.init_policy_checkpoint),
            padding_side='left'
        )

        for iter_idx in range(self.warp_args.num_iterations):
            with tmp.Pool(self._device_queue.qsize()) as pool:
                for run_idx in range(self.warp_args.num_policies):
                    pool.apply_async(self._train_policy, (iter_idx, run_idx))

                self._print_event.wait()
                with tqdm(total=steps_per_iteration, desc=f'Iteration {iter_idx + 1}/{self.warp_args.num_iterations}') as progress_bar:
                    for _ in range(steps_per_iteration):
                        self._progress_queue.get()
                        progress_bar.update()

                pool.close()
                pool.join()

            slerp_model = self._slerp(iter_idx)
            slerp_path = self._slerp_path(iter_idx)
            slerp_model.save_pretrained(slerp_path)
            tokenizer.save_pretrained(slerp_path)

            init_model = self._litti(slerp_model)
            self.init_policy_checkpoint = self._init_path(iter_idx + 1)
            init_model.save_pretrained(self.init_policy_checkpoint)
            tokenizer.save_pretrained(self.init_policy_checkpoint)

            del slerp_model, init_model
            gc.collect()
            self._evaluate_slerp_model(iter_idx)

    def _train_policy(self, iter_idx: int, run_idx: int):
        device_idx = self._pop_device()
        device = torch.device('cuda')
        train_args = self._get_train_args(iter_idx, run_idx)

        policy, policy_tokenizer = self._load_policy(self.init_policy_checkpoint, device_map=device)
        reward_model, reward_tokenizer = self._load_reward(device_map=device)
        callbacks = [AsyncProgressCallback(self._progress_queue)]

        with wandb.init(group=self.checkpoints_args.group_name, job_type=train_args.run_name, name=train_args.run_name) as run:
            if iter_idx == 0 and run_idx == 0:
                utils.print_wandb_run(run)
                self._print_event.set()

            trainer = policy_trainer.PolicyTrainer(
                policy,
                policy_tokenizer,
                reward_model,
                reward_tokenizer,
                self.generation_config,
                self.warp_args,
                training_args=train_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                callbacks=callbacks
            )

            trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
            trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
            trainer.train()

        self._device_queue.put(device_idx)

    def _litti(self, slerp_model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:
        init_model, _ = self._load_policy(self.init_policy_checkpoint)
        for init_param, slerp_param in zip(init_model.parameters(), slerp_model.parameters()):
            init_param = init_param.detach()
            init_param += self.warp_args.liti_rate * (slerp_param - init_param)
        return init_model

    def _slerp(self, iter_idx: int) -> transformers.PreTrainedModel:
        if self.warp_args.num_policies == 1:
            return self._load_policy(self._policy_path(iter_idx, 0))

        if self.warp_args.num_policies == 2:
            return self._slerp_first_two_policies(iter_idx)

        # merge M > 2 policies
        init_model, _ = self._load_policy(self.init_policy_checkpoint)
        slerp_model = self._slerp_first_two_policies(iter_idx)

        for run_idx in range(2, self.warp_args.num_policies):
            policy, _ = self._load_policy(self._policy_path(iter_idx, run_idx))
            params_iter = zip(init_model.named_parameters(), slerp_model.parameters(), policy.parameters())

            for (name, init_param), slerp_param, policy_param in params_iter:
                if not utils.is_lora_layer(name):
                    continue

                slerp_param = slerp_param.detach()
                task_vector_1 = slerp_param - init_param
                task_vector_2 = policy_param - init_param

                angle = utils.compute_angle(task_vector_1, task_vector_2)
                mult = torch.sin(angle / self.warp_args.num_policies) / torch.sin(angle)

                slerp_param *= mult
                slerp_param += (1 - mult) * init_param + mult * task_vector_2

        return slerp_model

    def _slerp_first_two_policies(self, iter_idx: int) -> transformers.PreTrainedModel:
        slerp_model, _ = self._load_policy(self.init_policy_checkpoint)
        policy_1, _ = self._load_policy(self._policy_path(iter_idx, 0))
        policy_2, _ = self._load_policy(self._policy_path(iter_idx, 1))

        params_iter = zip(slerp_model.named_parameters(), policy_1.parameters(), policy_2.parameters())

        for (name, slerp_param), policy1_param, policy2_param in params_iter:
            if not utils.is_lora_layer(name):
                continue

            slerp_param = slerp_param.detach()
            task_vector_1 = policy1_param - slerp_param
            task_vector_2 = policy2_param - slerp_param

            angle = utils.compute_angle(task_vector_1, task_vector_2)
            slerp_param += torch.sin((1 - self.warp_args.slerp_rate) * angle) / torch.sin(angle) * task_vector_1
            slerp_param += torch.sin(self.warp_args.slerp_rate * angle) / torch.sin(angle) * task_vector_2
    
        return slerp_model

    def _evaluate_slerp_model(self, iter_idx: int):
        device_idx = self._pop_device()
        device = torch.device('cuda')
        group_name = self.checkpoints_args.group_name
        eval_args = self._get_eval_args(iter_idx)

        slerp_model, slerp_tokenizer = self._load_policy(self._slerp_path(iter_idx), is_trainable=False, device_map=device)
        slerp_model.eval()
        sft_model, _ = self._load_policy(self.checkpoints_args.sft_checkpoint, device_map=device)
        sft_model.eval()
        reward_model, reward_tokenizer = self._load_reward(device_map=device)

        with wandb.init(group=group_name, job_type=eval_args.run_name, name=eval_args.run_name, id=self._eval_run_id, resume='allow') as run:
            self._eval_run_id = self._eval_run_id or run.id

            trainer = policy_trainer.PolicyTrainer(
                slerp_model,
                slerp_tokenizer,
                reward_model,
                reward_tokenizer,
                self.generation_config,
                self.warp_args,
                sft_model,
                eval_args,
                eval_dataset=self.eval_dataset,
            )

            trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
            trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
            trainer.evaluate()

            self._device_queue.put(device_idx)

    def _pop_device(self) -> int:
        device_idx = self._device_queue.get()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_idx)
        return device_idx

    def _get_train_args(self, iter_idx: int, run_idx: int) -> transformers.TrainingArguments:
        output_dir = self._policy_path(iter_idx, run_idx)
        run_name = f'iter_{iter_idx}_run_{run_idx}'
        seed = iter_idx * self.warp_args.num_policies + run_idx

        return transformers.TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            seed=seed,
            gradient_checkpointing_kwargs={"use_reentrant": False} if self.train_args.gradient_checkpointing else None,
            num_train_epochs=0,
            **asdict(self.train_args),
        )

    def _get_eval_args(self, iter_idx: int) -> transformers.TrainingArguments:
        output_dir = self._slerp_path(iter_idx)
        return transformers.TrainingArguments(
            output_dir=output_dir,
            run_name='eval_slerp',
            batch_eval_metrics=True,
            include_inputs_for_metrics=True,
            num_train_epochs=0,
            **asdict(self.train_args),
        )

    def _load_policy(self, path: str, is_trainable: bool = True, device_map: torch.device | None = None) -> peft.peft_model.PeftModelForCausalLM:
        path = utils.get_latest_checkpoint(path)
        policy = peft.AutoPeftModelForCausalLM.from_pretrained(path, is_trainable=is_trainable, device_map=device_map)
        policy.config.use_cache = not self.train_args.gradient_checkpointing
        tokenizer = transformers.AutoTokenizer.from_pretrained(path, padding_side='left')
        return policy, tokenizer

    def _load_reward(self, device_map: torch.device | None = None) -> transformers.PreTrainedModel:
        path = utils.get_latest_checkpoint(self.checkpoints_args.reward_checkpoint)
        reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(path, device_map=device_map)
        reward_model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(path, padding_side='left')
        return reward_model, tokenizer

    def _policy_path(self, iter_idx: int, run_idx: int):
        return os.path.join(self.checkpoints_args.save_folder, f'iter_{iter_idx}_run_{run_idx}')

    def _slerp_path(self, iter_idx: int):
        return os.path.join(self.checkpoints_args.save_folder, f'iter_{iter_idx}_slerp')

    def _init_path(self, iter_idx: int):
        return os.path.join(self.checkpoints_args.save_folder, f'iter_{iter_idx}_init')
