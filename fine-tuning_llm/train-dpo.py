"""Train a model with Direct Preference Optimization. We train the model using flash_attention_2
so your GPU has to support flash_attention_2. For the default configuration, 
you need a GPU with at least 24GB of memory. If you run out of memory, 
you can reduce the `lora_r` `max_prompt_length` and `max_length` or you can run the code with
a DeepSpeed json configuration file to offload the parameters to the CPU.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from jinja2.exceptions import TemplateError
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
    PreTrainedModel,
)
from trl import (
    DPOTrainer,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class TrainerConfig:
    """Arguments For trainer."""

    beta: float | None = field(
        default=0.1,
        metadata={
            "help": "The beta factor in DPO loss. Higher beta means less "
            "divergence from the initial policy."
        },
    )
    max_prompt_length: int | None = field(
        default=1024,
        metadata={
            "help": (
                "For DPO, the maximum length of the prompt to use "
                "for conditioning the model."
            )
        },
    )
    max_length: int | None = field(
        default=1536,
        metadata={
            "help": (
                "The max length for the whole prompt and completion. "
                "The max length for the completion will be "
                "`max_length - max_prompt_length`."
            )
        },
    )
    loss_type: str | None = field(
        default="sigmoid", metadata={"help": ("The loss type for DPO.")}
    )


def _check_insert_system_message(tokenizer: AutoTokenizer) -> bool:
    """Check if the tokenizer supports the ```system``` role."""
    chat_template = tokenizer.chat_template
    if chat_template is None:
        raise ValueError(
            "The tokenizer does not support chat template. "
            "Please provide a chat template for the tokenizer"
        )
    if "system" in chat_template:
        return True
    return False


def prepare_examples_for_training(
    example: dict,
    tokenizer: PreTrainedTokenizer,
    auto_insert_empty_system_msg: bool = True,
) -> dict:
    """Prepare examples for training."""
    prompt_message = example["chosen"][:-1]
    chosen_message = example["chosen"][-1]
    rejected_message = example["rejected"][-1]
    # adding empty system message to the prompt
    if (
        auto_insert_empty_system_msg
        and _check_insert_system_message(tokenizer)
        and prompt_message[0]["role"] == "user"
    ):
        prompt_message = [
            {"role": "system", "content": ""},
        ] + prompt_message

    example["prompt"] = tokenizer.apply_chat_template(prompt_message, tokenize=False)
    example["chosen"] = _formatting_assistant_message(
        chosen_message["content"], tokenizer
    )
    example["rejected"] = _formatting_assistant_message(
        rejected_message["content"], tokenizer
    )
    return example


def _formatting_assistant_message(
    assistant_message: str, tokenizer: PreTrainedTokenizer
) -> str:
    """Format assistant message to a string."""
    try:
        # some tokenizer does accept a messages with only assistant role
        assistant_message = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": assistant_message}],
            tokenize=False,
        )
        return assistant_message
    except TemplateError:
        if _check_insert_system_message(tokenizer):
            raw_message = tokenizer.apply_chat_template(
                [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
                tokenize=False,
            )
            raw_message_with_assistant = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": assistant_message},
                ],
                tokenize=False,
            )
            return raw_message_with_assistant[len(raw_message) :]
        else:
            raw_message = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False,
            )
            raw_message_with_assistant = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": assistant_message},
                ],
                tokenize=False,
            )
            return raw_message_with_assistant[len(raw_message) :]


def is_adapter_model(model_cfg: ModelConfig) -> bool:
    """Check if the model is an adapter model.

    A helpfer function to check if the model is an adapter model.
    This help to load the model correctly if you want to continue training
    from with lora adapter.
    """
    repo_files = os.listdir(model_cfg.model_name_or_path)
    return (
        "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files
    )


def load_model(model_cfg: ModelConfig) -> PreTrainedModel:
    """Load the model with the given configuration.

    Load the model from configurations.

    Args:
        model_cfg: The model configuration."""
    torch_dtype = (
        model_cfg.torch_dtype
        if model_cfg.torch_dtype in ["auto", None]
        else getattr(torch, model_cfg.torch_dtype)
    )
    quantization_config = get_quantization_config(model_cfg)
    # check if the model is an adapter model
    if is_adapter_model(model_cfg):
        peft_config = PeftConfig.from_pretrained(model_cfg.model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=True,
            attn_implementation=model_cfg.attn_implementation,
            use_cache=False,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map=get_kbit_device_map(),
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_cfg.model_name_or_path,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name_or_path,
            trust_remote_code=True,
            attn_implementation=model_cfg.attn_implementation,
            use_cache=False,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map=get_kbit_device_map(),
        )
    return model


def train_dpo(
    model_cfg: ModelConfig,
    training_args: TrainingArguments,
    trainer_args: TrainerConfig,
) -> None:
    """Train a model with Direct Preference Optimization."""

    accelerator = Accelerator()
    # we fix the dataset, so we can apply the same processing otherwise
    # you have to process the dataset with your own function
    dataset_name = "argilla/dpo-mix-7k"
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.model_max_length > trainer_args.max_length:
        tokenizer.model_max_length = trainer_args.max_length
    # this to make sure that you do not get an error
    # when you run the code in a distributed environment
    with accelerator.main_process_first():
        dataset = load_dataset(
            dataset_name,
            # store the cache of the dataset in the output directory
            cache_dir=Path(training_args.output_dir) / "dataset_cache",
        )
    column_names = dataset.column_names["train"]
    remove_columns = [
        column_name
        for column_name in column_names
        if column_name not in ["chosen", "rejected", "prompt"]
    ]
    # formating the dataset for training
    with accelerator.main_process_first():
        dataset = dataset.map(
            prepare_examples_for_training,
            fn_kwargs={
                "tokenizer": tokenizer,
            },
            remove_columns=remove_columns,
        )
    # load the quantize model
    model = load_model(model_cfg)
    ref_model = None if model_cfg.use_peft else model
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=trainer_args.beta,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        max_length=trainer_args.max_length,
        max_prompt_length=trainer_args.max_prompt_length,
        loss_type=trainer_args.loss_type,
        peft_config=get_peft_config(model_cfg),
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    dataset.cleanup_cache_files()


if __name__ == "__main__":
    # To run the code with the default value
    # You need to have a GPU with at least 20GB of memory
    # the GPu has to support bfloat16 and tf32 and also the flash_attention_2
    training_args = TrainingArguments(
        output_dir="training_llm_with_dida",
        bf16=True,
        tf32=True,
        evaluation_strategy="steps",
        eval_steps=100,
        # reduce the number of gradient accumulation steps if you use multiple devices.
        gradient_accumulation_steps=12,
        gradient_checkpointing=True,
        learning_rate=5e-6,
        log_level="info",
        logging_steps=50,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        optim="adamw_torch",
        do_eval=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_total_limit=1,
        warmup_ratio=0.1,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},  # do not change this
    )
    model_cfg = ModelConfig(
        # model_name_or_path can be the path to an adapter model
        # or a model from huggingface model hub
        # or a local model
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        use_peft=True,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        use_bnb_nested_quant=True,
    )
    trainer_args = TrainerConfig(
        max_prompt_length=1024,
        max_length=1536,
    )
    set_seed(42)
    # If you run out of memory, you can reduce the `lora_r` `max_prompt_length` and `max_length`
    # and you can also reduce the number of layers for lora adapter in `lora_target_modules`
    # if you are running with multiple GPUs, or CPUs for training
    # you should use deepspeed, the code does support deepspeed
    # you can only use zero stage 2 if you use quantization.
    train_dpo(model_cfg, training_args, trainer_args)
