import torch
from typing import Optional
from transformers import LlamaForCausalLM
from dataclasses import asdict

from my_datasets.serialization.serializers import get_serializer
from my_datasets.tokenization.text import prepare_tokenizer, sanity_check_tokenizer
from configs import SerializerConfig, TokenizerConfig
from tools.utils import freeze_transformer_layers
from dataset_info import validate_candidate_configuration

from peft import get_peft_model, LoraConfig, PeftConfig, prepare_model_for_kbit_training, PeftModel
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict


def load_stage_adapter(model, checkpoint_path, adapter_name="default", is_trainable=False):
    config = PeftConfig.from_pretrained(checkpoint_path)
    config.inference_mode = not is_trainable
    if isinstance(model, PeftModel):
        model.add_adapter(adapter_name, config)
    else:
        model = get_peft_model(model, config, adapter_name=adapter_name)

    weights = load_peft_weights(checkpoint_path, device="cpu")
    normalized_weights = {}
    for key, value in weights.items():
        # Historical checkpoints contain one wrapper prefix for each training stage.
        while key.startswith("base_model.model.base_model.model."):
            key = key[len("base_model.model."):]
        if key in normalized_weights:
            raise ValueError(f"Duplicate adapter weight after prefix normalization: {key}")
        normalized_weights[key] = value

    result = set_peft_model_state_dict(model, normalized_weights, adapter_name=adapter_name)
    missing = [key for key in result.missing_keys
               if "lora_" in key and f".{adapter_name}." in key]
    if missing or result.unexpected_keys:
        raise ValueError(
            f"Adapter checkpoint {checkpoint_path} does not match the model: "
            f"missing={missing}, unexpected={result.unexpected_keys}"
        )
    return model


def load_llama_2(tokenizer,
                    serializer_config: SerializerConfig,
                    tokenizer_config: TokenizerConfig,
                    model_name_or_path: str,
                    train_config
                 ):

    # Load base model
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        device_map = {"": torch.cuda.current_device()},
        attn_implementation="sdpa"
    )

    train_serializer = get_serializer(serializer_config)
    tokenizer, model = prepare_tokenizer(
        model,
        tokenizer=tokenizer,
        pretrained_model_name_or_path=model_name_or_path,
        model_max_length=train_config.context_length,
        use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
        serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
        serializer_tokens=train_serializer.special_tokens
        if tokenizer_config.add_serializer_tokens
        else None,
    )
    # sanity check for tokenizer
    sanity_check_tokenizer(tokenizer, model_name_or_path)

    # setting up FSDP if enable_fsdp is enabled
    if train_config.freeze_layers:
        print('-------------Freeze layers-------------')
        freeze_transformer_layers(model, train_config.num_freeze_layers)

    return model, tokenizer


def load_llama_3_lora(tokenizer,
                serializer_config: SerializerConfig,
                tokenizer_config: TokenizerConfig,
                model_name_or_path: str,
                task_type,
                train_config,
                lora_config,
                pretrain_state_dict_path: Optional[str] = None,
                screen_state_dict_path: Optional[str] = None,
                test_task: bool=False,
                local_rank: Optional[int]=None,
                assessment_state_dict_path: Optional[str]=None):

    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=True,
        attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()}
    )

    train_serializer = get_serializer(serializer_config)
    tokenizer, model = prepare_tokenizer(
        model,
        tokenizer=tokenizer,
        pretrained_model_name_or_path=model_name_or_path,
        model_max_length=train_config.context_length,
        use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
        serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
        serializer_tokens=train_serializer.special_tokens
        if tokenizer_config.add_serializer_tokens
        else None,
    )

    # sanity check for tokenizer
    sanity_check_tokenizer(tokenizer, model_name_or_path)
    validate_candidate_configuration(tokenizer, task_type)

    if task_type != "Pretrain":
        if local_rank == 0:
            print("-----Load pretrained weights-----")
        model = load_stage_adapter(
            model, pretrain_state_dict_path, adapter_name="pretrain"
        )

        if task_type == "Assessment":
            if local_rank == 0:
                print("-----Load LoRA weights from the Screen stage-----")
            model = load_stage_adapter(model, screen_state_dict_path, adapter_name="screen")

    if not test_task:
        if local_rank == 0:
            print("-----Initialize a new LoRA module-----")
        model.gradient_checkpointing_enable()

        # Prepare for k-bit LoRA training
        model = prepare_model_for_kbit_training(model)

        # Load LoRA config
        lora_config = LoraConfig(**asdict(lora_config))

        if isinstance(model, PeftModel):
            model.add_adapter("default", lora_config)
        else:
            model = get_peft_model(model, lora_config)
    else:
        checkpoint_paths = {
            "Pretrain": pretrain_state_dict_path,
            "Screen": screen_state_dict_path,
            "Assessment": assessment_state_dict_path,
        }
        checkpoint_path = checkpoint_paths[task_type]
        if not checkpoint_path:
            raise ValueError(f"A {task_type} adapter checkpoint is required for evaluation")
        model = load_stage_adapter(model, checkpoint_path)

    # PEFT 0.9 activates multiple LoRA modules through the underlying LoraModel.
    model.set_adapter("default")
    model.base_model.set_adapter(list(model.peft_config))
    for name, param in model.named_parameters():
        param.requires_grad = (
            not test_task and "lora_" in name and ".default." in name
        )

    return model, tokenizer
