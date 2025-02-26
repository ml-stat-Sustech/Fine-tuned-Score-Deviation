import os

import argparse
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import SFTTrainer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    logging,
    set_seed,
)

def get_dataset(args):
    if args.dataset_name == "WikiMIA":
        length = [32, 64, 128, 256]
        format_dataset = {"text": [], "label": []}
        for l in length:
            dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{l}")
            dataset.shuffle(seed=args.seed)
            if args.train_size < 1:
                split = dataset.train_test_split(
                    train_size=args.train_size, seed=args.seed
                )
                dataset = split["train"]
            for i in range(len(dataset)):
                format_dataset["text"].append(dataset[i]["input"])
                format_dataset["label"].append(dataset[i]["label"])
        dataset = Dataset.from_dict(format_dataset)
    else:
        raise ValueError("Please check if the dataset name is valid.")
    return dataset


def prompts(examples):
    output_text = []
    for i in range(len(examples["text"])):
        if (
            examples["label"][i] == 1
        ):  #  Choose non-members for fine-tuning the pre-trained model
            continue
        input_text = examples["text"][i]
        text = input_text
        output_text.append(text)
    return output_text


def print_trainable_params(model):
    """
    Print the number of trainable parameters in the model
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
    )


def run_training(args, train_data, tokenzier):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    train_data.start_iteration = 0
    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.ckpts_dir,
        dataloader_drop_last=True,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=args.run_name,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, load_in_8bit=True, device_map={"": Accelerator().process_index}
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenzier,
        max_seq_length=args.seq_length,
        args=training_args,
        eval_dataset=train_data,
        train_dataset=train_data,
        peft_config=lora_config,
        formatting_func=prompts,
        packing=False,
    )

    print_trainable_params(trainer.model)

    print("Training...")
    trainer.train()
    return trainer.model


def main(args):
    exp_dir = os.path.join(
        args.ckpts_dir,
        args.model_name.split("/")[-1],
        "seed_" + str(args.seed),
        args.dataset_name,
    )
    os.makedirs(exp_dir, exist_ok=True)
    train_dataset = get_dataset(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = run_training(args, train_dataset, tokenizer)
    print("Saving last checkpoint of the model")
    model.save_pretrained(exp_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Supervised Fintuning with PEFT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="WikiMIA")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ckpts_dir", type=str, default="./ckpts")
    parser.add_argument(
        "--model_name",
        type=str,
        default="huggyllama/llama-7b",
    )

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--run_name", type=str, default="finetune")
    parser.add_argument("--train_size", type=float, default=0.3)
    args = parser.parse_args()

    set_seed(args.seed)
    logging.set_verbosity_info()

    main(args)
