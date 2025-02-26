import zlib
import argparse
from utils.utils import seed_everything
from utils.tools import *
from tqdm import tqdm
import torch
import numpy as np
from utils.tools import fig_fpr_tpr
from peft import PeftConfig, PeftModel
from datasets import load_dataset
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(args, fine_tuning):

    if not fine_tuning:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, return_dict=True, device_map="auto"
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model_path = args.model_path
    elif fine_tuning:
        config = PeftConfig.from_pretrained(args.fine_tuned_para)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, device_map="auto"
        )
        lora_model = PeftModel.from_pretrained(model, args.fine_tuned_para)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = lora_model
        model.eval()
        model_path = args.fine_tuned_para
    print("model path: ", model_path)
    return model, tokenizer


def inference(model, tokenizer, sentence, example):
    pred = {}
    p1, all_prob, p1_likelihood = calculatePerplexity(
        sentence, model, tokenizer, gpu=model.device
    )

    p_lower, _, p_lower_likelihood = calculatePerplexity(
        sentence.lower(), model, tokenizer, gpu=model.device
    )

    pred["ppl"] = p1  # ppl

    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()

    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(sentence, "utf-8")))
    pred["ppl/zlib"] = np.log(p1) / zlib_entropy

    # min-k
    ratio = 0.2
    k_length = int(len(all_prob) * ratio)
    topk_prob = np.sort(all_prob)[:k_length]
    pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()
    example["pred"] = pred
    return example


def get_dataset(args):
    if args.dataset_name == "WikiMIA":
        length = [32, 64, 128, 256]
        dataset_all = []
        for l in length:
            dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{l}")
            dataset.shuffle(
                seed=args.seed
            )  # ! seed must be same as the seed in the sft.py
            if args.train_size < 1:
                split = dataset.train_test_split(
                    train_size=args.train_size, seed=args.seed
                )  # ! seed must be same as the seed in the sft.py
                if args.split == "test":
                    dataset = split["test"]
                elif args.split == "train":
                    dataset = split["train"]
            for i in range(len(dataset)):
                dict_text = {"text": dataset[i]["input"], "label": dataset[i]["label"]}
                dataset_all.append(dict_text)
    else:
        raise ValueError("Please check if the dataset name is valid.")
    print("dataset numbers: ", len(dataset_all))
    return dataset_all


def eval(data, args=None):
    model, tokenizer = load_model(args, fine_tuning=False)
    output_all = []
    for example in tqdm(data):
        text = example["text"]
        new_ex = inference(model, tokenizer, text, example)
        output_all.append(new_ex)

    model, tokenizer = load_model(args, fine_tuning=True)
    data_copy = copy.deepcopy(data)
    output_pretrained = []
    for example in tqdm(data_copy):
        text = example["text"]
        new_ex = inference(model, tokenizer, text, example)
        output_pretrained.append(new_ex)
    assert len(output_all) == len(
        output_pretrained
    ), "The length of the two lists must be equal"
    for i in range(len(output_all)):
        for metric in output_all[i]["pred"].keys():
            output_all[i]["pred"][metric] = (
                output_pretrained[i]["pred"][metric] - output_all[i]["pred"][metric]
            )

    fig_fpr_tpr(output_all)
    return output_all


def calculatePerplexity(sentence, model, tokenizer, gpu):

    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]  # loss, scale

    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]

    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="WikiMIA",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="huggyllama/llama-7b",
        help="the model to infer",
    )

    parser.add_argument("--fine_tuned_para", default="")

    parser.add_argument("--dataset", type=str, default="WikiMIA")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--train_size", type=float, default=0.3)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    seed_everything(args.seed)
    data = get_dataset(args)
    eval(data, args=args)
