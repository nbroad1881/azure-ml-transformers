# This is a very basic way to retrain an existing tokenizer on a new dataset.
# See here for the Hugging Face Course on Tokenizers: https://huggingface.co/learn/nlp-course/chapter6/1?fw=pt

import logging
import multiprocessing
from pathlib import Path
from argparse import ArgumentParser

from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def parse_args():

    parser = ArgumentParser()

    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the tokenizer to use. The vocabulary will be retrained, but the underlying tokenizing algorithm will be unchanged.")
    parser.add_argument("--vocab_size", type=int, default=98304, help="Size of the new vocabulary.")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu", help="Name of the dataset to use from Hugging Face Hub.")
    parser.add_argument("--dataset_config", type=str, default="CC-MAIN-2024-10", help="Configuration of the dataset to use from Hugging Face Hub.")
    parser.add_argument("--dataset_split", type=str, default="train", help="Split of the dataset to use from Hugging Face Hub.")
    parser.add_argument("--streaming", default=False, action="store_true", help="Whether to stream the dataset.")
    parser.add_argument("--text_files_dir", type=str, default="text-files", help="Directory containing text files.")
    parser.add_argument("--glob_pattern", type=str, default=None, help="Glob pattern to match files in the `text_files_dir`.")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the column containing text data.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use for training the tokenizer.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the new tokenizer.")
    parser.add_argument("--num_proc", type=int, default=-1, help="Number of processes to use with the `map` function. If -1, will use all available CPUs.")

    return parser.parse_args()


def main():

    args = parse_args()

    if args.num_proc == -1:
        args.num_proc = multiprocessing.cpu_count()

        logger.info("Using %s processes", args.num_proc)

    # Load files from local directory
    if args.text_files_dir is not None:
        if "." not in args.glob_pattern:
            # assume these are parquet files
            file_ext = "parquet"
        else:
            file_ext = args.glob_pattern.split(".")[-1]

        data_files = list(map(str, Path(args.text_files_dir).rglob(args.glob_pattern)))

        dataset = load_dataset(
            file_ext,
            data_files=data_files,
            split="train",
            num_proc=args.num_proc,
        )

    # Pull data from HF Hub
    else:
        
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
            streaming=args.streaming,
        )


    if args.streaming:
        dataset_iterator = iter(dataset) 
    else:
        dataset_iterator = dataset

    def text_generator(num_samples, batch_size=1000):
        batch = []
        for i, sample in enumerate(dataset_iterator):
            if i == num_samples:
                break
            batch.append(sample[args.text_column])
            if len(batch) == batch_size:
                yield batch
                batch = []

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    
    new_tokenizer = tokenizer.train_new_from_iterator(
        text_generator(args.num_samples), vocab_size=args.vocab_size
    )

    new_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
