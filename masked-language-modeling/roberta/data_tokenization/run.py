from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Name or path of the tokenizer to use.")
    parser.add_argument("--file_type", type=str, default="parquet", help="Type of the input files. One of {'parquet', 'csv', 'json'}.")
    parser.add_argument("--text_file_dir", type=str, default="text-files", help="Directory containing text files.")
    parser.add_argument("--glob_pattern", type=str, default="*.parquet", help="Glob pattern to match files in the `text_file_dir`.")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the column containing text data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the tokenized files.")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes to use with the `map` function.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max number of tokens per input sequence.")
    parser.add_argument("--max_char_length", type=int, default=10000, help="Max number of characters per tokenization. Tokenizers can be slow when working with long texts, thus breaking it into smaller chunks can be more efficient.")
    parser.add_argument("--file_num_samples", type=int, default=100000, help="Number of samples per output file")

    return parser.parse_args()


def main():

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)


    all_files = list(map(str, Path(args.text_file_dir).rglob(args.glob_pattern)))

    ds = load_dataset(
        args.file_type,
        data_files=all_files,
        split="train",
    )

    def divide_texts(batch):
        """
        Tokenization can be slow on long texts, so this function divides the texts into smaller chunks.
        This may split words in the middle (but so could `group_texts`), so it's not perfect, but it's a simple way to speed up tokenization.
        """
        divided_texts = []
        for text in batch[args.text_column]:
            divided_texts.extend(
                [
                    text[i : i + args.max_char_length]
                    for i in range(0, len(text), args.max_char_length)
                ]
            )

        return {"text": divided_texts}

    def tokenize(batch):
        return tokenizer(batch["text"], return_special_tokens_mask=True)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // args.max_seq_length) * args.max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + args.max_seq_length]
                for i in range(0, total_length, args.max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result


    ds = ds.map(
        divide_texts,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
        desc=f"Dividing texts into chunks of max_char_length ({args.max_char_length})"
    )

    ds = ds.map(tokenize, batched=True, num_proc=args.num_proc, desc="Tokenizing texts", remove_columns=ds.column_names)

    ds = ds.map(
        group_texts,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
        desc=f"Grouping texts into chunks of max_seq_length ({args.max_seq_length})"
    )

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)


    for start_idx in range(0, len(ds), args.file_num_samples):
        end_idx = min(start_idx + args.file_num_samples, len(ds))

        ds.select(range(start_idx, end_idx)).to_parquet(
            str(output_dir / f"tokenized_{start_idx}-{end_idx}.parquet")
        )


if __name__ == "__main__":
    main()
