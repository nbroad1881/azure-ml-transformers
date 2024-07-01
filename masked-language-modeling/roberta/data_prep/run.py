from argparse import ArgumentParser
from itertools import chain

from transformers import AutoTokenizer
from datasets import load_dataset


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, required=True)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--text_column", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_char_length", type=int, default=10000)
    parser.add_argument("--file_num_samples", type=int, default=100000)

    return parser.parse_args()


def main():

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=args.streaming,
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
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
    )

    ds = ds.map(tokenize, batched=True, num_proc=args.num_proc)

    ds = ds.map(
        group_texts,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
    )

    for start_idx in range(0, len(ds), args.file_num_samples):
        end_idx = min(start_idx + args.file_num_samples, len(ds))

        ds.select(range(start_idx, end_idx)).to_parquet(
            f"{args.output_dir}/tokenized_{start_idx}-{end_idx}.parquet"
        )


if __name__ == "__main__":
    main()
