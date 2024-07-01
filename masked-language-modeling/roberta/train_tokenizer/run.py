# This is a very basic way to retrain an existing tokenizer on a new dataset.
# See here for the Hugging Face Course on Tokenizers: https://huggingface.co/learn/nlp-course/chapter6/1?fw=pt

from argparse import ArgumentParser

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("--tokenizer_name", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--vocab_size", type=int, default=32768, help="Size of the new vocabulary.")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", type=str, default="CC-MAIN-2024-10")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--streaming", default=False, action="store_true")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_dir", type=str, default="output")

    return parser.parse_args()


def main():

    args = parse_args()

    # Note: Instead of streaming, you could also load from local files
    # See hf.co/docs/datasets for more info
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=args.streaming,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    def text_generator(num_samples, batch_size=1000):
        batch = []
        for i, sample in enumerate(iter(dataset)):
            if i == num_samples:
                break
            batch.append(sample["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []


    new_tokenizer = tokenizer.train_new_from_iterator(
        text_generator(args.num_samples), vocab_size=args.vocab_size
    )

    new_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
