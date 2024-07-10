import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from functools import partial
import random

import datasets
from datasets import load_dataset, DatasetDict
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    set_seed,
)
import evaluate

from modeling_roberta import RobertaForMaskedLM
from mlflow_callback import AzureMLflowCallback

logger = logging.getLogger(__name__)


def setup_logging(config):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if config.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = config.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {config.local_rank}, device: {config.device}, n_gpu: {config.n_gpu}, "
        + f"distributed training: {config.parallel_mode.value == 'distributed'}, 16-bits training: {config.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {config}")


@dataclass
class Config(TrainingArguments):

    config_name_or_path: str = field(
        default="roberta-base", metadata={"help": "Name or path of the model config."}
    )
    model_name_or_path: str = field(
        default=None, metadata={"help": "If None, trains from scratch."}
    )
    tokenizer_name_or_path: str = field(
        default="roberta-base",
        metadata={"help": "Name or path of the tokenizer to use."},
    )

    tokenized_files_dir: str = field(
        default="data", metadata={"help": "Directory containing the dataset files."}
    )
    glob_pattern: str = field(
        default="*.parquet",
        metadata={"help": "Glob pattern to match files in the `tokenized_files_dir`."},
    )

    validation_split_num_samples_or_percentage: float = field(
        default=0.01,
        metadata={
            "help": "If > 1, this is the number of samples to use for validation. If < 1, this is the percentage of samples to use for validation."
        },
    )

    masking_probability: float = field(
        default=0.15, metadata={"help": "Probability of masking tokens."}
    )

    attn_implementation: str = field(
        default="sdpa",
        metadata={
            "help": "choose from {'eager', 'sdpa', 'flash_attention_2'}. Not all models have sdpa or flash_attention_2."
        },
    )


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds, metric):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)




def main():

    arg_parser = HfArgumentParser(Config)

    config = arg_parser.parse_args_into_dataclasses()[0]

    set_seed(config.seed)
    setup_logging(config)

    tokenized_files = list(
        map(str, Path(config.tokenized_files_dir).rglob(config.glob_pattern))
    )

    ds = load_dataset("parquet", data_files=tokenized_files, split="train")

    if config.validation_split_num_samples_or_percentage > 1:

        if len(ds)//20 < config.validation_split_num_samples_or_percentage:
            num2sample = len(ds)//20
        else:
            num2sample = int(config.validation_split_num_samples_or_percentage)


        val_indices = random.sample(
            range(len(ds)), k=num2sample
        )
        train_indices = [i for i in range(len(ds)) if i not in val_indices]

        ds = DatasetDict(
            {
                "train": ds.select(train_indices),
                "test": ds.select(val_indices),
            }
        )

    else:
        ds = ds.train_test_split(test_size=config.validation_split_num_samples_or_percentage)

    num_train_samples = len(ds["train"])
    num_eval_samples = len(ds["test"])

    logger.info(f"Number of training samples: {num_train_samples}")
    logger.info(f"Number of evaluation samples: {num_eval_samples}")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)

    model_config = AutoConfig.from_pretrained(
        config.config_name_or_path,
        attn_implementation=config.attn_implementation,
        vocab_size = len(tokenizer),
    )

    if config.model_name_or_path is not None:
        # This loads a pretrained model
        model = RobertaForMaskedLM.from_pretrained(
            config.model_name_or_path, config=model_config
        )
    else:
        # This creates a model from scratch
        model = RobertaForMaskedLM(model_config)

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm_probability=config.masking_probability
    )

    metric = evaluate.load("accuracy")

    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=partial(compute_metrics, metric=metric),
        callbacks=[AzureMLflowCallback()],
    )

    if config.do_train:
        logger.info("*** Training ***")

        checkpoint = None
        if config.resume_from_checkpoint:
            checkpoint = config.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if config.do_eval:

        logger.info("*** Evaluation ***")

        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)


if __name__ == "__main__":
    main()
