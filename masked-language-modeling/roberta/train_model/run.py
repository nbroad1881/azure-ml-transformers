import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path

import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from collator import DataCollatorForOnlyMaskingLanguageModeling
from modeling_roberta import RobertaForMaskedLM

logger = logging.getLogger(__name__)

def setup_logging(config):
    # Setup logging
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

    config_name_or_path: str = field(type=str, default="roberta-base")
    model_name_or_path: str = field(type=str, default="roberta-base")
    tokenizer_name_or_path: str = field(type=str, default="roberta-base")

    dataset_directory: str = field(type=str, default="data")
    dataset_file_glob: str = field(type=str, default="*.parquet")

    validation_split_percentage: float = field(type=float, default=0.01)

    masking_probability: float = field(type=float, default=0.2)

    attn_implementation: str = field(
        type=str,
        default="sdpa",
        metadata={
            "help": "choose from {'eager', 'sdpa', 'flash_attention_2'}. Not all models have sdpa or flash_attention_2."
        },
    )


def main():

    arg_parser = HfArgumentParser((Config,))

    config = arg_parser.parse_args()

    setup_logging(config)

    tokenized_files = [
        str(x) for x in Path(config.dataset_directory).rglob(config.dataset_file)
    ]

    ds = load_dataset("parquet", data_files=tokenized_files)

    ds = ds.train_test_split(test_size=config.validation_split_percentage)

    num_train_samples = len(ds["train"])
    num_eval_samples = len(ds["test"])

    logger.info(f"Number of training samples: {num_train_samples}")
    logger.info(f"Number of evaluation samples: {num_eval_samples}")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)

    model_config = AutoConfig.from_pretrained(config.config_name_or_path)
    model_config.vocab_size = len(tokenizer)

    model = RobertaForMaskedLM.from_pretrained(
        config.model_name_or_path,
        config=model_config,
        attn_implementation=config.attn_implementation,
    )

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    collator = DataCollatorForOnlyMaskingLanguageModeling(
        tokenizer, mlm_probability=config.masking_probability
    )

    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        preprocess_logits_for_metrics=None,
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
