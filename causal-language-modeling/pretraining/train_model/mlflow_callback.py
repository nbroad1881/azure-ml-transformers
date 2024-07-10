import os
import json
import logging
import packaging.version

from transformers.integrations import MLflowCallback
from transformers.utils import ENV_VARS_TRUE_VALUES

from transformers.utils import flatten_dict

logger = logging.getLogger(__name__)

class AzureMLflowCallback(MLflowCallback):
    """
    In Azure ML, mlflow cannot log more than 200 parameters and TrainingArguments has more than that.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ignored_parameters = [
            "adafactor",
            "disable_tqdm",
            "do_predict",
            "eval_accumulation_steps",
            "eval_do_concat_batches",
            "eval_on_start",
            "fp16_opt_level",
            "group_by_length",
            "hub_always_push",
            "hub_model_id",
            "hub_private_repo",
            "hub_strategy",
            "hub_token",
            "include_inputs_for_metrics",
            "include_num_input_tokens_seen",
            "include_tokens_per_second",
            "label_names",
            "label_smoothing_factor",
            "length_column_name",
            "logging_first_step",
            "logging_nan_inf_filter",
            "logging_strategy",
            "mp_parameters",
            "neftune_noise_alpha",
            "no_cuda",
            "per_gpu_eval_batch_size",
            "per_gpu_train_batch_size",
            "push_to_hub",
            "push_to_hub_model_id",
            "push_to_hub_organization",
            "push_to_hub_token",
            "ray_scope",
            "remove_unused_columns",
            "restore_callback_states_from_checkpoint",
            "save_on_each_node",
            "save_only_model",
            "save_safetensors",
            "skip_memory_metrics",
            "tpu_metrics_debug",
            "use_cpu",
            "use_ipex",
            "use_legacy_prediction_loop",
            "use_mps_device",
            # model config
            "output_attentions",
            "output_hidden_states",
            "tie_encoder_decoder",
            "tie_word_embeddings",
            # generation args
            "num_beams",
            "output_scores",
            "repetition_penalty",
            "return_dict_in_generate",
            "top_k",
            "top_p",
            "temperature",
            "use_cache",
        ]

    def setup(self, args, state, model):
        """
        Setup the optional MLflow integration.

        Environment:
        - **HF_MLFLOW_LOG_ARTIFACTS** (`str`, *optional*):
            Whether to use MLflow `.log_artifact()` facility to log artifacts. This only makes sense if logging to a
            remote server, e.g. s3 or GCS. If set to `True` or *1*, will copy each saved checkpoint on each save in
            [`TrainingArguments`]'s `output_dir` to the local or remote artifact storage. Using it without a remote
            storage will just copy the files to your artifact location.
        - **MLFLOW_TRACKING_URI** (`str`, *optional*):
            Whether to store runs at a specific path or remote server. Unset by default, which skips setting the
            tracking URI entirely.
        - **MLFLOW_EXPERIMENT_NAME** (`str`, *optional*, defaults to `None`):
            Whether to use an MLflow experiment_name under which to launch the run. Default to `None` which will point
            to the `Default` experiment in MLflow. Otherwise, it is a case sensitive name of the experiment to be
            activated. If an experiment with this name does not exist, a new experiment with this name is created.
        - **MLFLOW_TAGS** (`str`, *optional*):
            A string dump of a dictionary of key/value pair to be added to the MLflow run as tags. Example:
            `os.environ['MLFLOW_TAGS']='{"release.candidate": "RC1", "release.version": "2.2.0"}'`.
        - **MLFLOW_NESTED_RUN** (`str`, *optional*):
            Whether to use MLflow nested runs. If set to `True` or *1*, will create a nested run inside the current
            run.
        - **MLFLOW_RUN_ID** (`str`, *optional*):
            Allow to reattach to an existing run which can be usefull when resuming training from a checkpoint. When
            `MLFLOW_RUN_ID` environment variable is set, `start_run` attempts to resume a run with the specified run ID
            and other parameters are ignored.
        - **MLFLOW_FLATTEN_PARAMS** (`str`, *optional*, defaults to `False`):
            Whether to flatten the parameters dictionary before logging.
        """
        self._log_artifacts = (
            os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper()
            in ENV_VARS_TRUE_VALUES
        )
        self._nested_run = (
            os.getenv("MLFLOW_NESTED_RUN", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        )
        self._tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
        self._experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", None)
        self._flatten_params = (
            os.getenv("MLFLOW_FLATTEN_PARAMS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        )
        self._run_id = os.getenv("MLFLOW_RUN_ID", None)

        # "synchronous" flag is only available with mlflow version >= 2.8.0
        # https://github.com/mlflow/mlflow/pull/9705
        # https://github.com/mlflow/mlflow/releases/tag/v2.8.0
        self._async_log = packaging.version.parse(
            self._ml_flow.__version__
        ) >= packaging.version.parse("2.8.0")

        logger.debug(
            f"MLflow experiment_name={self._experiment_name}, run_name={args.run_name}, nested={self._nested_run},"
            f" tags={self._nested_run}, tracking_uri={self._tracking_uri}"
        )
        if state.is_world_process_zero:
            if not self._ml_flow.is_tracking_uri_set():
                if self._tracking_uri:
                    self._ml_flow.set_tracking_uri(self._tracking_uri)
                    logger.debug(f"MLflow tracking URI is set to {self._tracking_uri}")
                else:
                    logger.debug(
                        "Environment variable `MLFLOW_TRACKING_URI` is not provided and therefore will not be"
                        " explicitly set."
                    )
            else:
                logger.debug(
                    f"MLflow tracking URI is set to {self._ml_flow.get_tracking_uri()}"
                )

            if self._ml_flow.active_run() is None or self._nested_run or self._run_id:
                if self._experiment_name:
                    # Use of set_experiment() ensure that Experiment is created if not exists
                    self._ml_flow.set_experiment(self._experiment_name)
                self._ml_flow.start_run(run_name=args.run_name, nested=self._nested_run)
                logger.debug(
                    f"MLflow run started with run_id={self._ml_flow.active_run().info.run_id}"
                )
                self._auto_end_run = True
            combined_dict = args.to_dict()
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            combined_dict = (
                flatten_dict(combined_dict) if self._flatten_params else combined_dict
            )
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f'Trainer is attempting to log a value of "{value}" for key "{name}" as a parameter. MLflow\'s'
                        " log_param() only accepts values no longer than 250 characters so we dropped this attribute."
                        " You can use `MLFLOW_FLATTEN_PARAMS` environment variable to flatten the parameters and"
                        " avoid this message."
                    )
                    del combined_dict[name]
                elif name in self.ignored_parameters:
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(
                0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH
            ):
                if self._async_log:
                    self._ml_flow.log_params(
                        dict(
                            combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]
                        ),
                        synchronous=False,
                    )
                else:
                    self._ml_flow.log_params(
                        dict(
                            combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]
                        )
                    )
            mlflow_tags = os.getenv("MLFLOW_TAGS", None)
            if mlflow_tags:
                mlflow_tags = json.loads(mlflow_tags)
                self._ml_flow.set_tags(mlflow_tags)
        self._initialized = True
