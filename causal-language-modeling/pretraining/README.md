# CLM Pretraining for small language models (<2B)


You will need to 
1. add your Azure ML config.json file to this directory
2. point the script to the correct text files
3. point the script to a trained tokenizer (should include a config.json too that has the right vocab size)
4. modify compute instances, hyperparameters to suit your needs
5. modify the deepspeed_config.json depending on what you want to offload


[run_pipeline.ipynb](./run_pipeline.ipynb) has everything to run the training.

## Update: 2024-02-27

Mistral added. Flash attention 2 added.

When doing a small test on a 1B mistral model, here are the differences that flash attention 2 can make.

For both tests, the batch size is 4, sequence length is 4096, gradient accumulation is 4, gradient checkpointing is turned on.

Attention | Memory Used | Samples/second |
| --- | --- | --- |
Without flash attention | 59.5GB | 0.8 |
With flash attention | 23.9GB | 3.2 |