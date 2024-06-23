---
base_model: mistralai/Mistral-7B-Instruct-v0.2
datasets:
- generator
library_name: peft
license: apache-2.0
tags:
- trl
- sft
- generated_from_trainer
model-index:
- name: Translator_Eng_Tel_instruct
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Translator_Eng_Tel_instruct

This model is a fine-tuned version of [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) on the generator dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2120

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 2

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.2549        | 0.9992 | 956  | 0.2554          |
| 0.2138        | 1.9984 | 1912 | 0.2120          |


### Framework versions

- PEFT 0.10.0
- Transformers 4.40.1
- Pytorch 2.1.1+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1