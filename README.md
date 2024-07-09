# Fine-tuning Mistral 7B Instruct Model for English-to-Telugu Translation

This repository contains the implementation of fine-tuning the Mistral 7B Instruct model for translating English words and sentences to Telugu using the QLoRA 4-bit technique for instruction fine-tuning.

## Project Overview

The goal of this project was to leverage the Mistral 7B Instruct model for accurate English-to-Telugu translation. We utilized a robust dataset consisting of 140k data points for training and 16k data points for testing to ensure high-quality translations.

## Repository Structure

- `Dataset/`: Directory containing the dataset used for training and testing.
- `Model_Weights/`: Directory for storing trained model weights.
- `wandb/`: Directory for storing Weights & Biases logs.
- `IFT_Mistral_7B.ipynb`: Jupyter notebook containing the code for training the Mistral 7B model.
- `inference.ipynb`: Jupyter notebook for performing inference with the trained model.
- `README.md`: This file.
- `requirements.txt`: File listing the Python libraries required to run the code.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository

2. **Installing Dependencies:**

   Ensure you have Python installed. Install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
   Note: For libraries requiring CUDA (like Flash Attention), ensure you have CUDA installed on your system.
   
3. **Run Inference:**

   Open inference.ipynb in Jupyter Notebook or JupyterLab. Update the path to the model weights (Model_Weights/) as necessary. Execute the notebook to perform inference and try out translating English statements to Telugu.

## Using the Model from Hugging Face:

You can simply use the model from Hugging Face with the following code:
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("MRR24/Translator_Eng_Tel_instruct")
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base_model, "MRR24/Translator_Eng_Tel_instruct")
```
## Dataset

  The dataset used in this project was sourced from Socionoftech/Sai kumar Yava. A shoutout to them for providing the dataset.

