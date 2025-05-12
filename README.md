# Language Model Instruction Fine-tuning

This repository contains code for fine-tuning language models on instruction datasets using both standard fine-tuning and LoRA (Low-Rank Adaptation) approaches.

## Project Overview

This project implements a complete LLM decoder-only architecture from scratch, with GPT-2 weights loaded for initialization. The custom implementation includes all core transformer components (attention mechanisms, feed-forward networks, layer normalization) while leveraging pre-trained GPT-2 weights for faster convergence. The codebase supports:

- Full custom decoder implementation with GPT-2 weight loading
- Standard full-parameter fine-tuning on the custom architecture
- LoRA (Low-Rank Adaptation) fine-tuning for parameter-efficient training
- Dataset preprocessing for instruction formats
- Training, validation, and testing workflows

## Repository Structure

```
├── instruction_finetune/       # Standard fine-tuning implementation
│   ├── instruction_finetune.py # Main implementation for full-parameter fine-tuning
│   └── RAG.py                  # Retrieval-Augmented Generation implementation
│
├── instruction_with_lora/      # LoRA fine-tuning implementation
│   ├── instruction_finetune_lora.py  # Main implementation for LoRA fine-tuning
│   └── continue_training.py    # Script for continuing training from checkpoints
│
├── dataset_preprocess/         # Dataset preprocessing utilities
│   ├── preprocess_instructions.py  # Instruction dataset preprocessing
│   ├── preprocess_dataset.py       # General dataset preprocessing
│   └── finance_data_scrapper.py    # Finance domain data collection
│
└── datasets/                   # Dataset storage
    ├── instructions.json                      # Raw instruction dataset
    ├── processed_instruction_dataset.json     # Processed full dataset
    └── processed_instruction_dataset_sample.json  # Sample dataset for testing
```

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Standard Fine-tuning

To run standard fine-tuning on the instruction dataset:

```bash
cd instruction_finetune
python instruction_finetune.py
```

### LoRA Fine-tuning

To run LoRA fine-tuning (more parameter-efficient):

```bash
cd instruction_with_lora
python instruction_finetune_lora.py
```

To continue training from a checkpoint:

```bash
cd instruction_with_lora
python continue_training.py
```

### Dataset Preprocessing

To preprocess raw instruction datasets:

```bash
cd dataset_preprocess
python preprocess_instructions.py
```

## Model Architecture

The project implements a complete decoder-only transformer architecture from scratch, including:
- Multi-head self-attention mechanisms
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Token embedding and positional encoding layers

The implementation then loads pre-trained GPT-2 weights for initialization, allowing the model to benefit from pre-trained knowledge while maintaining full control over the architecture. The instruction fine-tuning process adapts this custom implementation to follow specific instructions and generate appropriate responses.

## Training Process

The training process includes:

1. Loading and preprocessing instruction datasets
2. Tokenizing and formatting inputs with instruction templates
3. Training with causal language modeling objective
4. Validation and evaluation on held-out data
5. Saving model checkpoints

## License

This project is available for educational and research purposes.

## Acknowledgements

This implementation draws inspiration from various instruction fine-tuning approaches in the field of natural language processing.