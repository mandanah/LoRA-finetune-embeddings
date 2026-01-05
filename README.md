# 🎯 LoRA Fine-tuning for Fashion Embeddings

A lightweight implementation for fine-tuning BERT embeddings using LoRA (Low-Rank Adaptation) to create semantic embeddings for fashion product queries and descriptions.

## 📋 Overview

This project demonstrates how to fine-tune a pre-trained language model (BERT) using LoRA to create embeddings that understand the relationship between short fashion queries and detailed product descriptions. The model learns to map similar items close together in embedding space using contrastive learning.

## 🧠 Key Concepts

### LoRA (Low-Rank Adaptation)
Instead of fine-tuning all model parameters, LoRA adds small trainable matrices to specific layers. This:
- ✅ Dramatically reduces trainable parameters (only ~0.1% of the full model)
- ✅ Speeds up training and reduces memory usage
- ✅ Makes it easy to share and switch between different adaptations

### Multiple Negatives Ranking Loss
A contrastive learning approach where:
- Each anchor (query) has one positive (matching description)
- All other items in the batch serve as negatives
- The model learns to pull anchors closer to their positives and push away from negatives

## 🏗️ Project Structure

```
├── main.py                  # Training orchestration
├── model.py                 # Model architecture with LoRA
├── config.py               # Hyperparameters and settings
├── loss_function.py        # Contrastive loss implementation
├── collator.py             # Batch preparation
├── schema.py               # Data structures
├── data/
│   ├── dataset.py          # PyTorch Dataset wrapper
│   └── load_toy_data.py    # Sample fashion data
└── lora_fashion_embedder/  # Saved model artifacts
```

## 🔄 Training Pipeline

### 1️⃣ Data Preparation
- **Input**: Pairs of (short query, detailed description)
- **Example**: `"black maxi dress"` → `"women's black maxi dress, sleeveless, flowy silhouette, ankle length"`
- **Dataset**: 10 fashion item pairs for demonstration

### 2️⃣ Model Setup
- **Base Model**: `bert-base-uncased` (110M parameters)
- **LoRA Config**:
  - Rank (r): 4
  - Alpha: 8
  - Target modules: `query` and `value` attention layers
  - Dropout: 0.05
- **Pooling**: Mean pooling over token embeddings
- **Normalization**: L2-normalized embeddings for similarity comparison

### 3️⃣ Training Process
- **Batch Processing**: Tokenize anchor and positive texts separately
- **Forward Pass**: Generate normalized embeddings for both
- **Loss Calculation**: Multiple negatives ranking loss with temperature scaling (0.05)
- **Optimization**: AdamW optimizer with learning rate 2e-5
- **Scheduler**: Linear warmup (5% of steps) + decay
- **Gradient Clipping**: Max norm of 1.0 to prevent instability

### 4️⃣ Training Loop
```
For each epoch:
  ├── Shuffle data
  ├── Process batches
  │   ├── Encode anchors & positives
  │   ├── Compute contrastive loss
  │   ├── Backpropagate
  │   └── Update LoRA parameters only
  ├── Calculate average loss
  └── Save checkpoint
```

### 5️⃣ Output
- **Checkpoints**: `lora_finetuned_epoch{1,2,3}.pth` - Full model states per epoch
- **Final Model**: `lora_fashion_embedder/` - LoRA adapter weights + tokenizer
  - `adapter_model.safetensors` - Efficient LoRA weights
  - `adapter_config.json` - LoRA configuration
  - Tokenizer files for inference

## ⚙️ Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 3 | Training iterations over dataset |
| Batch Size | 4 | Pairs per batch |
| Learning Rate | 2e-5 | AdamW learning rate |
| Max Length | 128 | Maximum token sequence length |
| Temperature | 0.05 | Contrastive loss temperature |

## 🚀 Usage

### Training
```bash
uv run python main.py
```

### Loading the Fine-tuned Model
```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# Load base model
base_model = AutoModel.from_pretrained("bert-base-uncased")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./lora_fashion_embedder")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./lora_fashion_embedder")

# Generate embeddings
text = "blue summer dress"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
# Apply mean pooling and normalization as in model.py
```

## 📊 What the Model Learns

The model learns to:
- 🔍 Map similar semantic concepts together (e.g., "dress" ↔ "women's dress")
- 🎨 Understand fashion attributes (colors, styles, materials)
- 📏 Bridge the gap between short queries and detailed descriptions
- 🎯 Distinguish between different product categories

## 💡 Key Features

- **Efficient Fine-tuning**: Only trains subset of parameters instead of all
- **Portable Adapters**: Share just the small adapter weights (~2MB)
- **Production Ready**: Easy to deploy and inference
- **Modular Design**: Clean separation of concerns

## 🔧 Dependencies

- `torch` - Deep learning framework
- `transformers` - Hugging Face transformers library
- `peft` - Parameter-Efficient Fine-Tuning library
- `tqdm` - Progress bars

## 📝 Notes

This is a **toy example** with only 10 training pairs. In a real scenario:
- Use thousands of query-description pairs
- Add validation set for monitoring
- Implement early stopping
- Add data augmentation
- Tune hyperparameters (LoRA rank, learning rate, temperature)

