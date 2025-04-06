# Fine-Tuning BERT for Sentiment Analysis using LoRA

This implementation demonstrates how to fine-tune a pre-trained BERT model for sentiment analysis using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685), which is a parameter-efficient fine-tuning technique.

## 📚 Dataset

We use a truncated IMDb sentiment analysis dataset containing 1000 samples for training and validation:

```python
from datasets import load_dataset

dataset = load_dataset('shawhin/imdb-truncated')
```

The dataset has the following format:

```text
DatasetDict({
    train: Dataset({
        features: ['label', 'text'],
        num_rows: 1000
    })
    validation: Dataset({
        features: ['label', 'text'],
        num_rows: 1000
    })
})
```

## 🧠 Model and Tokenizer

We use the `distilbert-base-uncased` model from Hugging Face Transformers.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# Add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
```

## 🛠️ Tokenization

Tokenize the dataset using the pretrained tokenizer:

```python
def tokenize_function(examples):
    # Extract text
    text = examples["text"]

    # Tokenize and Truncate Text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset
```

## 🤖 Applying LoRA

LoRA allows you to fine-tune large language models efficiently by introducing trainable low-rank matrices into each layer.

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, peft_config)
```

## 🧑‍🏫 Training with Hugging Face Trainer

Set up training arguments and use the `Trainer` class for fine-tuning.

```python
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

training_args = TrainingArguments(
    output_dir= model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

trainer.train()
```

# Project overview and usage
```

Happy fine-tuning! 🚀
```
