# 🤖 BERT Fine-Tuning for Chatbot Question Classification

This project demonstrates how to fine-tune a pre-trained BERT model (from Hugging Face Transformers) on your own **chatbot dataset** stored in CSV format. It walks you through loading data, tokenizing, mapping labels, training the model, and saving the final result.

---

## 📁 Dataset Format

Make sure your CSV looks like this:

| Question                     | Answer         |
|-----------------------------|----------------|
| What is your name?          | greeting       |
| How can I reset my password?| account_help   |
| Tell me a joke              | entertainment  |

---

## 🛠️ Step-by-Step Instructions

### ✅ Step 0: Install Required Libraries
```bash
pip install -U transformers datasets pandas scikit-learn
````

---

### ✅ Step 1: Import Libraries

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
```

---

### ✅ Step 2: Load and Split Your CSV Dataset

```python
df = pd.read_csv("/content/chatbot_data.csv")
train_texts, test_texts, train_labels, test_labels = train_test_split(df['Question'], df['Answer'], test_size=0.2)
```

---

### ✅ Step 3: Save the Split Data

```python
train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
```

---

### ✅ Step 4: Load with Hugging Face Dataset Loader

```python
data_files = {"train": "train.csv", "test": "test.csv"}
dataset = load_dataset("csv", data_files=data_files)
```

---

### ✅ Step 5: Load Pretrained Tokenizer and Model

```python
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Label encoding
label_map = {label: i for i, label in enumerate(df['Answer'].unique())}
num_labels = len(label_map)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
```

---

### ✅ Step 6: Map Text Labels to Integers

```python
def map_labels_to_integers(example):
    example['label'] = label_map[example['label']]
    return example

dataset = dataset.map(map_labels_to_integers)
```

---

### ✅ Step 7: Tokenize the Texts

```python
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

---

### ✅ Step 8: Define Training Arguments

```python
training_args = TrainingArguments(
    output_dir="/content/results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)
```

---

### ✅ Step 9: Initialize the Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

---

### ✅ Step 10: Train the Model

```python
trainer.train()
```

---

### ✅ Step 11: Save the Trained Model

```python
trainer.save_model("fine_tuned_bert_model")
```

---

## 🚀 Example Inference

```python
text = "How do I change my password?"
inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)
predicted_label = outputs.logits.argmax().item()

# Map back to label
label_map_reverse = {v: k for k, v in label_map.items()}
print(label_map_reverse[predicted_label])
```

---

## 📌 Notes

* You can fine-tune on any text classification task by changing the dataset.
* Make sure your labels are clean and mapped properly.
* `num_labels` must match your number of unique answer categories.

---

## 📄 License

MIT License

---

## 💬 Questions or Suggestions?

Feel free to open an issue or fork this repo and contribute!

```
