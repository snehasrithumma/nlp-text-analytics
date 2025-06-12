from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    # Check for GPU
    print("GPU available:", torch.cuda.is_available())

    # Load dataset
    dataset = load_dataset("yelp_review_full")

    model_name = "distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize, batched=True)

    # Select smaller subsets for quick iteration
    small_train = dataset["train"].shuffle(seed=42).select(range(1000))
    small_eval = dataset["test"].shuffle(seed=42).select(range(1000))

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./huggingface_models/yelp_review_classifier",
        eval_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,  # Increase after debugging
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train,
        eval_dataset=small_eval,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Train model
    trainer.train()

    # Load pipeline for inference
    classification_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Test with an example
    sample = "The service was absolutely fantastic and the food was delicious!"
    result = classification_pipeline(sample)
    print("Classification result:", result)


if __name__ == "__main__":
    main()
