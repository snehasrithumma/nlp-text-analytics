from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from collections import Counter
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "..", "huggingface_models/ner_model")
import evaluate
metric = evaluate.load("seqeval")

def main():
    # Check for GPU
    print("GPU available:", torch.cuda.is_available())

    # Load dataset
    dataset = load_dataset("conll2003", trust_remote_code=True)
    label_list = dataset["train"].features["ner_tags"].feature.names 
    print(label_list)

    # class distribution check
    all_labels = [tag for sample in dataset['train']['ner_tags'] for tag in sample]
    print(Counter(all_labels))


    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}
    print(label_to_id, 'label_to_id')
    print(id_to_label, 'id_to_label')
    label_all_tokens = True 

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            padding="max_length",        # or "longest" if dynamically batching
            truncation=True,
            is_split_into_words=True,
            max_length=128,              # set max_length as needed
            return_tensors=None,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label_list[label[word_idx]]])  # Convert int to str label
                else:
                    label_ids.append(
                        label_to_id[label_list[label[word_idx]]] if label_all_tokens else -100
                    )
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = dataset.map(tokenize_and_align_labels, batched=True)

    # Select smaller subsets for quick iteration
    small_train = dataset["train"].shuffle(seed=42).select(range(1000))
    small_eval = dataset["test"].shuffle(seed=42).select(range(1000))

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True
    )


    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1", # or your preferred metric
        greater_is_better=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
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

    # Pick a sample from eval dataset
    sample_tokens = dataset["test"][0]["tokens"]
    sample_labels = dataset["test"][0]["ner_tags"]
    print("Tokens:", sample_tokens)
    print("LABELS:", sample_labels)

    # Tokenize the sample
    inputs = tokenizer(sample_tokens, is_split_into_words=True, return_tensors="pt")

    # Move inputs to model device (CPU or GPU)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    # Predictions
    pred_ids = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    pred_labels = [id_to_label[p] for p in pred_ids]
    true_labels = sample_labels

    # Compare token by token
    print("Tokens:", sample_tokens)
    print("True Labels:", true_labels)
    print("Predicted Labels:", pred_labels)

    # Load pipeline for inference
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Test with an example
    sample = "Sasha is a dog"
    result = ner_pipeline(sample)
    print("Classification result:", result)

    # Evaluate
    results = trainer.evaluate(dataset["test"])
    print("Evaluation Results:", results)



if __name__ == "__main__":
    main()
