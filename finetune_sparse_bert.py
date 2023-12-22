# evaluate_bert.py

from transformers import set_seed, AutoConfig, AutoTokenizer, TrainingArguments, Trainer, default_data_collator, TrainerCallback, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate
import numpy as np
import argparse
import os

from sparsify import get_sliding_window, dense_sparse_convert, get_blocks, compute_window_size
from BlockSparseFromLinear import sparse_convert

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def evaluation(model=None, dataset="glue", task_name=None, tokenizer=None, revision="main", output_dir=None, 
               batch_size=16, learning_rate=5e-5, num_epochs=3, logging_steps=50, run_name=None, seed=42, 
               verbose=False, do_predict=True):
    ## Set seed
    set_seed(seed)

    ## Load preprocessed data
    if verbose: print(f"Loading the '{task_name}' split of the '{dataset}' dataset...")
    data = load_dataset(dataset, task_name)

    if task_name == "mnli" and do_predict is True:
        data["ax"] = load_dataset("glue", "ax")["test"]

    # Get number of classification labels
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = data["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    ## Load 
    model_name = "bert-base-uncased" if model is None else model
    config = AutoConfig.from_pretrained(model_name, revision=revision, num_labels=num_labels)

    ## Load tokenizer
    if tokenizer is None: tokenizer = model_name
    if verbose: print(f"Loading {tokenizer} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=revision)

    ## Load model
    if verbose: print(f"Loading {model_name} model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, config=config, ignore_mismatched_sizes=True
    )

    ## Sparsify model
    if verbose: print(f"Sparsifying {model_name} model...")
    model = sparse_convert(dense_sparse_convert(model, get_sliding_window, window_size=64))

    ## Define training arguments
    if verbose: print(f"Defining training arguments...")
    if output_dir is None: output_dir = model_name.split("/")[-1]
    if run_name is None: run_name = "eval-" + dataset
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="no",
        push_to_hub=False,
        logging_steps=logging_steps,
        run_name=run_name
    )

    ## Preprocess data
    if verbose: print(f"Preprocessing the dataset for the '{task_name}' task...")
    sentence1_key, sentence2_key = task_to_keys[task_name]
    padding = "max_length"

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=128, truncation=True)   
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        data = data.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )

    # Define datasets
    train_dataset = data["train"]
    eval_dataset = data["validation_matched" if task_name == "mnli" else "validation"]

    ## Load evaluation metric
    if verbose: print(f"Load the evaluation metric for the '{task_name}' task...")
    metric = evaluate.load("glue", task_name)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    ## Create data collator
    data_collator = default_data_collator

    # Freezing of base model parameters is not needed as in pretraining for the reduction head
    # Evaluation is on whether fine-tuning works well on the model

    # Initialize our Trainer
    if verbose: print(f"Initialize Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    ## Add evaluation for 0th epoch
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                control.should_evaluate = True

    trainer.add_callback(EvaluateFirstStepCallback())

    ## Train the model
    if verbose: print(f"Train the model...")
    train_result = trainer.train()

    ## Save the model
    if verbose: print(f"Save the model...")
    metrics = train_result.metrics
    trainer.save_model(output_dir=output_dir)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    if verbose: print(f"Evaluate the model on the validation set...")
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [task_name]
    eval_datasets = [eval_dataset]
    if task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(data["validation_mismatched"])
        combined = {}

    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined if "mnli" in task else metrics)

    if do_predict:
        print("Predict the test set labels...")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [task_name]
        predict_dataset = data["test_matched" if task_name == "mnli" else "test"]
        predict_datasets = [predict_dataset]
        if task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(data["test_mismatched"])

            # Also add AX task
            tasks.append("ax")
            predict_datasets.append(data["ax"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_name, "tasks": "text-classification"}
    kwargs["language"] = "en"
    kwargs["dataset_tags"] = "glue"
    kwargs["dataset_args"] = task_name
    kwargs["dataset"] = f"GLUE {task_name.upper()}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the provided model on the given preprocessed dataset for MLM and NSP.")
    parser.add_argument(
        '--model',
        help=("The model to pretrain. Default is 'cayjobla/bert-base-uncased-reduced'."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--revision',
        help=("The revision of the model to use. Default is 'main'"),
        type=str,
        default="main"
    )
    parser.add_argument(
        '--dataset',
        help=("The dataset to train the model on. Defaults to the glue dataset."),
        type=str,
        default="glue"
    )
    parser.add_argument(
        '--task',
        help=("The glue task to train/evaluate on. Required for training."),
        dest="task_name",
        type=str,
        default=None
    )
    parser.add_argument(
        '--tokenizer',
        help=("The tokenizer to use during training. Default is to use the model's tokenizer."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--batch_size',
        help=("The batch size to use during training. Default is 16."),
        type=int,
        default=16
    )
    parser.add_argument(
        '--learning_rate',
        help=("The learning rate to use during training. Default is 2e-4."),
        type=float,
        default=2e-4
    )
    parser.add_argument(
        '--num_epochs',
        help=("The number of epochs to train for. Default is 1."),
        type=int,
        default=1
    )
    parser.add_argument(
        '--logging_steps',
        help=("The number of steps between logging during training. Default is 50."),
        type=int,
        default=50
    )
    parser.add_argument(
        '--output_dir',
        help=("The output directory to save the trained model to. If None, the model name is used."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--run_name',
        help=("The name of the WandB run. Default is 'mlm-nsp-{dataset}'."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--seed',
        help=("The seed to use for randomness in preprocessing. Default is 42."),
        type=int,
        default=42
    )
    parser.add_argument(
        '--verbose',
        '-v',
        help=("Whether to print preprocessing progress information. Default is False."),
        default=False,
        action="store_true"
    )  
    parser.add_argument(
        '--do_predict',
        help=("Whether to predict on the test set after training. Default is True."),
        default=True,
        action="store_true"
    )

    kwargs = parser.parse_args()
    evaluation(**vars(kwargs))