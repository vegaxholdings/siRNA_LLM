import os
import argparse
import torch
from model import SiRNAModel
from data import SiRNADataset, calculate_accuracy
from utils import set_seed, generate_predictions
import json

def evaluate_model(model_path, data_path, bin_size=10):
    """Evaluate the fine-tuned model on validation data"""
    print(f"Loading model from {model_path}...")
    # Initialize model
    sirna_model = SiRNAModel()
    model, tokenizer = sirna_model.load_trained_model(model_path)
    
    print(f"Loading validation data from {data_path}...")
    # Load validation dataset
    val_dataset = SiRNADataset(data_path, tokenizer=None)
    
    print("Generating predictions...")
    # Generate predictions
    predictions = generate_predictions(model, tokenizer, val_dataset)
    
    # Calculate accuracy
    true_labels = val_dataset.get_labels()
    accuracy = calculate_accuracy(predictions, true_labels, bin_size=bin_size)
    
    print(f"Accuracy (bin size {bin_size}): {accuracy:.4f}")
    
    # Save detailed results for analysis
    results = {
        "accuracy": accuracy,
        "predictions": [float(p) if p is not None else None for p in predictions],
        "true_labels": [float(l) for l in true_labels],
        "binned_predictions": [int(p // bin_size) if p is not None else None for p in predictions],
        "binned_true_labels": [int(l // bin_size) for l in true_labels]
    }
    
    results_path = os.path.join(os.path.dirname(model_path), "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to {results_path}")
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the siRNA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--data_path", type=str, default="data/test/validation.jsonl", 
                        help="Path to the validation data")
    parser.add_argument("--bin_size", type=int, default=10, 
                        help="Size of bins for categorizing predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    evaluate_model(args.model_path, args.data_path, args.bin_size)