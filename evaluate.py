import os
import json
import argparse

from model import SiRNAModel
from data import SiRNADataset, calculate_accuracy
from utils import set_seed, generate_predictions

def evaluate_model(model_path, data_path, max_diff=10):
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
    accuracy = calculate_accuracy(predictions, true_labels, max_diff=max_diff)
    
    print(f"Accuracy (max difference {max_diff}): {accuracy:.4f}")
    
    # Save detailed results for analysis
    results = {
        "accuracy": accuracy,
        "predictions": [float(p) if p is not None else None for p in predictions],
        "true_labels": [float(l) for l in true_labels],
        "differences": [abs(float(p) - float(l)) if p is not None else None for p, l in zip(predictions, true_labels)],
        "correct_predictions": [abs(float(p) - float(l)) < max_diff if p is not None else False for p, l in zip(predictions, true_labels)]
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
    parser.add_argument("--max_diff", type=float, default=10, 
                        help="Maximum difference threshold for correct predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    evaluate_model(args.model_path, args.data_path, args.max_diff)