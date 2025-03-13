학습: `python main.py train --train_data data/test/train.jsonl --output_dir ./trained_model`

평가: `python main.py evaluate --model_path ./trained_model --data_path data/test/validation.jsonl`
