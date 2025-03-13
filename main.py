import argparse
import os
from train import train_model
from evaluate import evaluate_model
from chat import chat_with_model
from utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="ChatsiRNA - siRNA 효율성 예측 모델")
    subparsers = parser.add_subparsers(dest="command", help="실행할 명령어")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="모델 학습")
    train_parser.add_argument("--train_data", type=str, default="data/test/train.jsonl", 
                             help="학습 데이터 경로")
    train_parser.add_argument("--output_dir", type=str, default="./trained_model", 
                             help="모델 저장 디렉토리")
    train_parser.add_argument("--batch_size", type=int, default=4, 
                             help="배치 크기")
    train_parser.add_argument("--grad_accum_steps", type=int, default=8, 
                             help="그래디언트 누적 단계")
    train_parser.add_argument("--num_epochs", type=int, default=3, 
                             help="학습 에폭 수")
    train_parser.add_argument("--lr", type=float, default=2e-4, 
                             help="학습률")
    train_parser.add_argument("--seed", type=int, default=42, 
                             help="랜덤 시드")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="모델 평가")
    eval_parser.add_argument("--model_path", type=str, required=True, 
                            help="학습된 모델 경로")
    eval_parser.add_argument("--data_path", type=str, default="data/test/validation.jsonl", 
                            help="평가 데이터 경로")
    eval_parser.add_argument("--max_diff", type=float, default=10, 
                            help="올바른 예측으로 간주하는 최대 차이값")
    eval_parser.add_argument("--seed", type=int, default=42, 
                            help="랜덤 시드")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="모델과 대화")
    chat_parser.add_argument("--model_path", type=str, required=True, 
                           help="학습된 모델 경로")
    chat_parser.add_argument("--seed", type=int, default=42, 
                           help="랜덤 시드")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    if args.command == "train":
        print("학습을 시작합니다...")
        train_model(
            args.train_data,
            args.output_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            num_epochs=args.num_epochs,
            learning_rate=args.lr
        )
    elif args.command == "evaluate":
        print("평가를 시작합니다...")
        evaluate_model(
            args.model_path,
            args.data_path,
            max_diff=args.max_diff
        )
    elif args.command == "chat":
        print("모델과 대화를 시작합니다...")
        chat_with_model(args.model_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()