import argparse
import os
import sys
import logging
from src.pipeline.inference import run_inference
from src.utils.logger import setup_logger

def evaluate_model(model_path, test_data_path, output_path):
    # Set up logging
    logger = logging.getLogger(__name__)
    setup_logger()

    logger.info("Starting evaluation of the model.")
    
    # Run inference on the test data
    results = run_inference(model_path, test_data_path)

    # Save the evaluation results
    with open(output_path, 'w') as f:
        f.write(str(results))
    
    logger.info(f"Evaluation results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model on test data.")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test data.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the evaluation results.')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error(f"Model path {args.model} does not exist.")
        sys.exit(1)

    if not os.path.exists(args.test_data):
        logger.error(f"Test data path {args.test_data} does not exist.")
        sys.exit(1)

    evaluate_model(args.model, args.test_data, args.output)

if __name__ == "__main__":
    main()