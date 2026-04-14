import sys
from ultralytics import YOLO

def train_model(run_name):
    print("Model loading...")
    model = YOLO("yolo26n.pt")
    print("Model loaded successfully.")

    print("Starting training...")
    results = model.train(
        data = "tmp/tmp_config.yaml",

        epochs = 150,
        patience = 20,
        imgsz = 640,
        batch = -1,

        device = 0, # Use GPU if available, otherwise CPU

        project = 'f1tenth_experiments',
        name = 'run_yolo26_complete',
        exist_ok = True, # Overwrite existing experiment with the same name
    )
    print("Training completed.")

if __name__ == "__main__":
    run_name = sys.argv[1].lower() if len(sys.argv) > 1 else None

    if run_name is None:
        print("No experiment name provided. Please specify an experiment name as a command-line argument.")
        sys.exit(1)

    train_model(run_name)