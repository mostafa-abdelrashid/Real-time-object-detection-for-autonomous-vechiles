import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import cv2
import time
from pathlib import Path

class MLflowTracker:
    def __init__(self, experiment_name="Autonomous_Vehicle_YOLO11"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def log_training_run(self, model_path, metrics, params=None):
        """Log a training run to MLflow"""
        with mlflow.start_run():
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.pytorch.log_model(
                pytorch_model=YOLO(model_path).model,
                artifact_path="yolo11_model",
                registered_model_name="YOLO11_Autonomous_Vehicle"
            )
            
            # Log additional files
            if Path("runs/detect/train4/results.png").exists():
                mlflow.log_artifact("runs/detect/train4/results.png")
            if Path("runs/detect/train4/confusion_matrix.png").exists():
                mlflow.log_artifact("runs/detect/train4/confusion_matrix.png")
            
            print("✅ Training run logged to MLflow!")

    def log_inference_test(self, video_path, results):
        """Log inference test results"""
        with mlflow.start_run(run_name="Inference_Test"):
            mlflow.log_metrics({
                'processing_fps': results['processing_fps'],
                'total_detections': results['total_detections'],
                'avg_detections_per_frame': results['avg_detections_per_frame'],
                'real_time_factor': results['real_time_factor']
            })
            
            mlflow.log_param('test_video', video_path)
            mlflow.log_param('model', 'YOLO11s')
            
            print("✅ Inference test logged to MLflow!")

# Usage example
if __name__ == "__main__":
    tracker = MLflowTracker()
    
    # Log your training results
    training_metrics = {
        'mAP50': 0.898,
        'mAP50_95': 0.671, 
        'precision': 0.909,
        'recall': 0.833,
        'fps': 73.1
    }
    
    training_params = {
        'epochs': 50,
        'dataset': 'KITTI',
        'model_size': 'YOLO11s',
        'img_size': 640
    }
    
    tracker.log_training_run('best.pt', training_metrics, training_params)