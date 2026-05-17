from ultralytics import YOLO

# Load your trained model
model = YOLO('models/best.pt')

# Run validation
metrics = model.val()

# Print the key metrics
print(f"mAP50 (mAP at IoU=0.5): {metrics.box.map50:.4f}")
print(f"mAP50-95 (Average across IoU thresholds): {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")