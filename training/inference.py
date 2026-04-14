import cv2
from ultralytics import YOLO

# 1. Load your trained model
# Update this path if your best weights are in a different run folder
model_path_complete_run = '/home/tm/Documents/universita/tirocinio_hipert/research/utils/training/training_experiments/run_complete/weights/best.pt'
model_complete = YOLO(model_path_complete_run)

model_path_lowbg_run = '/home/tm/Documents/universita/tirocinio_hipert/research/utils/training/training_experiments/run_complete/weights/best.pt'
model_lowbg = YOLO(model_path_lowbg_run)

model_path_nobg_run = '/home/tm/Documents/universita/tirocinio_hipert/research/utils/training/training_experiments/run_complete/weights/best.pt'
model_nobg = YOLO(model_path_nobg_run)

# 2. Open the default webcam (0 is usually the built-in PC camera)
# If you have an external USB camera plugged in, you might need to change this to 1 or 2.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting live detection... Press 'q' in the window to stop.")

# 3. Loop through the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # 4. Run inference on the current frame
    # conf=0.5 means it will only show boxes if it's 50% sure or higher
    # stream=True is highly recommended for live video feeds to save memory
    results = model_lowbg(frame, stream=True, conf=0.5)

    # 5. Draw the bounding boxes on the frame
    for r in results:
        annotated_frame = r.plot()

    # 6. Display the live feed
    cv2.imshow("F1Tenth Live Detection", annotated_frame)

    # 7. Quit condition
    # Wait 1 millisecond for a key press. If 'q' is pressed, break the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()