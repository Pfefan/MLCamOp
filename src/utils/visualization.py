def visualize_frame(frame, predictions, title="Frame Visualization"):
    import cv2
    import numpy as np

    # Create a copy of the frame to draw on
    output_frame = frame.copy()

    # Draw predictions on the frame
    for prediction in predictions:
        label, confidence, bbox = prediction
        x, y, w, h = bbox
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow(title, output_frame)
    cv2.waitKey(1)  # Wait for a short period to display the frame

def visualize_video(video_path, model, frame_interval=30):
    import cv2

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            predictions = model.predict(frame)  # Assuming model has a predict method
            visualize_frame(frame, predictions)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()