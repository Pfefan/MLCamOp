def read_video(file_path):
    import cv2
    video = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames

def write_video(frames, output_path, fps=30):
    import cv2
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        video.write(frame)
    video.release()