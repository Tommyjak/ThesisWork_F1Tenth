import cv2
import os
import sys

def extract_frames(video_file):
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: cannot open video file '{video_file}'")
        sys.exit(1)

    frame_rate = 10  # Desired FPS to extract
    native_fps = round(cap.get(cv2.CAP_PROP_FPS))  # Round to nearest int (e.g. 29.97 -> 30)
    frame_interval = max(1, native_fps // frame_rate)  # Integer division, safe floor

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_directory = f"../../datasets/{video_name}/images/"
    os.makedirs(output_directory, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_file = f"{output_directory}/frame_{frame_count:06d}.jpg"
            cv2.imwrite(output_file, frame)
            saved_count += 1
            print(f"Saved frame {frame_count} -> {output_file}")

        frame_count += 1

    cap.release()
    print(f"\nDone. {saved_count} frames saved from {frame_count} total (native: {native_fps} FPS, extracted: ~{frame_rate} FPS)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python vid2img.py <path_to_video_file>")
        sys.exit(1)

    extract_frames(sys.argv[1])