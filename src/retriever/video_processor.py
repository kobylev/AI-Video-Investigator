import cv2
import os
from PIL import Image
from typing import Iterator, Tuple
import logging

class VideoProcessor:
    """
    Handles frame extraction from video files.
    """
    
    def __init__(self, sample_rate_fps: float = 1.0):
        """
        Initialize VideoProcessor.
        :param sample_rate_fps: How many frames to extract per second of video.
        """
        self.sample_rate_fps = sample_rate_fps
        self.logger = logging.getLogger(__name__)

    def extract_frames(self, video_path: str) -> Iterator[Tuple[Image.Image, float]]:
        """
        Generator that yields extracted frames and their timestamps.
        
        :param video_path: Path to the video file.
        :yield: A tuple of (PIL Image, timestamp_in_seconds).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps == 0:
                raise ValueError("Could not determine video FPS.")

            # Calculate frame interval based on desired sample rate
            frame_interval = int(video_fps / self.sample_rate_fps)
            if frame_interval < 1:
                frame_interval = 1

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    
                    timestamp = frame_count / video_fps
                    yield img, timestamp

                frame_count += 1
        finally:
            cap.release()

if __name__ == "__main__":
    # Quick test if run directly
    # vp = VideoProcessor(fps=0.5)
    # for img, ts in vp.extract_frames("path/to/video.mp4"):
    #     print(f"Extracted frame at {ts:.2f}s")
    pass
