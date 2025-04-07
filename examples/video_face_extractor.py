#!/usr/bin/env python3
import os
import cv2
import numpy as np
import logging
from mtcnn.mtcnn import MTCNN
from PIL import Image
from datetime import datetime
import argparse

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class VideoFaceExtractor:
    def __init__(self, output_size=(128, 128), desired_left_eye=(0.35, 0.35)):
        """
        Initializes the VideoFaceExtractor.

        Parameters:
          output_size (tuple): The desired output face size, default is (128, 128).
          desired_left_eye (tuple): The desired position (as a fraction of width,height)
                                    of the left eye in the output image.
        """
        self.output_size = output_size
        self.desired_left_eye = desired_left_eye
        self.detector = MTCNN()
        logger.debug("Initialized MTCNN face detector.")

    def align_face(self, image, left_eye, right_eye):
        """
        Aligns a face in the image based on the left and right eye positions.

        Parameters:
          image (np.array): The input image in RGB.
          left_eye (tuple): (x, y) coordinates for the left eye.
          right_eye (tuple): (x, y) coordinates for the right eye.

        Returns:
          np.array: The aligned face image of size self.output_size.
        """
        logger.debug(f"Aligning face: left_eye={left_eye}, right_eye={right_eye}")
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        logger.debug(f"Angle between eyes: {angle:.2f} degrees")

        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        dist = np.sqrt(dX ** 2 + dY ** 2)
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.output_size[0]
        scale = desired_dist / dist
        logger.debug(f"Computed scale: {scale:.4f}")

        eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                       (left_eye[1] + right_eye[1]) / 2)
        logger.debug(f"Eyes center: {eyes_center}")

        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        tX = self.output_size[0] * 0.5
        tY = self.output_size[1] * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        logger.debug(f"Affine transform matrix:\n{M}")

        aligned = cv2.warpAffine(image, M, self.output_size, flags=cv2.INTER_CUBIC)
        return aligned

    def process_video(self, video_path, output_dir, start_frame=0, max_frames=None):
        """
        Processes a video file frame-by-frame, extracts aligned faces,
        and saves each face as a 128x128 image in the output directory.

        Parameters:
          video_path (str): Path to the video file.
          output_dir (str): Directory to save extracted face images.
          start_frame (int): Frame number to start processing.
          max_frames (int or None): Maximum number of frames to process.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.debug(f"Created output directory: {output_dir}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Unable to open video file: {video_path}")
            return

        frame_num = 0
        saved_faces = 0
        logger.info(f"Starting video processing for {video_path}")
        start_time = datetime.now()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.debug("No more frames to read; exiting loop.")
                break

            frame_num += 1
            if frame_num < start_frame:
                continue
            if max_frames is not None and frame_num - start_frame >= max_frames:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb_frame)
            logger.debug(f"Frame {frame_num}: Detected {len(detections)} faces.")
            if detections:
                for i, detection in enumerate(detections):
                    box = detection.get("box")
                    keypoints = detection.get("keypoints")
                    if box is None or keypoints is None:
                        continue

                    left_eye = keypoints.get("left_eye")
                    right_eye = keypoints.get("right_eye")
                    if left_eye is None or right_eye is None:
                        continue

                    try:
                        aligned_face = self.align_face(rgb_frame, left_eye, right_eye)
                        face_image = Image.fromarray(aligned_face)
                        filename = f"frame{frame_num:06d}_face{i}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        face_image.save(filepath)
                        saved_faces += 1
                        logger.debug(f"Saved face {i} from frame {frame_num} to {filepath}")
                    except Exception as e:
                        logger.error(f"Error aligning face in frame {frame_num}: {e}")

            if frame_num % 50 == 0:
                logger.info(f"Processed frame {frame_num}; total faces saved so far: {saved_faces}")

        cap.release()
        end_time = datetime.now()
        logger.info(f"Finished processing video. Total frames: {frame_num}, Faces saved: {saved_faces}")
        logger.info(f"Processing time: {end_time - start_time}")

# ----------------------------
# __main__ entry point
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract and align faces from video frames.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--output", type=str, default="extracted_faces", help="Directory to save extracted face images.")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame number to start processing.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")
    else:
        logger.setLevel(logging.INFO)

    logger.info("Starting face extraction process.")
    extractor = VideoFaceExtractor(output_size=(128, 128))
    extractor.process_video(args.video, args.output, start_frame=args.start_frame, max_frames=args.max_frames)

if __name__ == "__main__":
    main()
