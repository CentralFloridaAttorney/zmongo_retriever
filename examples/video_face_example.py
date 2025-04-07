#!/usr/bin/env python3
import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image


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
        self.detector = MTCNN()  # initialize face detector

    def align_face(self, image, left_eye, right_eye):
        """
        Aligns a face in the image based on the eye positions.

        Parameters:
            image (numpy.ndarray): The input image (as a numpy array, BGR or RGB).
            left_eye (tuple): (x, y) coordinates for the left eye.
            right_eye (tuple): (x, y) coordinates for the right eye.

        Returns:
            numpy.ndarray: The aligned face image cropped and resized to self.output_size.
        """
        # compute the angle between the eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # compute the desired distance between the eyes in the output image
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.output_size[0]
        scale = desired_dist / dist

        # compute the center point between the eyes
        eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                       (left_eye[1] + right_eye[1]) / 2)

        # get the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        tX = self.output_size[0] * 0.5
        tY = self.output_size[1] * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        # apply the affine transformation
        aligned = cv2.warpAffine(image, M, self.output_size, flags=cv2.INTER_CUBIC)
        return aligned

    def process_video(self, video_path, output_dir, start_frame=0, max_frames=None):
        """
        Processes a video file, extracts and aligns faces from each frame,
        and saves them as 128x128 images in the output directory.

        Parameters:
            video_path (str): Path to the video file.
            output_dir (str): Directory to save extracted face images.
            start_frame (int): Frame number to start processing.
            max_frames (int): Maximum number of frames to process (or None for all frames).
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        saved_faces = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            # skip frames until start_frame
            if frame_num < start_frame:
                continue
            # Optionally, break after processing max_frames
            if max_frames is not None and frame_num - start_frame >= max_frames:
                break

            # Convert frame to RGB (MTCNN expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb_frame)
            if detections:
                for i, detection in enumerate(detections):
                    # Extract bounding box and keypoints
                    box = detection.get("box")
                    keypoints = detection.get("keypoints")
                    if box is None or keypoints is None:
                        continue

                    # Get eye coordinates from keypoints
                    left_eye = keypoints.get("left_eye")
                    right_eye = keypoints.get("right_eye")
                    if left_eye is None or right_eye is None:
                        continue

                    # Align face
                    aligned_face = self.align_face(rgb_frame, left_eye, right_eye)
                    # Convert to PIL Image
                    face_image = Image.fromarray(aligned_face)
                    # Save face image with frame and face index in filename
                    filename = f"frame{frame_num:06d}_face{i}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    face_image.save(filepath)
                    saved_faces += 1
            if frame_num % 50 == 0:
                print(f"Processed frame {frame_num}, saved {saved_faces} faces so far.")
        cap.release()
        print(f"Finished processing video. Total frames processed: {frame_num}. Total faces saved: {saved_faces}")


# Example usage:
if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Extract and align faces from video frames.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--output", type=str, default="extracted_faces",
                        help="Directory to save extracted face images.")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame number to start processing.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process.")
    args = parser.parse_args()

    extractor = VideoFaceExtractor(output_size=(128, 128))
    extractor.process_video(args.video, args.output, start_frame=args.start_frame, max_frames=args.max_frames)
