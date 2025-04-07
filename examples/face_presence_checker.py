#!/usr/bin/env python3
import os
import cv2
import numpy as np
import logging
import shutil
from mtcnn.mtcnn import MTCNN
from PIL import Image
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FacePresenceChecker:
    def __init__(self, min_face_confidence=0.90):
        """
        Initializes the FacePresenceChecker with an MTCNN detector.

        Parameters:
          min_face_confidence (float): Minimum confidence to count a detection as a face.
        """
        self.detector = MTCNN()
        self.min_face_confidence = min_face_confidence
        logger.debug("Initialized MTCNN face detector.")

    def check_face(self, image_path):
        """
        Checks if an image contains at least one face with sufficient confidence.

        Parameters:
          image_path (str): Path to the image file.

        Returns:
          bool: True if a face is detected with confidence >= min_face_confidence, False otherwise.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            img_np = np.array(img)
            detections = self.detector.detect_faces(img_np)
            if detections:
                for detection in detections:
                    conf = detection.get("confidence", 0)
                    if conf >= self.min_face_confidence:
                        logger.debug(f"Face detected in {os.path.basename(image_path)} with confidence {conf:.2f}")
                        return True
            logger.debug(f"No valid face detected in {os.path.basename(image_path)}")
            return False
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False

    def check_faces_in_directory(self, directory):
        """
        Evaluates all images in the directory and returns a list of filenames that do not contain a face.

        Parameters:
          directory (str): Path to the directory with images.

        Returns:
          list: Filenames (with full paths) of images with no detected face.
        """
        missing_face_files = []
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        logger.info(f"Evaluating images in directory: {directory}")
        for filename in os.listdir(directory):
            if filename.lower().endswith(valid_extensions):
                file_path = os.path.join(directory, filename)
                if not self.check_face(file_path):
                    missing_face_files.append(file_path)
                    logger.debug(f"No face found in {filename}")
        return missing_face_files

    def move_images_without_faces(self, source_dir, dest_dir):
        """
        Moves images from source_dir that do not contain a face to dest_dir.

        Parameters:
          source_dir (str): Directory with images to check.
          dest_dir (str): Directory to move images without detected faces.
        """
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            logger.debug(f"Created destination directory: {dest_dir}")

        missing_face_files = self.check_faces_in_directory(source_dir)
        for file_path in missing_face_files:
            basename = os.path.basename(file_path)
            dest_path = os.path.join(dest_dir, basename)
            try:
                shutil.move(file_path, dest_path)
                logger.info(f"Moved {basename} to {dest_dir}")
            except Exception as e:
                logger.error(f"Error moving {basename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and move images without faces.")
    parser.add_argument("--source", type=str, required=True, help="Source directory containing images.")
    parser.add_argument("--dest", type=str, default="no_face_images", help="Destination directory for images with no faces.")
    parser.add_argument("--min_conf", type=float, default=0.90, help="Minimum face detection confidence.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")
    else:
        logger.setLevel(logging.INFO)

    checker = FacePresenceChecker(min_face_confidence=args.min_conf)
    logger.info("Starting to move images without faces.")
    checker.move_images_without_faces(args.source, args.dest)
    logger.info("Process complete.")
