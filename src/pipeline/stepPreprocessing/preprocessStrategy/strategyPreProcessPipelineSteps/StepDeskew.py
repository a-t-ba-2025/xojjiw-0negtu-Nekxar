from .AbstractPreprocessPipelineStep import AbstractPreprocessPipelineStep
import cv2 as cv
import numpy as np


class StepDeskew(AbstractPreprocessPipelineStep):
    #Deskewing: Detects and corrects small rotation. Uses Hough Line Detection to find angle of text and rotates image.
    def apply(self):
        binary = self.binarize_for_deskew(self.image) # simple binary version of image
        edges = cv.Canny(binary, 50, 150, apertureSize=3) # Find edges in image
        angle = self.estimate_skew_angle(edges) # find the angle of the text

        if angle is None:
            if self.log:
                print("###  No  skew angle found -> skipping deskewing.")
            return self.image

        # If the angle is too large, probably an error → skip rotation
        if abs(angle) > 30:
            if self.log:
                print(f"### [Warning]: high angle ({angle:.2f}°) -> skipping rotation.")
            return self.image

        rotated_image = self.rotate_image(self.image, angle)

        return rotated_image

    def binarize_for_deskew(self, image):
        # Converts image to binary and inverts if background is white.
        _, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # If the image is bright (white background), invert
        if np.mean(binary) > 127:
            binary = 255 - binary
            if self.log:
                print("### Inverted binary image (white background detected).")
        return binary

    def estimate_skew_angle(self, edges):
        # Estimates skew angle using Hough line detection.
        # Detect straight lines using the Hough Transform
        lines = cv.HoughLines(edges, 1, np.pi / 180.0, 200)
        if lines is None:
            return None

        # Get angles from detected lines. Each line = rho and theta, theta:angle in radians of the normal to the line. uff.
        angles = [
            (theta * 180 / np.pi) - 90 # Convert theta to degrees and normalize around 0 degrees
            for rho, theta in lines[:, 0]
            if -20 < ((theta * 180 / np.pi) - 90) < 20 # only accept angles in range 20 deg
        ]

        if not angles:
            return None

        median_angle = np.median(angles)

        return median_angle

    def rotate_image(self, image, angle):
        #Rotates the image by angle.
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # m: transformation_martix
        m = cv.getRotationMatrix2D(center, angle, 1.0)

        # INTER_CUBIC: bicubic interpolation (4x4 pixel)
        rotated_image = cv.warpAffine(image, m, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        if self.log:
            print(f"### Rotation performed by: {angle:.2f}°")
        return rotated_image
