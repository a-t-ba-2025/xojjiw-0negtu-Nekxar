from .AbstractPreprocessPipelineStep import AbstractPreprocessPipelineStep
import cv2 as cv
import numpy as np


class StepShadow(AbstractPreprocessPipelineStep):
    # Shadow Detection Thresholds
    SHADOW_IGNORE_THRESHOLD = 5
    SHADOW_LIGHT_THRESHOLD = 15
    SHADOW_MEDIUM_THRESHOLD = 25

    def apply(self):
        shadow_strength = self.estimate_shadow_strength()

        # Case: Shadow is negligible –> no removal needed
        if shadow_strength < self.SHADOW_IGNORE_THRESHOLD:
            if self.log:
                print("### Shadow negligible -> skipping shadow removal")
            return self.image

        # Get blur size Kernel based on shadow strength
        blur_size, level = self.get_blur_size_and_level(shadow_strength)

        if self.log:
            print(f"### Applying Gaussian Blur for {level} shadow with kernel size: {blur_size}")

        background = cv.GaussianBlur(self.image, blur_size, 0)
        background = np.where(background == 0, 1, background)  # Prevent divide-by-zero

        normalized = (self.image.astype(np.float32) / background.astype(np.float32)) * 255.0
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return normalized

    def estimate_shadow_strength(self):
        # Estimate shadow strength by measuring standard deviation (std) in a very blurred image.
        heavily_blurred = cv.GaussianBlur(self.image, (101, 101), 0)
        shadow_strength = np.std(heavily_blurred)

        if self.log:
            print(f"### Estimated shadow strength (std dev): {shadow_strength:.2f}")

        return shadow_strength

    def get_blur_size_and_level(self, shadow_strength):
        # Determine blur kernel size based on shadow strength.
        if shadow_strength < self.SHADOW_LIGHT_THRESHOLD:
            return (31, 31), "LIGHT"
        elif shadow_strength < self.SHADOW_MEDIUM_THRESHOLD:
            return (51, 51), "MEDIUM"
        else:
            return (101, 101), "HEAVY"
