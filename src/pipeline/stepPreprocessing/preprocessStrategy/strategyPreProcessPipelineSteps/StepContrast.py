from .AbstractPreprocessPipelineStep import AbstractPreprocessPipelineStep
import numpy as np
import cv2 as cv

from .StepDenoise import StepDenoise


class StepContrast(AbstractPreprocessPipelineStep):
    # StepContrast improves contrast using CLAHE (adaptive histogram equalization).
    def apply(self):
        # Measure contrast using standard deviation of pixel values
        contrast = np.std(self.image)
        if self.log:
            print(f"### Measured image contrast: {contrast:.2f}")

        # Choose CLAHE settings based on the contrast level
        clahe = self.select_clahe(contrast)
        if clahe is None:
            if self.log:
                print(f"### already good -> skipping enhancement.")
            return self.image

        # Apply CLAHE for improve contrast
        enhanced = clahe.apply(self.image)

        # Apply denoising again after contrast enhancement
        enhanced = self.run_post_denoising(enhanced)

        return enhanced

    def select_clahe(self, contrast):
        # Decide how strong contrast enhancement should be. If image already has good contrast, return None (->no contrast enhancement needed).
        if contrast < 30:
            if self.log:
                print(f"### Low contrast -> use strong CLAHE (clipLimit=2.0)")
            return cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        elif contrast < 50:
            if self.log:
                print(f"### Medium contrast -> use light CLAHE (clipLimit=1.0)")
            return cv.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
        else:
            return None  # No CLAHE needed

    def run_post_denoising(self, image):
        # Denoise image again after contrast enhancement.
        with StepDenoise(image=image, log=self.log) as step_denoise:
            return step_denoise.apply()
