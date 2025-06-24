from .AbstractPreprocessPipelineStep import AbstractPreprocessPipelineStep
import cv2 as cv


class StepGrayscale(AbstractPreprocessPipelineStep):
    def apply(self):
        if len(self.image.shape) == 3:
            if self.log:
                print("### Converting to greyscale.")
            gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        else:
            if self.log:
                print("### Image is already greyscale.")
            gray = self.image
        return gray
