import cv2 as cv
from .strategyPreProcessPipelineSteps.StepGrayscale import StepGrayscale
from .strategyPreProcessPipelineSteps.StepDenoise import StepDenoise
from .strategyPreProcessPipelineSteps.StepShadow import StepShadow
from .strategyPreProcessPipelineSteps.StepDeskew import StepDeskew
from .strategyPreProcessPipelineSteps.StepContrast import StepContrast
from .strategyPreProcessPipelineSteps.StepBinarize import StepBinarize


class StrategyPreProcessPipeline:
    def __init__(self, image, log: bool = False):
        self.strategy = None
        self.image = image
        self.log = log

    def __enter__(self):
        print(f"## [Pipeline] [ContextPreprocessor] [{self.__class__.__name__}] started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            print(f"## [Pipeline][ContextPreprocessor] [{self.__class__.__name__}] completed")

    def execute(self):
        with StepGrayscale(image=self.image, log=self.log) as step_grayscale:
            grayscaled_image = step_grayscale.apply()
        with StepShadow(image=grayscaled_image, log=self.log) as step_shadow:
            deshadowed_image = step_shadow.apply()
        with StepDenoise(image=deshadowed_image, log=self.log) as step_denoise:
            denoised_image = step_denoise.apply()
        with StepDeskew(image=denoised_image, log=self.log) as step_deskew:
            deskewed_image = step_deskew.apply()
        with StepContrast(image=deskewed_image, log=self.log) as step_contrast:
            contrasted_image = step_contrast.apply()
        with StepBinarize(image=contrasted_image, log=self.log) as step_binarize:
            binarized_image = step_binarize.apply()
        image_rgb = cv.cvtColor(binarized_image, cv.COLOR_BGR2RGB)
        # return all images for debugging purpose
        return [grayscaled_image, deshadowed_image, denoised_image, deskewed_image, contrasted_image, binarized_image, image_rgb]
