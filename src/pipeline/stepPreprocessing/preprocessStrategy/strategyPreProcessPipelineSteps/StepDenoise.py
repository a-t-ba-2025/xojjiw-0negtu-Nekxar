from .AbstractPreprocessPipelineStep import AbstractPreprocessPipelineStep
import cv2 as cv
import numpy as np


class StepDenoise(AbstractPreprocessPipelineStep):
    # Thresholds
    VARIANCE_LOW = 20  # below this: (almost) noise-free
    VARIANCE_MID = 100  # Below this: mild noise
    EDGE_DENSITY_THRESHOLD = 0.05  # Above: contains structural details (text, etc.)
    EDGE_DENSITY_HIGH = 0.1  # highly structured text-heavy

    def apply(self):
        # Variance
        variance = self.estimate_variance(self.image)
        # Edge-Density
        edge_density = self.estimate_edge_density(self.image)

        # Case 1: (almost) noise-free -> skip
        if variance < self.VARIANCE_LOW:
            return self._skip_denoising()
        # Case 2: mild noise -> apply medianBlur
        elif variance < self.VARIANCE_MID:
            return self._apply_median_filter()
        # Case 3: text or strong structure -> apply bilateral-filter
        elif edge_density > self.EDGE_DENSITY_THRESHOLD:
            return self.apply_bilateral_filter(variance, edge_density)
        # Case 4: noisy images -> apply fastNlMeansDenoising
        else:
            return self.apply_nl_means_filter(variance)

    def _skip_denoising(self):
        if self.log:
            print("### (Almost) Noise free -> Skip denoising Step")
        return self.image

    def _apply_median_filter(self):
        if self.log:
            print("### Mild noise -> Apply Median filter")
        return cv.medianBlur(self.image, 3)

    def apply_bilateral_filter(self, variance, edge_density):
        sigma_color = min(150, 50 + 0.5 * variance)
        sigma_space = 75 if edge_density > self.EDGE_DENSITY_HIGH else 100

        if self.log:
            print(f"### Lots of text in image -> Apply Bilateral filter "
                  f"(sigmaColor={sigma_color:.1f}, sigmaSpace={sigma_space})")
        return cv.bilateralFilter(self.image, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    def apply_nl_means_filter(self, variance):
        h = 5 + 0.1 * np.sqrt(variance)
        if self.log:
            print(f"### very noisy -> Apply NL-Means Denoising (h={h:.2f})")
        return cv.fastNlMeansDenoising(self.image, None, h, 7, 21) # greyscale denoising

    def estimate_variance(self, image):
        # Estimate noise level (pixel intensity variance.)
        variance = np.var(image)
        if self.log:
            print(f"### Noise Estimation -> Variance: {variance:.2f}")
        return variance

    def estimate_edge_density(self, image):
        # Estimate edge density. (Canny edge detection)
        edges = cv.Canny(image, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        if self.log:
            print(f"### Noise Estimation -> Edge Density: {edge_density:.4f}")
        return edge_density
