import cv2
import numpy as np

class LaplacianBlending:
    def __init__(self, left, right, blend_mask, levels):
        assert left.shape == right.shape == blend_mask.shape
        self.left = left.astype(np.float32)
        self.right = right.astype(np.float32)
        self.blend_mask = blend_mask.astype(np.float32)
        self.levels = levels
        self.left_lap_pyr = []
        self.right_lap_pyr = []
        self.left_smallest_level = None
        self.right_smallest_level = None
        self.result_lap_pyr = []
        self.result_smallest_level = None
        self.mask_gaussian_pyr = []
        self.build_pyramids()
        self.blend_lap_pyrs()

    def build_pyramids(self):
        self.build_laplacian_pyramid(self.left, self.left_lap_pyr, self.left_smallest_level)
        self.build_laplacian_pyramid(self.right, self.right_lap_pyr, self.right_smallest_level)
        self.build_gaussian_pyramid()

    def build_gaussian_pyramid(self):
        assert len(self.left_lap_pyr) > 0

        self.mask_gaussian_pyr.clear()
        current_img = cv2.cvtColor(self.blend_mask, cv2.COLOR_GRAY2BGR)
        self.mask_gaussian_pyr.append(current_img)  # highest level

        current_img = self.blend_mask
        for l in range(1, self.levels + 1):
            _down = cv2.pyrDown(current_img) if len(self.left_lap_pyr) > l else cv2.pyrDown(current_img, self.left_smallest_level.shape[:2])
            down = cv2.cvtColor(_down, cv2.COLOR_GRAY2BGR)
            self.mask_gaussian_pyr.append(down)
            current_img = _down

    def build_laplacian_pyramid(self, img, lap_pyr, smallest_level):
        lap_pyr.clear()
        current_img = img
        for l in range(self.levels):
            down = cv2.pyrDown(current_img)
            up = cv2.pyrUp(down, current_img.shape[:2])
            lap = current_img - up
            lap_pyr.append(lap)
            current_img = down
        current_img.copyTo(smallest_level)

    def reconstruct_img_from_lap_pyramid(self):
        current_img = self.result_smallest_level
        for l in range(self.levels - 1, -1, -1):
            up = cv2.pyrUp(current_img, dstsize=self.result_lap_pyr[l].shape[:2])
            current_img = up + self.result_lap_pyr[l]
        return current_img

    def blend_lap_pyrs(self):
        self.result_smallest_level = self.left_smallest_level * self.mask_gaussian_pyr[-1] + \
                                     self.right_smallest_level * (1.0 - self.mask_gaussian_pyr[-1])
        for l in range(self.levels):
            A = self.left_lap_pyr[l] * self.mask_gaussian_pyr[l]
            anti_mask = 1.0 - self.mask_gaussian_pyr[l]
            B = self.right_lap_pyr[l] * anti_mask
            blended_level = A + B
            self.result_lap_pyr.append(blended_level)

    def blend(self):
        return self.reconstruct_img_from_lap_pyramid()
