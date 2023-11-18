import cv2
import numpy as np

class ImageTools:
    def __init__(self):
        self.minHessian = 400
        self.inlierThreshold = 15

    def find_gain_transform(self, src_color, dst_color):
        if src_color.shape != dst_color.shape:
            print("Image sizes do not match!")
            return None

        print("Finding transform for gain consistency...")

        if src_color is None or dst_color is None:
            print("No data to match gain with!")
            return None

        BGR_src, BGR_dst = self.find_matching_BGR(src_color, dst_color)

        transform_matrix = np.eye(4, dtype=np.float32)
        BGR_src_transpose = BGR_src.T
        transform_matrix[...] = np.linalg.inv(BGR_src_transpose @ BGR_src) @ (BGR_src_transpose @ BGR_dst)

        print("Transform found.")
        return transform_matrix

    def find_matching_BGR(self, img_1, img_2, use_keypoints):
        if img_1.size == 0 or img_2.size == 0:
            print("No data to match gain with!")
            return None, None

        if img_1.shape[:2] != img_2.shape[:2]:
            print("Image sizes do not match!")
            return None, None

        src_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        detector = cv2.SURF_create(self.minHessian)
        keypoints_src, descriptors_src = detector.detectAndCompute(src_gray, None)
        keypoints_dst, descriptors_dst = detector.detectAndCompute(dst_gray, None)

        matcher = cv2.FlannBasedMatcher()
        matches = matcher.match(descriptors_src, descriptors_dst)

        max_dist = max(match.distance for match in matches)
        min_dist = min(match.distance for match in matches)

        good_matches = [match for match in matches if match.distance < 3 * min_dist]

        matches_src = np.float32([keypoints_src[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        matches_dst = np.float32([keypoints_dst[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

        mask = None
        if len(matches_src) < 4:
            print("Not enough matches found.")
            return None, None

        H, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5)

        inliers_src = matches_src[mask.ravel() == 1]
        inliers_dst = matches_dst[mask.ravel() == 1]

        print("Number of inliers found:", len(inliers_src))

        if len(inliers_dst) < self.inlierThreshold:
            print("Not enough inliers for gain matching.")
            return None, None

        BGRval_1 = np.zeros((len(inliers_src), 4), dtype=np.float32)
        BGRval_2 = np.zeros((len(inliers_dst), 4), dtype=np.float32)

        for i, (x, y) in enumerate(inliers_src.reshape(-1, 2)):
            BGRval_1[i, 0] = img_1[y, x, 0]
            BGRval_1[i, 1] = img_1[y, x, 1]
            BGRval_1[i, 2] = img_1[y, x, 2]
            BGRval_1[i, 3] = 1

        for i, (x, y) in enumerate(inliers_dst.reshape(-1, 2)):
            BGRval_2[i, 0] = img_2[y, x, 0]
            BGRval_2[i, 1] = img_2[y, x, 1]
            BGRval_2[i, 2] = img_2[y, x, 2]
            BGRval_2[i, 3] = 1

        return BGRval_1, BGRval_2

    def find_BGR_for_region(self, img_1, img_2):
        if img_1.size == 0 or img_2.size == 0:
            print("No data to match gain with!")
            return None, None

        if img_1.shape[:2] != img_2.shape[:2]:
            print("Image sizes do not match!")
            return None, None

        BGRval_1 = np.vstack([channel.flatten() for channel in cv2.split(img_1)] + [np.ones_like(img_1[:, :, 0].flatten())]).T.astype(np.float32)
        BGRval_2 = np.vstack([channel.flatten() for channel in cv2.split(img_2)] + [np.ones_like(img_2[:, :, 0].flatten())]).T.astype(np.float32)

        return BGRval_1, BGRval_2

    def apply_gain_transform(self, src, dst, transform_matrix):
        if src is None:
            print("No image to apply gain transform to!")
            return

        if transform_matrix is None:
            print("Transform matrix is None. No transformation applied.")
            return

        # printing the transform matrix
        print("Transform matrix:")
        print(transform_matrix)

        dst[...] = cv2.warpPerspective(src, transform_matrix, (dst.shape[1], dst.shape[0]))
        cv2.cvtColor(dst, dst, cv2.COLOR_BGRA2BGR)
