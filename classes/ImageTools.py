import cv2
import numpy as np

class ImageTools:
    def __init__(self):
        self.minHessian = 400
        self.inlierThreshold = 15

    def find_gain_transform(self, src_color, dst_color, transform_matrix, use_keypoints):
        if src_color.shape[:2] != dst_color.shape[:2]:
            print("Image sizes do not match!")
            return False

        print("Finding transform for gain consistency...")

        if src_color is None or dst_color is None:
            print("No data to match gain with!")
            return False

        transform_matrix = np.eye(4, dtype=np.float32)

        BGR_src, BGR_dst = self.find_matching_BGR(src_color, dst_color) if use_keypoints else self.find_BGR_for_region(src_color, dst_color)

        BGR_src_transpose = np.transpose(BGR_src)
        transform_matrix = np.linalg.inv(BGR_src_transpose @ BGR_src) @ (BGR_src_transpose @ BGR_dst)

        transform_matrix = np.transpose(transform_matrix)

        print("Transform found.")
        return True

    def find_matching_BGR(self, img_1, img_2):
        if img_1.shape[:2] != img_2.shape[:2]:
            print("Image sizes do not match!")
            return False

        src_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        # Adjust ORB parameters for sensitivity
        orb = cv2.ORB_create()
        keypoints_src, descriptors_src = orb.detectAndCompute(src_gray, None)
        keypoints_dst, descriptors_dst = orb.detectAndCompute(dst_gray, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_src, descriptors_dst)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = [match for match in matches if match.distance < 3 * matches[0].distance]

        matches_src = np.float32([keypoints_src[match.queryIdx].pt for match in good_matches])
        matches_dst = np.float32([keypoints_dst[match.trainIdx].pt for match in good_matches])

        if len(matches_src) < 4:
            print("Not enough matches found.")
            return False

        mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5)[1]

        inliers_src = [matches_src[i] for i in range(len(mask)) if mask[i] != 0]
        inliers_dst = [matches_dst[i] for i in range(len(mask)) if mask[i] != 0]

        print("Number of inliers found:", len(inliers_src))

        if len(inliers_dst) < self.inlierThreshold:
            print("Not enough inliers for gain matching.")
            return False

        BGRval_1 = np.zeros((len(inliers_src), 4), dtype=np.float32)
        BGRval_2 = np.zeros((len(inliers_dst), 4), dtype=np.float32)

        for i in range(len(inliers_src)):
            x, y = int(inliers_src[i][0]), int(inliers_src[i][1])
            BGRval_1[i, 0] = img_1[y, x, 0]
            BGRval_1[i, 1] = img_1[y, x, 1]
            BGRval_1[i, 2] = img_1[y, x, 2]
            BGRval_1[i, 3] = 1

        for i in range(len(inliers_dst)):
            x, y = int(inliers_dst[i][0]), int(inliers_dst[i][1])
            BGRval_2[i, 0] = img_2[y, x, 0]
            BGRval_2[i, 1] = img_2[y, x, 1]
            BGRval_2[i, 2] = img_2[y, x, 2]
            BGRval_2[i, 3] = 1

        return BGRval_1, BGRval_2

    def find_BGR_for_region(self, img_1, img_2):
        if img_1.shape[:2] != img_2.shape[:2]:
            print("Image sizes do not match!")
            return False

        channels1 = cv2.split(img_1)
        channels2 = cv2.split(img_2)

        BGRval_1 = np.empty((3, img_1.size), dtype=np.float32)
        BGRval_2 = np.empty((3, img_2.size), dtype=np.float32)

        for c in range(3):
            BGRval_1[c] = channels1[c].reshape(1, -1)
            BGRval_2[c] = channels2[c].reshape(1, -1)

        ones_matrix = np.ones((1, img_1.size), dtype=np.float32)
        BGRval_1 = np.vstack((BGRval_1, ones_matrix))
        BGRval_2 = np.vstack((BGRval_2, ones_matrix))

        BGRval_1 = BGRval_1.transpose()
        BGRval_2 = BGRval_2.transpose()

        return BGRval_1, BGRval_2

    def apply_gain_transform(self, src, dst, transform_matrix):
        if src is None:
            print("No image to apply gain transform to!")
            return

        print("Transform matrix:")
        print(transform_matrix)

        dst = cv2.warpPerspective(src, transform_matrix, src.shape[:2][::-1])
        dst = cv2.cvtColor(dst, cv2.COLOR_BGRA2BGR)
