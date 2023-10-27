import cv2
import numpy as np

class ImageTools:
    def __init__(self):
        self.minHessian = 400
        self.inlierThreshold = 15

    def findGainTransform(self, src_color, dst_color, useKeypoints=True):
        if src_color.shape[:2] != dst_color.shape[:2]:
            print("Image sizes do not match!")
            return False

        print("Finding transform for gain consistency...")

        if src_color is None or dst_color is None:
            print("No data to match gain with!")
            return False

        transformMatrix = np.eye(4, 4, dtype=np.float32)

        BGR_src, BGR_dst = self.findMatchingBGR(src_color, dst_color) if useKeypoints else self.findBGRForRegion(src_color, dst_color)

        BGR_src_transpose = np.transpose(BGR_src)
        transformMatrix = np.linalg.inv(BGR_src_transpose @ BGR_src) @ (BGR_src_transpose @ BGR_dst)
        transformMatrix = np.transpose(transformMatrix)

        print("Transform found.")
        return True, transformMatrix

    def findMatchingBGR(self, img_1, img_2):
        if img_1.shape[:2] != img_2.shape[:2]:
            print("Image sizes do not match!")
            return False

        src_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

        detector = cv2.ORB_create()
        keypoints_src, descriptors_src = detector.detectAndCompute(src_gray, None)
        keypoints_dst, descriptors_dst = detector.detectAndCompute(dst_gray, None)

        matcher = cv2.BFMatcher()
        matches = matcher.match(descriptors_src, descriptors_dst)

        min_dist = min(matches, key=lambda x: x.distance).distance
        good_matches = [match for match in matches if match.distance < 3 * min_dist]

        matches_src = np.float32([keypoints_src[match.queryIdx].pt for match in good_matches])
        matches_dst = np.float32([keypoints_dst[match.trainIdx].pt for match in good_matches])

        _, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5)

        inliers_mask = mask.ravel() == 1
        inliers_src = matches_src[inliers_mask]
        inliers_dst = matches_dst[inliers_mask]

        print(f"Number of inliers found: {len(inliers_src)}")

        if len(inliers_dst) < self.inlierThreshold:
            print("Not enough inliers for gain matching.")
            return False

        BGRval_1 = self.constructBGRMatrix(inliers_src, img_1)
        BGRval_2 = self.constructBGRMatrix(inliers_dst, img_2)

        return BGRval_1, BGRval_2

    def constructBGRMatrix(self, points, image):
        BGRval = np.zeros((len(points), 4), dtype=np.float32)
        for i, point in enumerate(points):
            x, y = int(point[0]), int(point[1])
            BGRval[i, 0] = image[y, x, 0]
            BGRval[i, 1] = image[y, x, 1]
            BGRval[i, 2] = image[y, x, 2]
            BGRval[i, 3] = 1
        return BGRval

    def findBGRForRegion(self, img_1, img_2):
        if img_1.shape[:2] != img_2.shape[:2]:
            print("Image sizes do not match!")
            return False

        channels1 = cv2.split(img_1)
        channels2 = cv2.split(img_2)

        BGRval_1 = np.hstack([channel.reshape(-1, 1) for channel in channels1] + [np.ones((img_1.shape[0] * img_1.shape[1], 1), dtype=np.float32)])
        BGRval_2 = np.hstack([channel.reshape(-1, 1) for channel in channels2] + [np.ones((img_2.shape[0] * img_2.shape[1], 1), dtype=np.float32)])

        BGRval_1 = np.transpose(BGRval_1)
        BGRval_2 = np.transpose(BGRval_2)

        BGRval_1 = BGRval_1.astype(np.float32)
        BGRval_2 = BGRval_2.astype(np.float32)

        return BGRval_1, BGRval_2

    def applyGainTransform(self, src, transformMatrix):
        if src is None:
            print("No image to apply gain transform to!")
            return None

        # Check if the transformation matrix is 4x4 (3D perspective transformation)
        if transformMatrix is not None and transformMatrix.shape == (4, 4):
            dst = cv2.perspectiveTransform(src, transformMatrix)
            dst = cv2.cvtColor(dst, cv2.COLOR_BGRA2BGR)
            return dst
        else:
            print("Invalid 3D perspective transformation matrix. Expected shape (4, 4).")
            return None
