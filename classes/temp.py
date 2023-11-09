import cv2
import numpy as np

class ColorBalanceMatcher:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_color_balance(self, image1, image2):
        kp1, des1 = self._detect_and_compute(image1)
        kp2, des2 = self._detect_and_compute(image2)

        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:100]  # Adjust as needed
        points1 = np.float32([kp1[match.queryIdx].pt for match in good_matches])
        points2 = np.float32([kp2[match.trainIdx].pt for match in good_matches])

        mean_color1 = np.mean(image1[points1.astype(int)[:, 1], points1.astype(int)[:, 0]], axis=0)
        mean_color2 = np.mean(image2[points2.astype(int)[:, 1], points2.astype(int)[:, 0]], axis=0)
        gain_factors = mean_color2 / (mean_color1 + 1e-6)

        matched_image = (image1 * gain_factors).clip(0, 255).astype('uint8')
        return matched_image

    def _detect_and_compute(self, image):
        kp, des = self.orb.detectAndCompute(image, None)
        return kp, des

# if __name__ == "__main__":
#     matcher = ColorBalanceMatcher()

#     image1 = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)
#     image2 = cv2.imread('image2.jpg', cv2.IMREAD_COLOR)

#     matched_image = matcher.match_color_balance(image1, image2)

#     cv2.imwrite('matched_image.jpg', matched_image)
#     cv2.imshow('Matched Image', matched_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
