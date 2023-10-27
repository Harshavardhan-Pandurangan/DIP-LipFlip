import cv2
import numpy as np

class DetectionTools:
    def __init__(self):
        self.DEBUG = False
        self.mouthOpen = False
        self.scale = 1
        self.t = 0

        self.fn_haar = "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
        self.face_cascade = cv2.CascadeClassifier(self.fn_haar)

        if self.face_cascade.empty():
            print("Cannot find Haar cascade.")

        self.mouth_cascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml")

        if self.mouth_cascade.empty():
            print("Cannot find Haar cascade.")

        self.mouthROI = None

    def findLipRegion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smallFrame = cv2.resize(gray, None, fx=1/self.scale, fy=1/self.scale, interpolation=cv2.INTER_LINEAR)

        faces = self.face_cascade.detectMultiScale(
            smallFrame,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT,
            minSize=(30, 30)
        )

        if not faces.any():
            self.mouthROI = None
            return

        rectSmall = max(faces, key=lambda x: x[2])  # Select the largest face

        x, y, w, h = rectSmall
        largeTL = (int(x * self.scale), int(y * self.scale))
        largeSize = (int((w - 1) * self.scale), int((h - 1) * self.scale))
        rectLarge = (largeTL, largeSize)

        mouthSize = (int(largeSize[0] * 0.6), int(largeSize[1] * 0.5))
        mouthTL = (largeTL[0] + int(largeSize[0] * 0.2), largeTL[1] + int(largeSize[1] * 0.65))
        rectMouth = (mouthTL, mouthSize)

        if (rectMouth[1][1] + rectMouth[0][1]) >= frame.shape[0] or (rectMouth[1][0] + rectMouth[0][0]) >= frame.shape[1]:
            return

        self.mouthROI = frame[rectMouth[0][1]:rectMouth[0][1] + rectMouth[1][1], rectMouth[0][0]:rectMouth[0][0] + rectMouth[1][0]]

        mouths = self.mouth_cascade.detectMultiScale(
            self.mouthROI,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT,
            minSize=(10, 10)
        )

        if mouths.any():
            rectMouthCascade = mouths[0]
            newTL = (rectMouthCascade[0] + rectMouth[0][0], rectMouthCascade[1] + rectMouth[0][1])
            rectMouthCascade = (newTL, rectMouthCascade[2:])
            deltaX = int(rectMouthCascade[2] * 0.25)
            deltaY = int(rectMouthCascade[3] * 0.3)
            rectMouthCascade = (
                rectMouthCascade[0][0] - deltaX, rectMouthCascade[0][1] - deltaY,
                rectMouthCascade[2] + 2 * deltaX, rectMouthCascade[3] + 2 * deltaY
            )

            self.mouthROI = frame[rectMouthCascade[1]:rectMouthCascade[1] + rectMouthCascade[3],
                            rectMouthCascade[0]:rectMouthCascade[0] + rectMouthCascade[2]]
            self.mouthOpen = True

    def checkMouthOpen(self):
        if self.mouthOpen:
            bottomMouth = self.mouthROI[int(self.mouthROI.shape[0] * 0.3):int(self.mouthROI.shape[0] * 0.5),
                         int(self.mouthROI.shape[1] * 0.3):int(self.mouthROI.shape[1] * 0.7)]
        else:
            bottomMouth = self.mouthROI[int(self.mouthROI.shape[0] * 0.5):self.mouthROI.shape[0],
                         int(self.mouthROI.shape[1] * 0.3):int(self.mouthROI.shape[1] * 0.7)]

        topCornerMouth = self.mouthROI[0:int(self.mouthROI.shape[0] * 0.3), 0:int(self.mouthROI.shape[1] * 0.3)]

        meanBottom = np.mean(bottomMouth)
        meanCorner = np.mean(topCornerMouth)

        if meanBottom < 0.7 * meanCorner:
            self.mouthOpen = True
            print("One of the mouths is open!")
        if meanBottom > 0.8 * meanCorner:
            self.mouthOpen = False
