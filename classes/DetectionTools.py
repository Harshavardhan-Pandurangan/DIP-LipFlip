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
            print("Cannot find Haar cascade for face.")

        self.mouth_cascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml")
        if self.mouth_cascade.empty():
            print("Cannot find Haar cascade for mouth.")

        self.mouthROI = None

    def find_lip_region(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(gray, (int(frame.shape[1] / self.scale), int(frame.shape[0] / self.scale)))

        faces = self.face_cascade.detectMultiScale(
            small_frame,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=0 | cv2.CASCADE_FIND_BIGGEST_OBJECT,
            minSize=(30, 30)
        )

        if not faces:
            self.mouthROI = None
            return

        # Restrict to largest face
        facesize = faces[0][2]
        rect_small = faces[0]
        for face in faces[1:]:
            if face[2] > facesize:
                rect_small = face
                facesize = face[2]

        # Scale the rectangle ROI back up
        large_tl = (int(rect_small[0] * self.scale), int(rect_small[1] * self.scale))
        large_size = (int((rect_small[2] - 1) * self.scale), int((rect_small[3] - 1) * self.scale))
        rect_large = (large_tl, large_size)

        mouth_size = (int(large_size[0] * 0.6), int(large_size[1] * 0.5))
        mouth_tl = (int(large_tl[0] + (large_size[0] * 0.2)), int(large_tl[1] + (large_size[1] * 0.65)))
        rect_mouth = (mouth_tl, mouth_size)

        # Make sure the mouth region is not out of frame
        if (rect_mouth[1][1] + rect_mouth[0][1]) >= frame.shape[0] or (rect_mouth[1][0] + rect_mouth[0][0]) >= frame.shape[1]:
            return

        # Extract mouth region
        self.mouthROI = frame[rect_mouth[0][1]:rect_mouth[0][1] + rect_mouth[1][1],
                        rect_mouth[0][0]:rect_mouth[0][0] + rect_mouth[1][0]]

        # Try mouth detection within mouth region
        mouths = self.mouth_cascade.detectMultiScale(
            self.mouthROI,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=0 | cv2.CASCADE_FIND_BIGGEST_OBJECT,
            minSize=(10, 10)
        )

        if mouths:
            rect_mouth_cascade = mouths[0]
            new_tl = (rect_mouth_cascade[0] + rect_mouth[0][0], rect_mouth_cascade[1] + rect_mouth[0][1])
            rect_mouth_cascade = (new_tl, (rect_mouth_cascade[2], rect_mouth_cascade[3]))

            # Make mouth region larger
            delta_x = int(rect_mouth_cascade[1][0] * 0.25)
            delta_y = int(rect_mouth_cascade[1][1] * 0.3)
            rect_mouth_cascade = (
                rect_mouth_cascade[0][0] - delta_x, rect_mouth_cascade[0][1] - delta_y,
                rect_mouth_cascade[1][0] + 2 * delta_x, rect_mouth_cascade[1][1] + 2 * delta_y
            )

            # Mouth region is now the one found by the mouth cascade
            self.mouthROI = frame[rect_mouth_cascade[1]:rect_mouth_cascade[1] + rect_mouth_cascade[3],
                            rect_mouth_cascade[0]:rect_mouth_cascade[0] + rect_mouth_cascade[2]]
        else:
            self.mouthROI = None

    def check_mouth_open(self):
        if self.mouthROI is not None:
            # Following code is only working like 50% of the time when you open your mouth.

            # Do a quick check to see if the mouth is open (this might not work for people who have beards...)
            if self.mouthOpen:
                bottom_mouth = self.mouthROI[int(self.mouthROI.shape[0] * 0.3):int(self.mouthROI.shape[0] * 0.5),
                               int(self.mouthROI.shape[1] * 0.3):int(self.mouthROI.shape[1] * 0.7)]
            else:
                bottom_mouth = self.mouthROI[int(self.mouthROI.shape[0] * 0.5):,
                               int(self.mouthROI.shape[1] * 0.3):int(self.mouthROI.shape[1] * 0.7)]

            top_corner_mouth = self.mouthROI[:int(self.mouthROI.shape[0] * 0.3),
                               :int(self.mouthROI.shape[1] * 0.3)]

            # Draw these regions
            mean_bottom = np.mean(bottom_mouth)
            mean_corner = np.mean(top_corner_mouth)

            if mean_bottom < 0.7 * mean_corner:
                self.mouthOpen = True
                print("One of the mouths is open!")
            if mean_bottom > 0.8 * mean_corner:
                self.mouthOpen = False
