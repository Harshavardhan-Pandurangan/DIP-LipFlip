#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class DetectionTools{
public:
    DetectionTools();
    void findLipRegion(Mat& frame);
    Mat matchHistograms(Mat& src, Mat& dst);
    Mat mouthROI;
    Size mouthSize;
    Point mouthLoc;
    bool DEBUG;

    Rect rectLarge;
    Rect rectMouth;
    Rect rectMouthCascade;

    Mat bottomMouth;
    Mat topCornerMouth;

private:
    bool mouthOpen;
    double scale; // How much to downsize the image for face search
    double t; // Used to keep track of frame rate
    string fn_haar;
    CascadeClassifier face_cascade;
    CascadeClassifier mouth_cascade;
    Rect rectSmall; //Location of face in small image
    void checkMouthOpen();
    void findHist(const Mat& input, double* h, double* cdf);
};
