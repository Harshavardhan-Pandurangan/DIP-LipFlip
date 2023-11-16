
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class LaplacianBlending{
public:
    LaplacianBlending(const Mat_<Vec3f>& _left, const Mat_<Vec3f>& _right, const Mat_<float>& _blendMask, int _levels);
    Mat_<Vec3f> blend();

private:
    Mat_<Vec3f> left;
    Mat_<Vec3f> right;
    Mat_<float> blendMask;
    vector<Mat_<Vec3f> > leftLapPyr,rightLapPyr,resultLapPyr;
    Mat leftSmallestLevel, rightSmallestLevel, resultSmallestLevel;
    vector<Mat_<Vec3f> > maskGaussianPyramid;
    int levels;

    void buildPyramids();
    void buildGaussianPyramid();
    void buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& smallestLevel);
    Mat_<Vec3f> reconstructImgFromLapPyramid();
    void blendLapPyrs();

};
