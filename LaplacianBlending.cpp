#include "opencv2/opencv.hpp"
#include "include/LaplacianBlending.hpp"

using namespace cv;

void LaplacianBlending::buildPyramids() {
    buildLaplacianPyramid(left,leftLapPyr,leftSmallestLevel);
    buildLaplacianPyramid(right,rightLapPyr,rightSmallestLevel);
    buildGaussianPyramid();
}

void LaplacianBlending::buildGaussianPyramid() {
    assert(leftLapPyr.size()>0);

    maskGaussianPyramid.clear();
    Mat currentImg;
    cvtColor(blendMask, currentImg, COLOR_GRAY2BGR);
    maskGaussianPyramid.push_back(currentImg); //highest level

    currentImg = blendMask;
    for (int l=1; l<levels+1; l++) {
        Mat _down;
        if (leftLapPyr.size() > l) {
            pyrDown(currentImg, _down, leftLapPyr[l].size());
        } else {
            pyrDown(currentImg, _down, leftSmallestLevel.size()); //smallest level
        }

        Mat down;
        cvtColor(_down, down, COLOR_GRAY2BGR);
        maskGaussianPyramid.push_back(down);
        currentImg = _down;
    }
}

void LaplacianBlending::buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& smallestLevel) {
    lapPyr.clear();
    Mat currentImg = img;
    for (int l=0; l<levels; l++) {
        Mat down,up;
        pyrDown(currentImg, down);
        pyrUp(down, up, currentImg.size());
        Mat lap = currentImg - up;
        lapPyr.push_back(lap);
        currentImg = down;
    }
    currentImg.copyTo(smallestLevel);
}

Mat_<Vec3f> LaplacianBlending::reconstructImgFromLapPyramid() {
    Mat currentImg = resultSmallestLevel;
    for (int l=levels-1; l>=0; l--) {
        Mat up;

        pyrUp(currentImg, up, resultLapPyr[l].size());
        currentImg = up + resultLapPyr[l];
    }
    return currentImg;
}

void LaplacianBlending::blendLapPyrs() {
    resultSmallestLevel = leftSmallestLevel.mul(maskGaussianPyramid.back()) +
                                rightSmallestLevel.mul(Scalar(1.0,1.0,1.0) - maskGaussianPyramid.back());
    for (int l=0; l<levels; l++) {
        Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
        Mat antiMask = Scalar(1.0,1.0,1.0) - maskGaussianPyramid[l];
        Mat B = rightLapPyr[l].mul(antiMask);
        Mat_<Vec3f> blendedLevel = A + B;

        resultLapPyr.push_back(blendedLevel);
    }
}


LaplacianBlending::LaplacianBlending(const Mat_<Vec3f>& _left, const Mat_<Vec3f>& _right, const Mat_<float>& _blendMask, int _levels):
left(_left),right(_right),blendMask(_blendMask),levels(_levels)
{
    assert(_left.size() == _right.size());
    assert(_left.size() == _blendMask.size());
    buildPyramids();
    blendLapPyrs();
};

Mat_<Vec3f> LaplacianBlending::blend() {
    return reconstructImgFromLapPyramid();
}





