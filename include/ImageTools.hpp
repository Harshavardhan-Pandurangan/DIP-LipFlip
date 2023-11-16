#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/xfeatures2d/nonfree.hpp> //For surf detector

using namespace cv;

class ImageTools{
public:
    ImageTools();

    // Match src to dst. Flag indicated whether we want to use corresponding keypoints or whether we want to use all the pixels in the two regions given
    bool findGainTransform(Mat &src_color, Mat &dst_color, Mat &transformMatrix, bool useKeypoints);

    // Apply the transformation found
    void applyGainTransform(Mat &src, Mat &dst, Mat &transform);


private:
    int minHessian;
    int inlierThreshold;

    // Find corresponding keypoints and construct a matrix of the matching BGR values
    bool findMatchingBGR(Mat &img_1, Mat &img_2, Mat &BGRval_1, Mat &BGRval_2);

    // Construct two BGR matrixes using all pixels in a given region
    bool findBGRForRegion(Mat &img_1, Mat &img_2, Mat &BGRval_1, Mat &BGRval_2);



};
