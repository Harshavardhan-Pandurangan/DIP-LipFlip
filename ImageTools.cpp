
#include "include/ImageTools.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

ImageTools::ImageTools(){
    minHessian = 400;
    inlierThreshold = 15;
}


/// This function finds the transform that will match the src to the dst
bool ImageTools::findGainTransform(Mat &src_color, Mat &dst_color, Mat &transformMatrix, bool useKeypoints){

    /// Check to make sure the images are the same size. If they aren't then we have a problem with matching the coordinate locations
    if((src_color.cols != dst_color.cols) || (dst_color.rows != src_color.rows)){
        std::cout << "Image sizes do not match!" << std::endl; return false;
    }

    std::cout << "Finding transform for gain consistency..." << std::endl;

    if( !src_color.data || !dst_color.data )
    { std::cout<< " No data to match gain with!" << std::endl; return false; }

    transformMatrix = Mat::eye(4,4,CV_32FC3);


    Mat BGR_src, BGR_dst;

    if(useKeypoints){

        bool foundMatches = false;
        foundMatches = findMatchingBGR(src_color, dst_color, BGR_src, BGR_dst);

        if (!foundMatches) {
            return false;
        }
    }else{
        findBGRForRegion(src_color, dst_color, BGR_src, BGR_dst);
    }

    Mat BGR_src_transpose;
    transpose(BGR_src, BGR_src_transpose);
    transformMatrix = ((BGR_src_transpose*BGR_src).inv())*(BGR_src_transpose*BGR_dst);
    transpose(transformMatrix, transformMatrix);

    /* Note we transpose the transform matrix so we can apply it using the transpose command in opencv. OpenCV expect it to be likes this:
     [           ] [ R ]
     [   4 x 4   ] [ G ]  = Result
     [ transform ] [ B ]
     [           ] [ 1 ]

     While the way we have found the transform matrix above is
     [ R G B 1 ] [           ]
     [   4 x 4   ]  = Result
     [ transform ]
     [           ]
     */
    std::cout << "Transform found." << std::endl;

    return true;

}

/// Find corresponding keypoints and construct a matrix of the matching BGR values
bool ImageTools::findMatchingBGR(Mat &img_1, Mat &img_2, Mat &BGRval_1, Mat &BGRval_2){

    if( !img_1.data || !img_2.data )
    { std::cout<< " No data to match gain with!" << std::endl; return false; }

    /// Check to make sure the images are the same size. If they aren't then we have a problem with matching the coordinate locations
    if((img_1.cols != img_2.cols) || (img_1.rows != img_2.rows)){
        std::cout << "Image sizes do not match!" << std::endl; return false;
    }

    // Convert to grayscale
    Mat src_gray, dst_gray;
    cvtColor(img_1, src_gray, COLOR_BGR2GRAY );
    cvtColor(img_2, dst_gray, COLOR_BGR2GRAY );

    /// Detect the keypoints using SURF
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints_src, keypoints_dst;
    detector->detect( src_gray, keypoints_src );
    detector->detect( dst_gray, keypoints_dst );

    /// Calculate descriptors
    Ptr<xfeatures2d::SURF> extractor = xfeatures2d::SURF::create();
    Mat descriptors_src, descriptors_dst;
    extractor->detectAndCompute( src_gray, Mat(), keypoints_src, descriptors_src );
    extractor->detectAndCompute( dst_gray, Mat(), keypoints_dst, descriptors_dst );

    /// Find matching descriptors;
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_src, descriptors_dst, matches );

    double max_dist = 0; double min_dist = 100;

    /// Calculate max and min distances between keypoints
    for( int i = 0; i < descriptors_src.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    /// Take only good matches
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_src.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
    { good_matches.push_back( matches[i]); }
    }

    /// Get keypoints from good matches
    std::vector<Point2f> matches_src;
    std::vector<Point2f> matches_dst;
    for( int i = 0; i < good_matches.size(); i++ )
    {
        matches_src.push_back( keypoints_src[ good_matches[i].queryIdx ].pt );
        matches_dst.push_back( keypoints_dst[ good_matches[i].trainIdx ].pt );
    }

    cv::Mat mask;
    if(matches_src.size() < 4){
        std::cout << "Not enough matches found." << std::endl;
        return false;
    }
    Mat H = findHomography(matches_src, matches_dst, RANSAC, 5, mask);

    /// Get inlier coordinate locations
    std::vector<Point2f> inliers_src, inliers_dst;
    for(int i = 0; i < mask.rows; i++){
        if((unsigned int)mask.at<uchar>(i) != 0 ){
            inliers_src.push_back(matches_src[i]);
            inliers_dst.push_back(matches_dst[i]);
        }
    }

    std::cout << " Number of inliers found: " << inliers_src.size() <<  std::endl;

    /// Check if there's enough inliers
    if (inliers_dst.size() < inlierThreshold) {
        std::cout<< " Not enough inliers for gain matching." << std::endl;
        return false;
    }

    /// Construct matching BGR matrices
    /*
     [B G R 1]
     [B G R 1]
     [B G R 1]
     [  ...  ]
     */

    BGRval_1 = Mat::zeros(inliers_src.size(), 4, CV_32FC1);
    BGRval_2 = Mat::zeros(inliers_dst.size(), 4, CV_32FC1);
    int x, y;
    for(int i = 0; i < inliers_src.size(); i++){
        x = cvRound(inliers_src[i].x);
        y = cvRound(inliers_src[i].y);
        BGRval_1.at<float>(i,0) = img_1.at<cv::Vec3b>(y,x)[0];
        BGRval_1.at<float>(i,1) = img_1.at<cv::Vec3b>(y,x)[1];
        BGRval_1.at<float>(i,2) = img_1.at<cv::Vec3b>(y,x)[2];
        BGRval_1.at<float>(i,3) = 1;
    }
    for(int i = 0; i < inliers_dst.size(); i++){
        x = cvRound(inliers_dst[i].x);
        y = cvRound(inliers_dst[i].y);
        BGRval_2.at<float>(i,0) = img_2.at<cv::Vec3b>(y,x)[0];
        BGRval_2.at<float>(i,1) = img_2.at<cv::Vec3b>(y,x)[1];
        BGRval_2.at<float>(i,2) = img_2.at<cv::Vec3b>(y,x)[2];
        BGRval_2.at<float>(i,3) = 1;
    }

    return true;
}

/// Construct two BGR matrixes using all pixels in a given region
bool ImageTools::findBGRForRegion(Mat &img_1, Mat &img_2, Mat &BGRval_1, Mat &BGRval_2){

    /// Construct matching BGR matrices
    /*
     [B G R 1]
     [B G R 1]
     [B G R 1]
     [  ...  ]
     */

    if( !img_1.data || !img_2.data )
    { std::cout<< " No data to match gain with!" << std::endl; return false; }

    /// Check to make sure the images are the same size. There needs to be the same number of pixels in each BGR matrix
    if((img_1.cols != img_2.cols) || (img_1.rows != img_2.rows)){
        std::cout << "Image sizes do not match!" << std::endl; return false;
    }

    /// Split into channels
    vector<Mat> channels1, channels2;
    split(img_1, channels1);
    split(img_2, channels2);

    /// Initialize BGR matrices
    Size imgsize = img_1.size();

    /// Vectorize
    Mat currChannel;
    for(int c = 0; c < 3; c++){
        BGRval_1.push_back(channels1[c].reshape(0,1));
        BGRval_2.push_back(channels2[c].reshape(0,1));
    }

    Mat onesMatrix = Mat::ones(1,imgsize.width*imgsize.height, CV_8U);

    BGRval_1.push_back(onesMatrix);
    BGRval_2.push_back(onesMatrix);

    /// Transpose
    transpose(BGRval_1, BGRval_1);
    transpose(BGRval_2, BGRval_2);

    /// Change type to float
    BGRval_1.convertTo(BGRval_1, CV_32FC1);
    BGRval_2.convertTo(BGRval_2, CV_32FC1);

//    /// Check size
//    std::cout << "Size of BGRval_1 is " << BGRval_1.size() << std::endl;
//    std::cout << "Size of BGRval_2 is " << BGRval_2.size() << std::endl;

    return true;
}

void ImageTools::applyGainTransform(Mat &src, Mat &dst, Mat &transformMatrix){

    if(src.empty()){
        std::cout<<"No image to apply gain transform to!"<<std::endl;
        return ;
    }

    // printing the transform matrix
    std::cout << "Transform matrix: " << std::endl;
    std::cout << transformMatrix << std::endl;

    transform(src, dst, transformMatrix);
    cvtColor(dst,dst,COLOR_BGRA2BGR);
    return;
}
