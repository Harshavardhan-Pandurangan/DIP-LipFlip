
/// Use all libraries because I'm lazy.
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp> //For surf detector

#include <iostream>

#include "include/DetectionTools.hpp"
#include "include/LaplacianBlending.hpp"
#include "include/ImageTools.hpp"

using namespace cv;
using namespace std;


///Function for calling Laplacian Blend
Mat_<Vec3f>  LaplacianBlend(const Mat_<Vec3f>& l, const Mat_<Vec3f>& r, const Mat_<float>& m);

bool DEBUG = false;

// SCALE SIZE
//Size scaleSize = Size(320,180);
//Size scaleSize = Size(640,360);
Size scaleSize = Size(400,225);

int main(int argc, const char *argv[]) {

    int deviceId1 = 0; /// Default camera ID
    int deviceId2 = 1; /// Secondary camera ID

    cout << "OpenCV version: " << CV_VERSION << endl;
    cout << "OpenCV build information: " << cv::getBuildInformation() << endl;

    /// Open first webcam
    VideoCapture cap1(deviceId1);
    if(!cap1.isOpened()) {
        cerr << "Capture Device ID " << deviceId1 << " cannot be opened." << endl;
        return -1;
    }

    /// Open second webcam
    VideoCapture cap2(deviceId2);
    if(!cap2.isOpened()) {
        cerr << "Capture Device ID " << deviceId2 << " cannot be opened." << endl;
        return -1;
    }

    /// Initialize the window
    string Window1 = "Webcam 1";
    namedWindow(Window1);

    string Window2 = "Webcam 2";
    namedWindow(Window2);

    /// WTF kyle's webcam is dumb
    waitKey(1000);

    /// Find the camera equalization transform
    Mat orig, match, tMatrix;
    ImageTools IT;
    bool foundGoodTransform = false;
    bool possibleTransform = false;

    while (!foundGoodTransform) {
        cap1 >> match;
        cap2 >> orig;

        // Scale image down
        resize(orig ,orig, scaleSize);
        resize(match, match, scaleSize);

        bool useKeypoints = 1;
        possibleTransform = IT.findGainTransform(orig, match, tMatrix,useKeypoints);

        if (!possibleTransform) {
            std::cout << "Trying again..." << std::endl;
            continue;
        }

        /// Test transform
        Mat img_transform;
        IT.applyGainTransform(orig, img_transform, tMatrix);

        // TODO: Display images strategicially on screen, or perhaps display them side by side on the same window?

        imshow("Camera 1", orig);
        imshow("Camera 2", match);
        imshow("Transformed image", img_transform);
        waitKey(200);

        string userInput;
        do{
            std::cout << "Is this a good transform? (y/n)" << std::endl;
            std::cin >> userInput;
            if(userInput == "y"){
                foundGoodTransform = true;
            }
        }while (userInput != "y" && userInput != "n");

    }

    std::cout << "Starting lip flip session..." << std::endl;

    // TODO: save transform as a text file and have option to load it up instead of doing this calibration every session.

    destroyAllWindows();

    Mat frame1; /// Holds the current frame from the Video device:
    Mat frame2;

    DetectionTools d1;
    DetectionTools d2;

    Mat frame1_temp;
    Mat frame2_temp;

    int blend_range = 6;
    double t = 0;

    for(;;) {

        t = (double)getTickCount();

        cap1 >> frame1;
        cap2 >> frame2;

        resize(frame1,frame1,scaleSize);
        resize(frame2,frame2,scaleSize);


        // Apply the gain matching transform
        IT.applyGainTransform(frame2,frame2,tMatrix);


        frame2.copyTo(frame2_temp);
        frame1.copyTo(frame1_temp);

        d1.findLipRegion(frame1);
        d2.findLipRegion(frame2);


        if(!d1.mouthROI.empty() && !d2.mouthROI.empty()){

            /// Scale mouths to match the corresponding mouth (we have to use columns so that the "openMouth" capability works
            double cols1 = (double)d1.mouthROI.cols;
            double cols2 = (double)d2.mouthROI.cols;

            Mat scaledMouth1(cvRound(d1.mouthROI.rows*(cols2/cols1)),
                             cvRound(d1.mouthROI.cols*(cols2/cols1)), CV_8UC1) ;
            Mat scaledMouth2(cvRound(d2.mouthROI.rows*(cols1/cols2)),
                             cvRound(d2.mouthROI.cols*(cols1/cols2)), CV_8UC1) ;

            resize(d1.mouthROI, scaledMouth1, scaledMouth1.size());
            resize(d2.mouthROI, scaledMouth2, scaledMouth2.size());

            if(d1.mouthLoc.y+scaledMouth2.rows <= frame1.rows && d1.mouthLoc.x+scaledMouth2.cols <= frame1.cols) {
                /// copy mouth over
                scaledMouth2.copyTo(frame1(Rect(d1.mouthLoc.x,
                                                d1.mouthLoc.y,
                                                scaledMouth2.cols,
                                                scaledMouth2.rows)));

                Mat l8u_1 = frame1;

                /// blending
                Mat r8u_1 = frame1_temp;
                Mat_<Vec3f> l_1; l8u_1.convertTo(l_1,CV_32F,1.0/255.0);
                Mat_<Vec3f> r_1; r8u_1.convertTo(r_1,CV_32F,1.0/255.0);

                Mat_<float> m_1(l_1.rows,l_1.cols,0.0);
                Point mouthCenter1 = Point(d1.mouthLoc.x+0.5*scaledMouth2.cols,d1.mouthLoc.y+0.5*scaledMouth2.rows);
                Size mouthAxis1 = Size(scaledMouth2.cols*0.5-blend_range,scaledMouth2.rows*0.5-blend_range);
                ellipse(m_1,mouthCenter1, mouthAxis1, 0,0,360,Scalar(1), -1);

                Mat_<Vec3f> blend_1 = LaplacianBlend(l_1, r_1, m_1);

                if(DEBUG){
                rectangle(blend_1, d1.rectMouth, CV_RGB(0, 255,0), 1);
                rectangle(blend_1, d1.rectMouthCascade, CV_RGB(0, 0,255), 1);
                }

                imshow(Window1,blend_1);
                if(DEBUG){imshow("Mask1", m_1);}
            }

            if(d2.mouthLoc.y+scaledMouth1.rows <= frame2.rows && d2.mouthLoc.x+scaledMouth1.cols <= frame2.cols) {
                /// copy mouth over
                scaledMouth1.copyTo(frame2(Rect(d2.mouthLoc.x,
                                                d2.mouthLoc.y,
                                                scaledMouth1.cols,
                                                scaledMouth1.rows)));
                /// blending
                Mat l8u_2 = frame2;
                Mat r8u_2 = frame2_temp;
                Mat_<Vec3f> l_2; l8u_2.convertTo(l_2,CV_32F,1.0/255.0);
                Mat_<Vec3f> r_2; r8u_2.convertTo(r_2,CV_32F,1.0/255.0);

                Mat_<float> m_2(l_2.rows,l_2.cols,0.0);
                Point mouthCenter2 = Point(d2.mouthLoc.x+0.5*scaledMouth1.cols,d2.mouthLoc.y+0.5*scaledMouth1.rows);
                Size mouthAxis2 = Size(scaledMouth1.cols*0.5-blend_range,scaledMouth1.rows*0.5-blend_range);
                ellipse(m_2,mouthCenter2, mouthAxis2, 0,0,360,Scalar(1), -1);


                Mat_<Vec3f> blend_2 = LaplacianBlend(l_2, r_2, m_2);

                if(DEBUG){
                rectangle(blend_2, d2.rectMouthCascade, CV_RGB(0,0,255), 1);
                rectangle(blend_2, d2.rectMouth, CV_RGB(0, 255,0), 1);
                }

                Mat blend_2_mat = Mat(blend_2);
                imshow(Window2,blend_2_mat);
                if(DEBUG){imshow("Mask2", m_2);}
            }
            t = (double)getTickCount() - t;
            printf( "%g ms\n", t/((double)getTickFrequency()*1000.) );
        } else{
            imshow(Window2,frame2_temp);
            imshow(Window1, frame1_temp);
        }


        /// And display it:
        char key = (char) waitKey(20);
        /// Exit this loop on escape:
        if(key == 27)
            break;
    }


    return 0;
}

Mat_<Vec3f>  LaplacianBlend(const Mat_<Vec3f>& l, const Mat_<Vec3f>& r, const Mat_<float>& m) {
    LaplacianBlending lb(l,r,m,20); ///Adjust the last integer to adjust number of pyramid levels
    return lb.blend();
}



