
#include <opencv2/opencv.hpp>
#include <iostream>
#include "include/DetectionTools.hpp"

using namespace cv;


DetectionTools::DetectionTools(){
    DEBUG = false;
    mouthOpen = false;
    scale = 1;
    t = 0;
    fn_haar = "/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
    //fn_haar = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
    face_cascade.load(fn_haar);
    if(face_cascade.empty()){
        std::cerr << "Cannot find Haar cascaade." << std::endl;
    }

    mouth_cascade.load("/usr/local/Cellar/opencv/2.4.10.1/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml");
    if(mouth_cascade.empty()){
        std::cerr << "Cannot find Haar cascaade." << std::endl;
    }

    mouthROI.release();
}

void DetectionTools::findLipRegion(Mat& frame){

    Mat gray, smallFrame( cvRound (frame.rows/scale),
                         cvRound(frame.cols/scale), CV_8UC1 );

    /// Convert current frame to grayscale
    cvtColor(frame, gray, COLOR_BGR2GRAY );

    /// Downsize the frame
    resize( gray, smallFrame, smallFrame.size(), 0, 0, INTER_LINEAR );

    /// Find the faces in the small frame:
    vector< Rect_<int> > faces;
    face_cascade.detectMultiScale(
                                  smallFrame, ///Image
                                  faces, /// Rect to hold detected faces
                                  1.1, ///scaleFactor
                                  3, ///minNeighbors
                                  0|CASCADE_FIND_BIGGEST_OBJECT, ///Flag
                                  Size(30, 30) ///minSize
                                  );


    if(!faces.empty()){

        /// Restrict to largest face
        int facesize = faces[0].size().width;
        rectSmall = faces[0];
        for(int i=1; i<faces.size(); i++){
            if (faces[i].size().width > facesize){
                rectSmall = faces[i];                 facesize = faces[i].size().width;
            }
        }



        /// Scale the rectangle ROI back up
        Point largeTL = Point(cvRound(rectSmall.x*scale), cvRound(rectSmall.y*scale));
        Size largeSize = Size(cvRound((rectSmall.width-1)*scale),cvRound((rectSmall.height-1)*scale));
        rectLarge = Rect(largeTL,largeSize);

        //    if(!mouthOpen){
        //        mouthSize = Size(largeSize.width*0.6,
        //                         largeSize.height*0.4);
        //    }else{
        //        mouthSize = Size(largeSize.width*0.6,
        //                         largeSize.height*0.6);
        //    }

        /// Find approximate mouth ROI of region
        mouthSize = Size(largeSize.width*0.6,
                         largeSize.height*0.5);

        Point mouthTL = Point(largeTL.x+(largeSize.width*0.2),
                                largeTL.y+(largeSize.height*0.65));
        rectMouth = Rect(mouthTL,mouthSize);

        // Make sure the mouth region is not out of frame
        if ((rectMouth.y + rectMouth.height) >= frame.rows||
            (rectMouth.x + rectMouth.width) >= frame.cols) {
            return;
        }

        /// Extract mouth region
        mouthROI = frame(rectMouth);
        mouthLoc = rectMouth.tl();

        // Try mouth detection within mouth region
        vector< Rect_<int> > mouths;
        mouth_cascade.detectMultiScale(
                                       mouthROI, ///Image
                                       mouths, /// Rect to hold detected faces
                                       1.1, ///scaleFactor
                                       3, ///minNeighbors
                                       0|CASCADE_FIND_BIGGEST_OBJECT, ///Flag
                                       Size(10, 10) ///minSize
                                       );

        if(!mouths.empty()){
            rectMouthCascade = mouths[0];
            Point newTL = rectMouthCascade.tl() + rectMouth.tl(); // Top left within entire frame
            rectMouthCascade.x = newTL.x;
            rectMouthCascade.y = newTL.y;

            // Make mouth region larger
            float deltaX = rectMouthCascade.width*0.25;
            float deltaY = rectMouthCascade.height*0.3;
            rectMouthCascade.x = rectMouthCascade.x - deltaX;
            rectMouthCascade.y = rectMouthCascade.y - deltaY;
            rectMouthCascade.height = rectMouthCascade.height + 2*deltaY;
            rectMouthCascade.width = rectMouthCascade.width + 2*deltaX;

            // Mouth region is now the one found by the mouth cascade
            mouthROI = frame(rectMouthCascade);
            mouthLoc = rectMouthCascade.tl();

        } // If no mouths found



        /// Check if mouth seems to be open
        // checkMouthOpen();

    }else{ //If no faces are detected

        mouthROI.release();
        return;
    }




}

void DetectionTools::checkMouthOpen(){
    /// Following code is only working like 50% of the time when you open your mouth.

    /// Do a quick check to see if the mouth is open (this might not work for people who have beards...)
    if(mouthOpen){
        bottomMouth = mouthROI(Range(mouthSize.height*0.3, mouthSize.height*0.5),
                               Range(mouthSize.width*0.3, mouthSize.width*0.7));
    }else {
        bottomMouth = mouthROI(Range(mouthSize.height*0.5, mouthSize.height),
                               Range(mouthSize.width*0.3, mouthSize.width*0.7));
    }

    topCornerMouth =  mouthROI(Range(0, mouthSize.height*0.3),
                               Range(0, mouthSize.width*0.3));

    // Draw these regions


    Scalar meanBottom = mean(bottomMouth);
    Scalar meanCorner = mean(topCornerMouth);

    //    std::cout << "Mean bottom is now " << meanBottom[0] << std::endl;
    //    std::cout << "Mean of corner is now " << meanCorner[0] << std::endl;

    if(meanBottom[0] < 0.7*meanCorner[0]){
        mouthOpen = true;
        std::cout << "One of the mouths is open!" << std::endl;
    }
    if(meanBottom[0] > 0.8*meanCorner[0]){
        mouthOpen = false;

    }
}

