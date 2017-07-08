/*

███████╗███████╗███╗   ██╗██████╗      ██████╗ ██████╗ ██████╗ ██████╗
██╔════╝██╔════╝████╗  ██║██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔══██╗
███████╗█████╗  ██╔██╗ ██║██║  ██║    ██║     ██║   ██║██████╔╝██████╔╝
╚════██║██╔══╝  ██║╚██╗██║██║  ██║    ██║     ██║   ██║██╔══██╗██╔═══╝
███████║███████╗██║ ╚████║██████╔╝    ╚██████╗╚██████╔╝██║  ██║██║
╚══════╝╚══════╝╚═╝  ╚═══╝╚═════╝      ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝

1. installera opencv3, brew install opencv3 --with-ffmpeg (brew info opencv3 för att lista alla flaggor)
2. kompilera med: g++ $(pkg-config --cflags --libs /usr/local/Cellar/opencv3/3.2.0/lib/pkgconfig/opencv.pc) timetrax.cpp -o timetrax
3. ???
4. ./timetrax
5  profit!
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include "transform.hpp"

using namespace std;

std::mutex mtxCam;

struct score_t {
    int home;
    int away;
} score;

void displayScore(cv::Mat frame, score_t score) {
    stringstream sstr;
        sstr << score.home << " - " << score.away;
        cv::putText(frame, sstr.str(), cv::Point(30, 50),
            cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(0,255,0), 2, CV_AA);
}

int process_frames(cv::Mat *f, bool *get_frames, int *debug)
{
    // Set up score
    score.home = score.away = 0;
    int showGoalUntilFrame = 0;
    int showScoreboardPointUntilFrame = 0;
    string scoreboardPoint = "";
    // >>>> Kalman Filter
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;

    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

    cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
    cv::Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
    //cv::Mat procNoise(stateSize, 1, type)
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

    // Transition State Matrix A, set dT at each processing step
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));

    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    // <<<< Kalman Filter

    // Camera Index
    int idx = 0;

    // Camera Capture
    cv::VideoCapture cap;

    // >>>>> Setup shit
    if (!cap.open("positive.mov"))
    //if (!cap.open(0))
    {
        cout << "Webcam not connected.\n";
        return EXIT_FAILURE;
    }

    cap.set(CV_CAP_PROP_POS_MSEC, 0); // Position in milliseconds
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 854);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    cout << "\nHit 'q' to exit...\n";

    bool init = true;
    cv::Rect tableRect_init[10];
    cv::Rect tableRect;
    cv::Rect blueScoreboard_current;
    int blueScoreboard_last_goal_height = 0;
    cv::Rect blueScoreboard[60];
    cv::Rect whiteScoreboard_current;
    int whiteScoreboard_last_goal_height = 0;
    cv::Rect whiteScoreboard[60];
    char ch = 0;

    double ticks = 0;
    bool found = false;

    int notFoundCount = 0;
    cv::Mat frame;
    int frames = 0;
    int lastSeenBallFrame = 0;
    int lastGoalFrame = 0;
    int lastScoreFrame = 0;
    bool potentialGoal = false;
    bool potentialGoalHome = false;

    double lastshot = 0;
    bool shot = false;
    bool blueshot = false;
    while (get_frames)
    {
        double precTick = ticks;
        ticks = (double) cv::getTickCount();

        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

        // Get a frame
        mtxCam.lock();
        cap >> frame;
        mtxCam.unlock();
        if (frame.empty()) break;

        cv::resize(frame, frame, cv::Size(854, 480));

        cv::Mat res;
        frame.copyTo( res );
        if(score.home == 10 || score.away == 10){
            cv::cvtColor(res, res, CV_BGR2GRAY);
            res.convertTo(res, -1, 0.5, 0);
            cv::cvtColor(res, res, CV_GRAY2BGR);
            stringstream sstr;
            sstr << score.home << " - " << score.away;
            cv::putText(res, sstr.str(), cv::Point(220, 200),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 6, cv::Scalar(100,255,100), 5, CV_AA);

            string txtwin;
            cv::Scalar colorwin;
            if(score.away == 10){
                txtwin = "White wins!";
                colorwin = cv::Scalar(255,255,255);
            }else{
                txtwin = "Blue wins!";
                colorwin = cv::Scalar(255,200,50);
            }
            cv::putText(res, txtwin, cv::Point(150, 300),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 4, colorwin, 7, CV_AA);

            res.copyTo(*f);
            continue;
        }
        if (found)
        {
            // >>>> Matrix A
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;
            // <<<< Matrix A

            state = kf.predict();

            // <<<<< Detect shot
            if (abs(state.at<float>(2)) >= 250 && !shot && ticks-lastshot > 1000000000 * 1.7){
                if (state.at<float>(2) > 0){
                    //cout << "BLUE SHOT!" << endl;
                    blueshot = true;
                }else{
                    //cout << "WHITE SHOT!" << endl;
                    blueshot = false;
                }
                shot = true;
                lastshot = ticks;
            }
            if(abs(state.at<float>(2)) < 100 && shot)
                shot=false;

            if (shot && blueshot){
                cv::putText(res, "BLUE POWERSHOT!",
                        cv::Point(250, 50),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(255, 0, 0), 5);
            }
            if (shot && !blueshot){
                cv::putText(res, "WHITE POWERSHOT!",
                        cv::Point(250, 50),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cv::Scalar(255, 255, 255), 5);
            }
            // >>>>> Detect shot

            //cout << "State post:" << endl << state << endl;

            cv::Rect predRect;
            predRect.width = state.at<float>(4);
            predRect.height = state.at<float>(5);
            predRect.x = state.at<float>(0) - predRect.width / 2;
            predRect.y = state.at<float>(1) - predRect.height / 2;

            cv::Point center;
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            cv::circle(res, center, 2, CV_RGB(255,0,0), -1);

            cv::rectangle(res, predRect, CV_RGB(255,0,0), 2);

            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            //cv::putText(res, sstr.str(),
            //            cv::Point(center.x + 15, center.y + 30),
            //            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255,0,0), 2);
        }

        // >>>>> Blur it
        cv::Mat blur;
        cv::GaussianBlur(frame, blur, cv::Size(5, 5), 3.0, 3.0);
        // <<<<< Blur it

        // >>>>> convert to HSV
        cv::Mat frmHsv;
        cv::cvtColor(blur, frmHsv, CV_BGR2HSV);
        // <<<<< convert to HSV

        // >>>>> Color Thresholding (find sendball)
        cv::Mat rangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::inRange(frmHsv, cv::Scalar(0, 92, 141),
            cv::Scalar(80, 255, 255), rangeRes);
        // <<<<< Color Thresholding (find sendball)

        // >>>>> Improve result
        cv::erode(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        // <<<<< Improve result

        // >>>>> Find contours
        vector<vector<cv::Point> > contours;
        cv::findContours(rangeRes, contours, CV_RETR_EXTERNAL,
                         CV_CHAIN_APPROX_NONE);
        // <<<<< Find contours

        // >>>>> Find blue scoreboard and track it
        cv::Mat blueRangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::inRange(frmHsv, cv::Scalar(60, 100, 170), cv::Scalar(255, 255, 239), blueRangeRes);
        cv::erode(blueRangeRes, blueRangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(blueRangeRes, blueRangeRes, cv::Mat(), cv::Point(-1, -1), 2);

        vector<vector<cv::Point> > bluecontours;
        cv::findContours(blueRangeRes, bluecontours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        int blueLargestArea = 0;
        for (size_t i = 0; i < bluecontours.size(); i++) {
            cv::Rect bBox = cv::boundingRect(bluecontours[i]);
            int x = bBox.width * bBox.height;
            if (bBox.x > tableRect.x &&
                bBox.x < tableRect.x + 50 &&
                bBox.y > tableRect.y + tableRect.height/5 &&
                bBox.y < tableRect.y + tableRect.height/3
                && x > blueLargestArea) {
                    blueScoreboard[frames % 60] = bBox;
                    blueLargestArea = x;
            }
        }
        if(frames % 60 == 0){
            int totalheight = 0;
            for(int i=0; i<60; i++){
                totalheight += blueScoreboard[i].height;
            }

            int avgHeight = totalheight/60;
            //cout << "running avg height: " << avgHeight << " Last goal height:" << blueScoreboard_last_goal_height << endl;
            if (frames > 200 && (abs(blueScoreboard_current.height - avgHeight)  > 6 || blueScoreboard_last_goal_height - avgHeight >= 7) && avgHeight > 0){
                scoreboardPoint = "BLUE";
                showScoreboardPointUntilFrame = frames + 100;
                blueScoreboard_last_goal_height = avgHeight;
                if(frames - lastScoreFrame > 240)
                    score.home += 1;
            }
            blueScoreboard_current = blueScoreboard[frames % 60];
            blueScoreboard_current.height = avgHeight;
        }
        if(*debug > 1){
            cv::rectangle(res, blueScoreboard_current, cv::Scalar(255, 255, 0), 2);
        }
        if(showScoreboardPointUntilFrame > frames && scoreboardPoint == "BLUE"){
            if(frames % 2 == 0)
                cv::rectangle(res, blueScoreboard_current, cv::Scalar(0, 255, 0), 2);
            else
                cv::rectangle(res, blueScoreboard_current, cv::Scalar(255, 255, 255), 2);
        }
        // <<<<< Find blue scoreboard and track it

        // >>>>> Find white scoreboard and track it
        cv::Mat whiteRangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::inRange(frmHsv, cv::Scalar(83, 0, 208), cv::Scalar(255, 255, 255), whiteRangeRes);
        cv::erode(whiteRangeRes, whiteRangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(whiteRangeRes, whiteRangeRes, cv::Mat(), cv::Point(-1, -1), 2);

        vector<vector<cv::Point> > whitecontours;
        cv::findContours(whiteRangeRes, whitecontours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        int whiteLargestArea = 0;
        for (size_t i = 0; i < whitecontours.size(); i++) {
            cv::Rect bBox = cv::boundingRect(whitecontours[i]);
            int x = bBox.width * bBox.height;
            if (bBox.x > tableRect.x + tableRect.width - 50 &&
                bBox.x < tableRect.x + tableRect.width &&
                bBox.y > tableRect.y  &&
                bBox.y > tableRect.y + tableRect.height/3 &&
                bBox.y < tableRect.y + tableRect.height &&
                x > whiteLargestArea) {
                    whiteScoreboard[frames % 60] = bBox;
                    whiteLargestArea = x;
            }
        }
        if(frames % 60 == 0){
            int totalheight = 0;
            for(int i=0; i<60; i++){
                totalheight += whiteScoreboard[i].height;
            }

            int avgHeight = totalheight/60;
            //cout << "running avg height: " << avgHeight << " Last goal height:" << whiteScoreboard_last_goal_height << endl;
            if (frames > 200 &&
                (abs(whiteScoreboard_current.height - avgHeight)  > 6 || whiteScoreboard_last_goal_height - avgHeight >= 7) &&
                avgHeight > 0)
            {
                scoreboardPoint = "WHITE";
                showScoreboardPointUntilFrame = frames + 100;
                whiteScoreboard_last_goal_height = avgHeight;
                if(frames - lastScoreFrame > 240)
                    score.away += 1;
            }
            whiteScoreboard_current = whiteScoreboard[frames % 60];
            whiteScoreboard_current.height = avgHeight;
        }
        if(*debug > 1)
            cv::rectangle(res, whiteScoreboard_current, cv::Scalar(255, 0, 255), 2);
        if(showScoreboardPointUntilFrame > frames && scoreboardPoint == "WHITE"){
            if(frames % 2 == 0)
                cv::rectangle(res, whiteScoreboard_current, cv::Scalar(0, 255, 0), 2);
            else
                cv::rectangle(res, whiteScoreboard_current, cv::Scalar(255, 0, 255), 2);
        }
        // <<<<< Find white scoreboard and track it

        // >>>>> Find board
        if(init){
            vector<vector<cv::Point> > tableContours;
            cv::Mat tableRangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
            cv::inRange(frmHsv, cv::Scalar(0, 0, 133), cv::Scalar(179, 255, 255), tableRangeRes);
            cv::findContours(tableRangeRes, tableContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            int largestArea = 0;
            for (size_t i = 0; i < tableContours.size(); i++) {
                cv::Rect bBox = cv::boundingRect(tableContours[i]);
                int x = bBox.width * bBox.height;
                if (x > largestArea) {
                    tableRect = bBox;
                    largestArea = x;
                }
            }
            tableRect.y += (tableRect.height - tableRect.width/2)/2;
            tableRect.height = tableRect.width/2;
            tableRect_init[frames] = tableRect;
            int totalheight = 0;
            int totalwidth = 0;
            if(frames >= 10){
                for(int i=0; i<10; i++){
                    totalheight += tableRect_init[i].height;
                    totalwidth += tableRect_init[i].width;
                }
                int finalheight = totalheight/10;
                int finalwidth = totalwidth/10;

                tableRect.y += (finalheight - finalwidth/2)/2;
                tableRect.height = finalwidth/2;
                init = false;
            }
        }
        if(*debug > 2) cv::rectangle(res, tableRect, cv::Scalar(255, 0, 0), 2);
        // <<<<< Find board

        // >>>>> Goals
        cv::Scalar blueGoalColor = cv::Scalar(0, 0, 255);
        cv::Scalar whiteGoalColor = cv::Scalar(0, 0, 255);
        cv::Rect blueGoal;
        blueGoal.width = 30;
        blueGoal.height = 84;
        blueGoal.x = tableRect.x + 32;
        blueGoal.y = (tableRect.height/2) + 38;

        cv::Rect whiteGoal;
        whiteGoal.width = 30;
        whiteGoal.height = 84;
        whiteGoal.x = tableRect.x + tableRect.width - 64;
        whiteGoal.y = (tableRect.height/2) + 34;

        if(*debug > 0){
            cv::rectangle(res, blueGoal, blueGoalColor, 2);
            cv::rectangle(res, whiteGoal, whiteGoalColor, 2);
        }

        int blinkWhiteGoalUntilFrame = 0;
        int blinkBlueGoalUntilFrame = 0;
        if (potentialGoalHome && potentialGoal){
            blinkBlueGoalUntilFrame = frames + 1000;
        }
        if (!potentialGoalHome && potentialGoal){
            blinkWhiteGoalUntilFrame = frames + 1000;
        }

        if (blinkWhiteGoalUntilFrame > frames){
            if (frames % 2 == 0){
                blueGoalColor = cv::Scalar(0, 255, 0);
            }
            else{
                blueGoalColor = cv::Scalar(0, 100, 0);
            }
            cv::rectangle(res, blueGoal, blueGoalColor, 2);
        }

        if (blinkBlueGoalUntilFrame > frames){
            if (frames % 2 == 0){
                whiteGoalColor = cv::Scalar(0, 255, 0);
            }
            else{
                whiteGoalColor = cv::Scalar(0, 100, 0);
            }
            cv::rectangle(res, whiteGoal, whiteGoalColor, 2);
        }
        // <<<<< Goals

        // >>>>> Filtering
        vector<vector<cv::Point> > balls;
        vector<cv::Rect> ballsBox;
        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Rect bBox;
            bBox = cv::boundingRect(contours[i]);
            cv::rectangle(res, bBox, cv::Scalar(255, 0, 0), 2);
            //bCircle = cv::boundingCircle(contours[i]);

            //float ratio = (float) bBox.width / (float) bBox.height;
            //if (ratio > 1.0f)
            //    ratio = 1.0f / ratio;

            // Searching for a bBox "rounded" square
            if (bBox.area() >= 10)
            {
                balls.push_back(contours[i]);
                ballsBox.push_back(bBox);
            }
        }
        // <<<<< Filtering

        // >>>>> Detection result
        for (size_t i = 0; i < balls.size(); i++)
        {
            //cv::drawContours(res, balls, i, CV_RGB(20,150,20), 1);
            cv::Scalar color = cv::Scalar(0, 255, 0);

            if (shot && blueshot){
                color = cv::Scalar(255, 0, 0);
            }
            if (shot && !blueshot){
                color = cv::Scalar(255, 255, 255);
            }
            cv::rectangle(res, ballsBox[i], color, 2);

            cv::Point center;
            center.x = ballsBox[i].x + ballsBox[i].width / 2;
            center.y = ballsBox[i].y + ballsBox[i].height / 2;
            cv::circle(res, center, 2, CV_RGB(20,150,20), -1);

            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            //cv::putText(res, sstr.str(),
            //         cv::Point(center.x + 15, center.y - 15),
            //         cv::FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(0,255,0), 2);

        }
        if (balls.size() > 0) lastSeenBallFrame = frames;

        // <<<<< Detection result

        // >>>>> Kalman Update
        if (balls.size() == 0)
        {
            notFoundCount++;
            //cout << "notFoundCount:" << notFoundCount << endl;
            if( notFoundCount >= 100 )
            {
                found = false;
            }
            /*else
                kf.statePost = state;*/
        }
        else
        {
            notFoundCount = 0;

            meas.at<float>(0) = ballsBox[0].x + ballsBox[0].width / 2;
            meas.at<float>(1) = ballsBox[0].y + ballsBox[0].height / 2;
            meas.at<float>(2) = (float)ballsBox[0].width;
            meas.at<float>(3) = (float)ballsBox[0].height;

            if (!found) // First detection!
            {
                // >>>> Init
                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(7) = 1; // px
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1; // px
                kf.errorCovPre.at<float>(35) = 1; // px

                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = 0;
                state.at<float>(3) = 0;
                state.at<float>(4) = meas.at<float>(2);
                state.at<float>(5) = meas.at<float>(3);
                // <<<< Init

                kf.statePost = state;

                found = true;
            }
            else
                kf.correct(meas); // kalman correction

            //cout << "measure matrix:" << endl << meas << endl;
        }
        // <<<<< kalman update

        // Check if ball is sent to goal
        long now = time(0);


        // If the ball is inside the goal rect, there is a potential goal.

        if (blueGoal.contains(cv::Point(state.at<float>(0), state.at<float>(1)))) {
            //cout << "GOAL ???" << endl;
            potentialGoalHome = false; // HOME POINT = SCORE ON THE RIGHT GOAL
            potentialGoal = true;
            lastGoalFrame = frames;
        }
        else if (whiteGoal.contains(cv::Point(state.at<float>(0), state.at<float>(1)))) {
            //cout << "GOAL ???" << endl;
            potentialGoalHome = true; // HOME POINT = SCORE ON THE RIGHT GOAL
            potentialGoal = true;
            lastGoalFrame = frames;
        }

        // When there is a potential goal, only count it as a real goal if we don't
        // see the ball 30 frames later.
        if (frames > (lastSeenBallFrame + 30)) { // 30 frames since last seen ball
            if (potentialGoal) {
                showGoalUntilFrame = frames + 100;
                //cout << "REALLY GOAL!!!" << endl;
                potentialGoal = false;
                lastGoalFrame = 0;

                if (potentialGoalHome) {
                    score.home += 1;
                } else {
                    score.away += 1;
                }
                lastScoreFrame = frames;
            }
        } else {
            // If we see a the ball 5 frames later, it probably wasn't a goal...
            if (lastSeenBallFrame > (lastGoalFrame + 5))
            {
                potentialGoal = false;
                lastGoalFrame = 0;
            }
        }

        if (showScoreboardPointUntilFrame > frames){
            cv::putText(res, "GOOD JOB!",
                        cv::Point(280, 450),
                        cv::FONT_HERSHEY_TRIPLEX, 2, cv::Scalar(100, 255, 0), 3);

        }
        if (showGoalUntilFrame > frames){
            cv::putText(res, "GOOOAL!!! Take a point!",
                        cv::Point(10, 200),
                        cv::FONT_HERSHEY_TRIPLEX, 2, cv::Scalar(255, 0, 255), 3);

        }
        cv::circle(res, cv::Point(144, 70), 2, CV_RGB(255,0,0), -1);
        cv::circle(res, cv::Point(156, 400), 2, CV_RGB(255,0,0), -1);
        cv::circle(res, cv::Point(763, 395), 2, CV_RGB(255,0,0), -1);
        cv::circle(res, cv::Point(768, 65), 2, CV_RGB(255,0,0), -1);
        cv::Mat rect = cv::Mat::zeros(frame.size(), CV_8UC1);

        vector<cv::Point> points;
        points.push_back(cv::Point(144, 70));
        points.push_back(cv::Point(768, 65));
        points.push_back(cv::Point(763, 395));
        points.push_back(cv::Point(156, 400));

        cv::Mat warped = Transform::four_point_transform(res, points);
        displayScore(warped, score);
        warped.copyTo(*f);
        frames++;
    }

    cout << "Video ended." << endl;
    return EXIT_SUCCESS;
}

int main(){
    bool get_frames = true;
    cv::Mat frame;
    int debug = 0;
    char ch;
    thread t(process_frames, &frame, &get_frames, &debug);

    while(ch != 'q' && ch != 'Q'){
        if (!frame.empty()) {
            cv::imshow("Tracking", frame);
        }
        ch = cv::waitKey(1);
        if(ch == 'd'){
            debug += 1;
            if(debug > 3)
                debug = 0;
        }
    }
    t.join();
    get_frames = false;
}
