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

using namespace std;

std::mutex mtxCam;

struct score_t {
    int home;
    int away;
} score;

cv::Rect combineBoxes(cv::Rect box1, cv::Rect box2) {
    int x = min(box1.x, box2.x);
    int width = max(box1.width, box2.width);
    int y = min(box1.y, box2.y);
    int maxY = max(box1.height + box1.y, box2.height + box2.y);
    int height = maxY - y;
    cv::Rect newBox(x, y, width, height);
    return newBox;
}
void displayScore(cv::Mat frame, score_t score) {
    stringstream sstr;
        sstr << score.home << " - " << score.away;
        cv::putText(frame, sstr.str(), cv::Point(10, 20),
            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,255,0), 1, CV_AA);
}


int process_frames(cv::Mat *f, bool *get_frames)
{
    // Set up score
    score.home = score.away = 0;

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
    {
        cout << "Webcam not connected.\n" << "Please verify\n";
        return EXIT_FAILURE;
    }

    cap.set(CV_CAP_PROP_POS_MSEC, 0); // Position in milliseconds
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 854);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    cout << "\nHit 'q' to exit...\n";

    char ch = 0;

    double ticks = 0;
    bool found = false;

    int notFoundCount = 0;
    cv::Mat frame;
    int frames = 0;
    int lastSeenBallFrame = 0;
    int lastGoalFrame = 0;
    bool potentialGoal = false;
    bool potentialGoalHome = false;
    // >>>>> Main loop

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
                    cout << "BLUE SHOT!" << endl;
                    blueshot = true;
                }else{
                    cout << "WHITE SHOT!" << endl;
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
            cv::putText(res, sstr.str(),
                        cv::Point(center.x + 15, center.y + 30),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255,0,0), 2);
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
        //((0, 92, 141), (80, 255, 255))

        cv::Mat blackRangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::inRange(frmHsv, cv::Scalar(0, 0, 0),
            cv::Scalar(255, 255, 37), blackRangeRes);

        cv::Mat bluePlayerRangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::inRange(frmHsv, cv::Scalar(100, 70, 100),
            cv::Scalar(150, 255, 255), bluePlayerRangeRes);
        // <<<<< Color Thresholding (find sendball)

        // >>>>> Improve result
        cv::erode(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        // <<<<< Improve result

        //cv::imshow("Threshold", rangeRes);

        // >>>>> Find contours
        vector<vector<cv::Point> > contours;
        cv::findContours(rangeRes, contours, CV_RETR_EXTERNAL,
                         CV_CHAIN_APPROX_NONE);
        // <<<<< Find contours

        vector<vector<cv::Point> > tableContours;
        cv::Mat tableRangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::inRange(frmHsv, cv::Scalar(0, 0, 133), cv::Scalar(179, 255, 255), tableRangeRes);
        cv::findContours(tableRangeRes, tableContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        vector<cv::Point> largestContour;
        cv::Rect largest;
        int largestArea = 0;
        for (size_t i = 0; i < tableContours.size(); i++) {
            cv::Rect bBox = cv::boundingRect(tableContours[i]);
            int x = bBox.width * bBox.height;
            if (x > largestArea) {
                largest = bBox;
                largestContour = tableContours[i];
                largestArea = x;
            }
        }

        // >>>>> Find goals
        // Goals are black
        vector<vector<cv::Point> > blackContours;
        cv::findContours(blackRangeRes, blackContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        vector<vector<cv::Point> > bluePlayerContours;
        cv::findContours(bluePlayerRangeRes, bluePlayerContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

        vector<cv::Rect> potentialGoals;
        for (size_t i = 0; i < blackContours.size(); i++)
        {
            cv::Rect bBox = cv::boundingRect(blackContours[i]);
            if (bBox.width < 10 and bBox.width > 2) {
                potentialGoals.push_back(bBox);
            }
        }

        // Sometimes the goalie is in the way of the goal, giving us two
        // rectangles, so they need to be merged.
        vector<cv::Rect> actualGoals;
        for (size_t i = 0; i < potentialGoals.size(); i++) {
            cv::Rect box = potentialGoals[i];
            cv::Rect mergeBox;
            bool sameGoal = false;
            for (size_t x = 0; x < potentialGoals.size(); x++) {
                if (i != x) { // Don't match the same
                    int threshold = 100;
                    int threshold_x = 15;
                    if (   potentialGoals[i].y > potentialGoals[x].y - threshold
                        && potentialGoals[i].y < potentialGoals[x].y + threshold
                        && potentialGoals[i].x > potentialGoals[x].x - threshold_x
                        && potentialGoals[i].x < potentialGoals[x].x + threshold_x)
                    {
                        mergeBox = potentialGoals[x];
                        sameGoal = true;
                        break;
                    }
                }

            }
            if (sameGoal) box = combineBoxes(box, mergeBox);
            if (box.height > 70 && box.width < 15) { // Have the rectangles the size of a goal?
                // Make the goal rect bigger to have a more accurate hitbox.
                box.x -= (box.x < 427) ? 60 : -10;
                box.width += 50;
                box.y -= 20;
                box.height += 40;

                actualGoals.push_back(box);
            }
        }
        // <<<<< Find goals

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

        cv::Rect goal(763,195, 45,100); // x, y, width, height
        cv::Rect goal2(85,195, 45,100);

        // >>>>> Detection result
        for (size_t i = 0; i < balls.size(); i++)
        {
            cv::drawContours(res, balls, i, CV_RGB(20,150,20), 1);
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
        for (size_t i = 0; i < actualGoals.size(); i++) {
            cv::rectangle(res, actualGoals[i], CV_RGB(255,0,0), 2);
            if (actualGoals[i].contains(cv::Point(state.at<float>(0), state.at<float>(1)))) {
                cout << "GOAL ???" << endl;
                potentialGoalHome = actualGoals[i].x > 427; // HOME POINT = SCORE ON THE RIGHT GOAL
                potentialGoal = true;
                lastGoalFrame = frames;
            }
        }

        // When there is a potential goal, only count it as a real goal if we don't
        // see the ball 30 frames later.
        if (frames > (lastSeenBallFrame + 30)) { // 30 frames since last seen ball
            if (potentialGoal) {
                cout << "REALLY GOAL!!!" << endl;
                potentialGoal = false;
                lastGoalFrame = 0;

                if (potentialGoalHome) {
                    score.home += 1;
                } else {
                    score.away += 1;
                }
            }
        } else {
            // If we see a the ball 5 frames later, it probably wasn't a goal...
            if (lastSeenBallFrame > (lastGoalFrame + 5))
            {
                potentialGoal = false;
                lastGoalFrame = 0;
            }
        }

        cv::rectangle(
            res,
            cv::Point(150, 70),
            cv::Point(760, 400),
            cv::Scalar(255, 255, 255)
        );
        displayScore(res, score);
        res.copyTo(*f);
        frames++;
    }

    cout << "Video ended." << endl;
    return EXIT_SUCCESS;
}

int main(){
    bool get_frames = true;
    cv::Mat frame;
    char ch;
    thread t(process_frames, &frame, &get_frames);

    while(ch != 'q' && ch != 'Q'){
        if (!frame.empty()) {
            cv::imshow("Tracking", frame);
        }
        ch = cv::waitKey(1);
    }
    t.join();
    get_frames = false;
}
