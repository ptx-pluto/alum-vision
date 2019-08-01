//
// Created by sheep on 19-7-23.
//

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>


#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/Point.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>

geometry_msgs::Point eigen2ros( Eigen::Vector3d const & src ){
    geometry_msgs::Point rst;
    rst.x = src(0);
    rst.y = src(1);
    rst.z = src(2);
    return rst;
}

Eigen::Quaterniond cv2eigen( cv::Vec3d const & rvec ){

    cv::Mat R; cv::Rodrigues(rvec,R);

    Eigen::Matrix3d RR;
    for ( int i=0; i<3; i++) {
        for ( int j=0; j<3; j++) {
            RR(i,j) = R.at<double>(i,j);
        }
    }

    return Eigen::Quaterniond(RR);

}


Eigen::Vector3d calc_translation( cv::Vec3d const & rvec, cv::Vec3d const & tvec ){

    Eigen::Quaterniond quat = cv2eigen(rvec);
    Eigen::Vector3d trans( tvec(0), tvec(1), tvec(2) );

    return -1 * quat.inverse().toRotationMatrix() * trans;

}


tf::Quaternion rvec2tf( cv::Vec3d const & rvec ){

    cv::Mat R; cv::Rodrigues(rvec,R);

    Eigen::Matrix3d RR;
    for ( int i=0; i<3; i++) {
        for ( int j=0; j<3; j++) {
            RR(i,j) = R.at<double>(i,j);
        }
    }

    Eigen::Quaterniond quat(RR);

    return { quat.x(), quat.y(), quat.z(), quat.w() };

}

struct hole_marker_t {

    hole_marker_t(){
        set_dict(cv::aruco::DICT_4X4_50);
    }

    cv::Ptr<cv::aruco::Dictionary> dictionary;

    std::vector<cv::Point2f> pts;

    std::vector<int> ids;

    cv::Point2f hole_pt;

    float hole_size = 18.2;

    float tag_size = 10;

    void add_tag( cv::Point2f const & pt, int tag_id ){
        pts.push_back(pt);
        ids.push_back(tag_id);
    }

    void set_hole( cv::Point2f const & pt, float size ){
        hole_pt = pt;
        hole_size = size;
    }

    void set_dict( cv::aruco::PREDEFINED_DICTIONARY_NAME name = cv::aruco::DICT_4X4_50 ){
        dictionary = cv::aruco::getPredefinedDictionary(name);
    }

    cv::Ptr<cv::aruco::Board> make_board(){

        std::vector<std::vector<cv::Point3f> > objPoints;

        float center_x = hole_pt.x + hole_size/2;
        float center_y = hole_pt.y + hole_size/2;

        for ( size_t i=0; i<ids.size(); i++ ) {

            float x = pts.at(i).x - center_x;
            float y = pts.at(i).y - center_y;

            y = -y;

            objPoints.emplace_back();
            objPoints.back().emplace_back(x,y,0);
            objPoints.back().emplace_back(x+tag_size,y,0);
            objPoints.back().emplace_back(x+tag_size,y-tag_size,0);
            objPoints.back().emplace_back(x,y-tag_size,0);

        }

        return cv::aruco::Board::create(objPoints,dictionary,ids);

    }

    static hole_marker_t create_marker_1(){

        hole_marker_t marker;

        marker.set_hole({47.6,104.2},18.2);

        marker.add_tag({51.7,91.3},0);
        marker.add_tag({67.8,91.5},1);
        marker.add_tag({34,91.5},2);
        marker.add_tag({68.6,108.3},3);
        marker.add_tag({33.8,108.3},4);

        return marker;

    }


};




struct marker_detector_t {

    marker_detector_t(){
        parameters = cv::aruco::DetectorParameters::create();
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

        pub_pos = nh.advertise<geometry_msgs::Point>("/perception/gun/eye",10);
        pub_err = nh.advertise<geometry_msgs::Point>("/perception/gun/err",10);

    }

    ros::NodeHandle nh;

    cv::Ptr<cv::aruco::DetectorParameters> parameters;

    cv::Ptr<cv::aruco::Dictionary> dictionary;

    bool calibrated = false;

    sensor_msgs::CameraInfo camera_info;

    cv::Mat K;

    std::vector<double> dist_coeff;

    tf::TransformBroadcaster br;

    bool visualize = true;

    bool send_tf = true;

    float marker_length = 0.15;

    // ros related

    ros::Publisher pub_pos;

    ros::Publisher pub_err;

    void camera_info_callback( sensor_msgs::CameraInfoConstPtr const & msg ){

        if (!calibrated) {

            camera_info = *msg;

            K = cv::Mat(3,3,CV_64F,camera_info.K.elems).clone();

            dist_coeff = { 0, 0, 0, 0, 0 };

            std::cout << K << std::endl;

            calibrated = true;

        }

    }

    void image_callback( sensor_msgs::ImageConstPtr const & msg ){

        if ( !calibrated ) { return; }

        cv_bridge::CvImageConstPtr cv_ptr;
        cv_ptr = cv_bridge::toCvShare(msg,sensor_msgs::image_encodings::BGR8);

        //mpSLAM->TrackMonocular(cv_ptr->image,cv_ptr->header.stamp.toSec());

        cv::Mat mat = cv_ptr->image;

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

        cv::aruco::detectMarkers(mat,dictionary,markerCorners,markerIds,parameters,rejectedCandidates);

        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(markerCorners,marker_length,K,dist_coeff,rvecs,tvecs);

        bool detected = !markerIds.empty();

        if ( visualize ) {

            cv::Mat outputImage = mat;

            if ( detected ) {

                outputImage = mat.clone();

                cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);

                for ( size_t i=0; i<markerIds.size(); i++ ) {

                    cv::aruco::drawAxis(outputImage, K, dist_coeff, rvecs.at(i), tvecs.at(i), marker_length);

                }

            }

            cv::imshow("test",outputImage);
            cv::waitKey(1);

        }

        if ( send_tf ) {

            for ( size_t i=0; i<markerIds.size(); i++ ) {

                std::stringstream ss; ss << "marker_"<< markerIds.at(i);

                cv::Vec3d vr = rvecs.at(i);
                cv::Vec3d vt = tvecs.at(i);

                tf::Transform transform;
                transform.setOrigin(tf::Vector3(vt(0),vt(1),vt(2)));
                transform.setRotation(rvec2tf(vr));
                br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_rgb_optical_frame", ss.str()));

            }

        }

    }


    void image_callback_board( sensor_msgs::ImageConstPtr const & msg ){

        if ( !calibrated ) { return; }

        cv_bridge::CvImageConstPtr cv_ptr;
        cv_ptr = cv_bridge::toCvShare(msg,sensor_msgs::image_encodings::BGR8);

        //mpSLAM->TrackMonocular(cv_ptr->image,cv_ptr->header.stamp.toSec());

        cv::Mat mat = cv_ptr->image;

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

        cv::aruco::detectMarkers(mat,dictionary,markerCorners,markerIds,parameters,rejectedCandidates);

        auto marker = hole_marker_t::create_marker_1();

        auto board = marker.make_board();

        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(markerCorners,marker_length,K,dist_coeff,rvecs,tvecs);

        cv::Vec3d rvec, tvec;
        cv::aruco::estimatePoseBoard(markerCorners,markerIds,board,K,dist_coeff,rvec,tvec);

        bool detected = !markerIds.empty();

        if ( detected ) {
            Eigen::Vector3d T = calc_translation(rvec,tvec);
            pub_pos.publish(eigen2ros(T));

            Eigen::Vector3d tgt(0,89.5,46);

            Eigen::Vector3d err = T-tgt;

            pub_err.publish(eigen2ros(err));

        }

        if ( visualize ) {

            cv::Mat outputImage = mat;

            if ( detected ) {

                outputImage = mat.clone();

                cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);

                cv::aruco::drawAxis(outputImage, K, dist_coeff, rvec, tvec, marker.tag_size);

            }

            cv::imshow("test",outputImage);
            cv::waitKey(1);

        }

    }


};


int main( int argc, char** args ){

    ros::init(argc,args,"alum_tag_detector_node");

    ros::NodeHandle nh;
//
//    hole_marker_t board = hole_marker_t::create_marker_1();
//
//    auto b = board.make_board();
//
//    cv::Mat img_board;
//    cv::aruco::drawPlanarBoard(b,{800,600},img_board);
//    cv::imshow("test",img_board);
//    cv::waitKey(0);
//

    marker_detector_t detector;

    ros::Subscriber sub = nh.subscribe<sensor_msgs::Image>("/camera/color/image_raw",1,&marker_detector_t::image_callback_board,&detector);

    ros::Subscriber sub_calib = nh.subscribe<sensor_msgs::CameraInfo>("/camera/color/camera_info",1,&marker_detector_t::camera_info_callback,&detector);

    while ( ros::ok() ) { ros::spinOnce(); }

    return 0;

}