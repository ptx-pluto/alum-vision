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
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>


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


struct marker_detector_t {

    marker_detector_t(){
        parameters = cv::aruco::DetectorParameters::create();
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    }

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


};


int main( int argc, char** args ){

    ros::init(argc,args,"alum_tag_detector_node");

    ros::NodeHandle nh;

    marker_detector_t detector;

    ros::Subscriber sub = nh.subscribe<sensor_msgs::Image>("/camera/color/image_raw",1,&marker_detector_t::image_callback,&detector);

    ros::Subscriber sub_calib = nh.subscribe<sensor_msgs::CameraInfo>("/camera/color/camera_info",1,&marker_detector_t::camera_info_callback,&detector);

    while ( ros::ok() ) { ros::spinOnce(); }

    return 0;

}