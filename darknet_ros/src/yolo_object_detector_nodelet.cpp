/*
 * yolo_object_detector_nodelet.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#include <darknet_ros/YoloObjectDetector.hpp>

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>


class DarknetRosNodelet : public nodelet::Nodelet {

  public:
    DarknetRosNodelet(){}
    ~DarknetRosNodelet(){}

  private:
    virtual void onInit() {
      
      ros::NodeHandle NodeHandle("~");
      NodeHandle = getPrivateNodeHandle();
      darknet_ros_ = new darknet_ros::YoloObjectDetector(NodeHandle);
    }

    darknet_ros::YoloObjectDetector* darknet_ros_;
};

//Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(DarknetRosNodelet, nodelet::Nodelet);
