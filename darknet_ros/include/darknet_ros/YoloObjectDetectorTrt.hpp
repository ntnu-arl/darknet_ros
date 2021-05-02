/*
 * YoloObjectDetector.h
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#pragma once

// c++
#include <math.h>
#include <string>
#include <vector>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <pthread.h>
#include <memory>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <chrono>
#include <cstddef>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

// ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Int8.h>
#include <actionlib/server/simple_action_server.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>

#include <Eigen/Core>

// OpenCv
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/cvdef.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

//Xtensor
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xrepeat.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xslice.hpp>

// darknet_ros_msgs
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/CheckForObjectsAction.h>

// Darknet.
#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

extern "C" {
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "darknet_ros/image_interface.h"
#include <sys/time.h>
}

extern "C" image mat_to_image(cv::Mat src);
extern "C" int show_image_cv(image im, const char* name, int ms);
extern "C" cv::Mat image_to_mat(image im);

using namespace cv;
using namespace dnn;

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "\n";
        }
    }
} gLoggerYolo;

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;



struct BufferMSSIM                                     // Optimized CUDA versions
{   // Data allocations are very expensive on CUDA. Use a buffer to solve: allocate once reuse later.
    cuda::GpuMat gI1, gI2, gs, t1,t2;

    cuda::GpuMat I1_2, I2_2, I1_I2;
    std::vector<cuda::GpuMat> vI1, vI2;

    cuda::GpuMat mu1, mu2;
    cuda::GpuMat mu1_2, mu2_2, mu1_mu2;

    cuda::GpuMat sigma1_2, sigma2_2, sigma12;
    cuda::GpuMat t3;

    cuda::GpuMat ssim_map;

    cuda::GpuMat buf;
};

namespace darknet_ros {

//! Bounding box of the detected object.
typedef struct
{
  float x, y, w, h, prob;
  int num, Class;
} RosBox_;

typedef struct
{
  cv::Mat image;
  std_msgs::Header header;
} MatImageWithHeader_;

class YoloObjectDetectorTrt
{
 public:
  /*!
   * Constructor.
   */
  explicit YoloObjectDetectorTrt(ros::NodeHandle nh);

  /*!
   * Destructor.
   */
  ~YoloObjectDetectorTrt();

 private:
  /*!
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  /*!
   * Initialize the ROS connections.
   */
  void init();

  /*!
   * Callback of camera.
   * @param[in] msg image pointer.
   */
  void cameraCallback(const sensor_msgs::ImageConstPtr& msg);

  /*!
   * Check for objects action goal callback.
   */
  void checkForObjectsActionGoalCB();

  /*!
   * Check for objects action preempt callback.
   */
  void checkForObjectsActionPreemptCB();

  /*!
   * Check if a preempt for the check for objects action has been requested.
   * @return false if preempt has been requested or inactive.
   */
  bool isCheckingForObjects() const;

  /*!
   * Publishes the detection image.
   * @return true if successful.
   */
  bool publishDetectionImage(const cv::Mat& detectionImage);
    
  //! Typedefs.
  typedef actionlib::SimpleActionServer<darknet_ros_msgs::CheckForObjectsAction> CheckForObjectsActionServer;
  typedef std::shared_ptr<CheckForObjectsActionServer> CheckForObjectsActionServerPtr;

  //! ROS node handle.
  ros::NodeHandle nodeHandle_;

  //! Class labels.
  int numClasses_;
  std::vector<std::string> classLabels_;

  //! Check for objects action server.
  CheckForObjectsActionServerPtr checkForObjectsActionServer_;

  //! Advertise and subscribe to image topics.
  image_transport::ImageTransport imageTransport_;

  //! ROS subscriber and publisher.
  image_transport::Subscriber imageSubscriber_;
  ros::Publisher objectPublisher_;
  ros::Publisher boundingBoxesPublisher_;

  //! Detected objects.
  std::vector<std::vector<RosBox_> > rosBoxes_;
  std::vector<int> rosBoxCounter_;
  darknet_ros_msgs::BoundingBoxes boundingBoxesResults_;

  //! Camera related parameters.
  int frameWidth_;
  int frameHeight_;

  //! Publisher of the bounding box image.
  image_transport::Publisher detectionImagePublisher_;

  // Yolo running on thread.
  std::thread yoloThread_;
  
  /*!
   * Instantiates a sigmoid function
   */  
  static double sigmoid_f(float val);
  
  /*!
   * Instantiates an exponential function
   */  
  static double exponential_f(float val);
  
  /*!
   * Not sure what this does 
   */     
  void reshapeOutput(xt::xarray<float>& cpu_reshape);
  
  /*!
   * Makes sure the bounding boxes don’t go outside of the image boundary 
   */  
  void rectifyBox(cv::Rect& bbox);
  
  /*!
   * Does some XTensor calculations on the output of the TensorRT inference
   */  
  void interpretOutput(xt::xarray<float> &cpu_reshape, xt::xarray<float> &anchors_tensor, xt::xarray<float> &boxes, xt::xarray<float> &box_class_scores, xt::xarray<int> &box_classes);
  
  /*!
   * Loads TensorRT engine
   */     
  void loadEngine(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                     TRTUniquePtr<nvinfer1::IExecutionContext>& context);
                     
  /*!
   * Not sure what this does 
   */                     
  size_t getSizeByDim(const nvinfer1::Dims& dims);
  
  /*!
   *  Preprocessing of the image for TensorRT inference. Uploads image to GPU
   */                     
  void preprocessImage(cv::Mat frame, float* gpu_input, const nvinfer1::Dims& dims);
  
  /*!
   * Post precessing of the image after TensorRT inference.
   */  
  void postprocessResults(std::vector<void*> gpu_output, const std::vector<nvinfer1::Dims> &dims, int batch_size, std::vector<std::vector<int>> &yolo_masks, std::vector<std::vector<float>> &yolo_anchors, const cv::Size orig_dims, float threshold, float nms_threshold, std::vector<cv::Rect> &boxes_return, std::vector<int> &classes_return, std::vector<std::pair<float,int>> &scores_return);

  // Darknet.
  char **demoNames_;
  image **demoAlphabet_;
  int demoClasses_;

  network *net_;
  std_msgs::Header headerBuff_[3];
  image buff_[3];
  image buffLetter_[3];
  int buffId_[3];
  int buffRdInd_ = 0;
  int buffWrtInd_ = 0;
  cv::Mat mat_;
  float fps_ = 0;
  float demoThresh_ = 0;
  float demoHier_ = .5;
  int running_ = 0;

  int demoDelay_ = 0;
  int demoFrame_ = 3;
  float **predictions_;
  int demoIndex_ = 0;
  int demoDone_ = 0;
  float *lastAvg2_;
  float *lastAvg_;
  float *avg_;
  int demoTotal_ = 0;
  double demoTime_;
  bool initDone_ = false;

  RosBox_ *roiBoxes_;
  bool viewImage_;
  bool enableConsoleOutput_;
  int waitKeyDelay_;
  int fullScreen_;
  char *demoPrefix_;

  std_msgs::Header imageHeader_;
  cv::Mat camImageCopy_;
  boost::shared_mutex mutexImageCallback_;

  bool imageStatus_ = false;
  boost::shared_mutex mutexImageStatus_;

  bool isNodeRunning_ = true;
  boost::shared_mutex mutexNodeStatus_;

  int actionId_;
  boost::shared_mutex mutexActionStatus_;

  // double getWallTime();

  void *detectInThread();

  void *fetchInThread();

  void *displayInThread(void *ptr);

  void *displayLoop(void *ptr);

  void *detectLoop(void *ptr);

  void yolo();

  MatImageWithHeader_ getMatImageWithHeader();

  bool getImageStatus(void);

  bool isNodeRunning(void);

  void *publishInThread();

  void writeImageToBuffer();


  //TRT model stuff
  int batch_size_ = 1;
  TRTUniquePtr<nvinfer1::ICudaEngine> engine_{nullptr};
  TRTUniquePtr<nvinfer1::IExecutionContext> context_{nullptr};
  
  // get sizes of input and output and allocate memory required for input data and for output
  data
  std::vector<nvinfer1::Dims> input_dims_; // we expect only one input
  std::vector<nvinfer1::Dims> output_dims_; // and one output
  std::vector<void*> buffers_;
  std::vector<std::vector<int>> yolo_masks_;
  std::vector<std::vector<float>> yolo_anchors_;
  cv::Size original_dims_;
  float yolo_threshold_;
  double nms_threshold_ = 0.2;
  int yolo_res;
    
  // general model stuff
  std::vector<std::string> labels;
  std::string weightsPath;
  std::string configPath;
  std::string labelsPath;
  std::string modelPath;
};

} /* namespace darknet_ros*/



  
  
  
