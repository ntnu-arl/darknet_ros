/*
 * YoloObjectDetectorTrt.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

// yolo object detector
#include "darknet_ros/YoloObjectDetectorTrt.hpp"

// Check for xServer
#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_ros {

char *cfg;
char *engine;
char *data;
char **detectionNames;

YoloObjectDetectorTrt::YoloObjectDetectorTrt(ros::NodeHandle nh)
    : nodeHandle_(nh),
      imageTransport_(nodeHandle_),
      numClasses_(0),
      classLabels_(0),
      rosBoxes_(0),
      rosBoxCounter_(0)
{
  ROS_INFO("[YoloObjectDetectorTrt] Node started.");

  // Read parameters from config file.
  if (!readParameters()) {
    ros::requestShutdown();
  }

  init();
}

YoloObjectDetectorTrt::~YoloObjectDetectorTrt()
{
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
  
    for (void* buf : buffers_)
  {
      cudaFree(buf);
  }
}

bool YoloObjectDetectorTrt::readParameters()
{
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

  // Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[YoloObjectDetectorTrt] Xserver is running.");
  } else {
    ROS_INFO("[YoloObjectDetectorTrt] Xserver is not running.");
    viewImage_ = false;
  }

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_,
                    std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

  return true;
}

void YoloObjectDetectorTrt::init()
{
  ROS_INFO("[YoloObjectDetectorTrt] init().");

  // Initialize deep network of darknet.
  std::string configPath;
  std::string dataPath;
  std::string enginePath;
  std::string engineModel;
  std::string configModel;

  // Threshold of object detection.
  nodeHandle_.param("tensorrt_model/threshold/value", yolo_threshold_, (float) 0.3);

  // Path to TensorRT engine file
  nodeHandle_.param("tensorrt_model/engine_file/name", engineModel,
                      std::string("yolo_subt.trt"));
    nodeHandle_.param("engine_path", enginePath, std::string("/default"));
    enginePath += "/" + engineModel;
    engine = new char[enginePath.length() + 1];
    strcpy(engine, enginePath.c_str());

  // Path to config file.
  nodeHandle_.param("tensorrt_model/config_file/name", configModel, std::string("yolo_subt.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  cfg = new char[configPath.length() + 1];
  strcpy(cfg, configPath.c_str());

  // Path to data folder.
  dataPath = darknetFilePath_;
  dataPath += "/data";
  data = new char[dataPath.length() + 1];
  strcpy(data, dataPath.c_str());

  // Get classes.
  detectionNames = (char**) realloc((void*) detectionNames, (numClasses_ + 1) * sizeof(char*));
  for (int i = 0; i < numClasses_; i++) {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  // Get YOLO resolution
  nodeHandle_.param("tensorrt_model/yolo_resolution/value", yolo_res, (int) 608);

  // Load network
  yoloThread_ = std::thread(&YoloObjectDetectorTrt::yolo, this);

  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;

  nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName,
                    std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName,
                    std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName,
                    std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

  imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize,
                                               &YoloObjectDetectorTrt::cameraCallback, this);
  objectPublisher_ = nodeHandle_.advertise<std_msgs::Int8>(objectDetectorTopicName,
                                                           objectDetectorQueueSize,
                                                           objectDetectorLatch);
  boundingBoxesPublisher_ = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(
      boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
  detectionImagePublisher_ = imageTransport_.advertise(detectionImageTopicName,
                                                                       detectionImageQueueSize,
                                                                       detectionImageLatch);

  // Action servers.
  std::string checkForObjectsActionName;
  nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName,
                    std::string("check_for_objects"));
  checkForObjectsActionServer_.reset(
      new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
  checkForObjectsActionServer_->registerGoalCallback(
      boost::bind(&YoloObjectDetectorTrt::checkForObjectsActionGoalCB, this));
  checkForObjectsActionServer_->registerPreemptCallback(
      boost::bind(&YoloObjectDetectorTrt::checkForObjectsActionPreemptCB, this));
  checkForObjectsActionServer_->start();

  //start TRT stuff
   loadEngine(enginePath,engine_,context_);
   ROS_INFO_STREAM("detectionJustYolo::loaded TRT model");
   // buffers for input and output data
   buffers_.resize(engine_->getNbBindings());
   for (size_t i = 0; i < engine_->getNbBindings(); ++i)
   {
       auto binding_size = getSizeByDim(engine_->getBindingDimensions(i)) * batch_size_ * sizeof(float);
       cudaMalloc(&buffers_[i], binding_size);

       if (engine_->bindingIsInput(i))
       {
           input_dims_.emplace_back(engine_->getBindingDimensions(i));
       }
       else
       {
           output_dims_.emplace_back(engine_->getBindingDimensions(i));
       }
   }
   if (input_dims_.empty() || output_dims_.empty())
   {
       ROS_ERROR_STREAM("detectionJustYolo::Expect at least one input and one output for network\n");
   }

  yolo_masks_.push_back(std::vector<int>());
     yolo_masks_.push_back(std::vector<int>());
     yolo_masks_.push_back(std::vector<int>());
     yolo_masks_[0].push_back(6);
     yolo_masks_[0].push_back(7);
     yolo_masks_[0].push_back(8);
     yolo_masks_[1].push_back(3);
     yolo_masks_[1].push_back(4);
     yolo_masks_[1].push_back(5);
     yolo_masks_[2].push_back(0);
     yolo_masks_[2].push_back(1);
     yolo_masks_[2].push_back(2);

     yolo_anchors_.push_back(std::vector<float>());
     yolo_anchors_[0].push_back(116.0);
     yolo_anchors_[0].push_back(90.0);
     yolo_anchors_[0].push_back(156.0);
     yolo_anchors_[0].push_back(198.0);
     yolo_anchors_[0].push_back(373.0);
     yolo_anchors_[0].push_back(326.0);
     yolo_anchors_.push_back(std::vector<float>());
     yolo_anchors_[1].push_back(30.0);
     yolo_anchors_[1].push_back(61.0);
     yolo_anchors_[1].push_back(62.0);
     yolo_anchors_[1].push_back(45.0);
     yolo_anchors_[1].push_back(59.0);
     yolo_anchors_[1].push_back(119.0);
     yolo_anchors_.push_back(std::vector<float>());
     yolo_anchors_[2].push_back(10.0);
     yolo_anchors_[2].push_back(13.0);
     yolo_anchors_[2].push_back(16.0);
     yolo_anchors_[2].push_back(30.0);
     yolo_anchors_[2].push_back(33.0);
     yolo_anchors_[2].push_back(23.0);
}

void YoloObjectDetectorTrt::cameraCallback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_DEBUG("[YoloObjectDetectorTrt] USB image received.");

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      imageHeader_ = msg->header;
      camImageCopy_ = cam_image->image.clone();
      writeImageToBuffer();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetectorTrt::checkForObjectsActionGoalCB()
{
  ROS_DEBUG("[YoloObjectDetectorTrt] Start check for objects action.");

  boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal> imageActionPtr =
      checkForObjectsActionServer_->acceptNewGoal();
  sensor_msgs::Image imageAction = imageActionPtr->image;

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
      actionId_ = imageActionPtr->id;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetectorTrt::checkForObjectsActionPreemptCB()
{
  ROS_DEBUG("[YoloObjectDetectorTrt] Preempt check for objects action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetectorTrt::isCheckingForObjects() const
{
  return (ros::ok() && checkForObjectsActionServer_->isActive()
      && !checkForObjectsActionServer_->isPreemptRequested());
}


bool YoloObjectDetectorTrt::publishDetectionImage(const cv::Mat& detectionImage)
{
  if (detectionImagePublisher_.getNumSubscribers() < 1)
    return false;
  cv_bridge::CvImage cvImage;
  //cvImage.header = headerBuff_[buffRdInd_];
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublisher_.publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}


// double YoloObjectDetectorTrt::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

void *YoloObjectDetectorTrt::detectInThread()
{
  running_ = 1;
  
  Mat im = image_to_mat(buffLetter_[buffRdInd_]);
  cv::rotate(im, im, cv::ROTATE_90_CLOCKWISE);
  preprocessImage(im, (float *) buffers_[0], input_dims_[0]);
        
  //inference
  context_->enqueue(batch_size_, buffers_.data(), 0, nullptr);
        
  std::vector<cv::Rect> boxes;
  std::vector<int> classes;
  std::vector<std::pair<float,int>> scores;
  cv::Size orig_size;
  orig_size.width = im.cols;
  orig_size.height = im.rows;
        
  //postprocess
  postprocessResults(buffers_, output_dims_, batch_size_, yolo_masks_, yolo_anchors_,
  orig_size, yolo_threshold_, nms_threshold_, boxes, classes, scores);
  
  int nboxes = scores.size();
  int count = 0;
  
  Mat image_orig = im.clone();
  Mat roi = im.clone();
  Rect bbox;
  
  //Sort pair scores, extract bounding boxes and send them to ROS
  if(nboxes>0)
  {
    int object_id = 0;
    std::sort(scores.begin(), scores.end());
    for(int i=0; i<nboxes; i++)
    {
      int idx = scores[i].second; //index of highest score
      bbox = boxes[idx];

      float xmin = bbox.x - bbox.width / 2.;
      float xmax = bbox.x + bbox.width / 2.;
      float ymin = bbox.y - bbox.height / 2.;
      float ymax = bbox.y + bbox.height / 2.;

      // Make sure the bounding box doesn't leave the image boundary and draw bounding box
      rectifyBox(bbox);
      rectangle(im, bbox, Scalar(0,255,0),2.0);

      // iterate through possible boxes and collect the bounding boxes
      float x_center = (xmin + xmax) / 2;
      float y_center = (ymin + ymax) / 2;
      float BoundingBox_width = xmax - xmin;
      float BoundingBox_height = ymax - ymin;

      // define bounding box
      // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
      if (BoundingBox_width > 0.01 && BoundingBox_height > 0.01) {
        roiBoxes_[count].x = x_center;
        roiBoxes_[count].y = y_center;
        roiBoxes_[count].w = BoundingBox_width;
        roiBoxes_[count].h = BoundingBox_height;
        roiBoxes_[count].Class = classes[idx];
        roiBoxes_[count].prob = scores[idx].first;
        count++;
    }
  }
  }

  if (enableConsoleOutput_) {
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps_);
    printf("Objects:\n\n");
  }

  mat_ = im;

   //create array to store found bounding boxes
   //if no object detected, make sure that ROS knows that num = 0
   if (count == 0) {
     roiBoxes_[0].num = 0;
   } else {
     roiBoxes_[0].num = count;
   }

  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;
  return 0;
}

void YoloObjectDetectorTrt::writeImageToBuffer() {
  if (!initDone_) return;
  // Free space before assigning new value to avoid memory leak.
  free_image(buff_[buffWrtInd_]);
  buff_[buffWrtInd_] = mat_to_image(camImageCopy_); // this create new memory.
  headerBuff_[buffWrtInd_] = imageHeader_;

  // buffId_[buffRdInd_] = actionId_; // unsure about this
  letterbox_image_into(buff_[buffWrtInd_], yolo_res, yolo_res, buffLetter_[buffWrtInd_]);

  // Increase the index or not.
  int buff_new_wrt_ind = (buffWrtInd_ + 1)%3;
  if (buff_new_wrt_ind != buffRdInd_) {
    buffWrtInd_ = buff_new_wrt_ind;
  }
}

void *YoloObjectDetectorTrt::fetchInThread()
{
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    MatImageWithHeader_ imageAndHeader = getMatImageWithHeader();
    //  free space before assigning new value to avoid memory leak.
    // delete buff_[buffRdInd_].data;
    free_image(buff_[buffRdInd_]);
    buff_[buffRdInd_] = mat_to_image(imageAndHeader.image); // this create new memory.
    headerBuff_[buffRdInd_] = imageAndHeader.header;
    buffId_[buffRdInd_] = actionId_;
  }
  letterbox_image_into(buff_[buffRdInd_], yolo_res, yolo_res, buffLetter_[buffRdInd_]);
  return 0;
}

void *YoloObjectDetectorTrt::displayInThread(void *ptr)
{
  // int c = show_image_cv(buff_[(buffRdInd_ + 1)%3], "YOLO V3", waitKeyDelay_);
  cv::imshow("YOLO V3", mat_);
  int c = cv::waitKey(waitKeyDelay_);
  if (c != -1) c = c%256;
  if (c == 27) {
      demoDone_ = 1;
      return 0;
  } else if (c == 82) {
      demoThresh_ += .02;
  } else if (c == 84) {
      demoThresh_ -= .02;
      if(demoThresh_ <= .02) demoThresh_ = .02;
  } else if (c == 83) {
      demoHier_ += .02;
  } else if (c == 81) {
      demoHier_ -= .02;
      if(demoHier_ <= .0) demoHier_ = .0;
  }
  return 0;
}

void *YoloObjectDetectorTrt::displayLoop(void *ptr)
{
  while (1) {
    displayInThread(0);
  }
}

void *YoloObjectDetectorTrt::detectLoop(void *ptr)
{
  while (1) {
    detectInThread();
  }
}

void YoloObjectDetectorTrt::yolo()
{
  const auto wait_duration = std::chrono::milliseconds(2000);
  while (!getImageStatus()) {
    printf("Waiting for image.\n");
    if (!isNodeRunning()) {
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  std::thread detect_thread;
  std::thread fetch_thread;

  srand(2222222);

  int i;
  //demoTotal_ = sizeNetwork(net_);
  predictions_ = (float **) calloc(demoFrame_, sizeof(float*));
  for (i = 0; i < demoFrame_; ++i){
      predictions_[i] = (float *) calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float *) calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_ *) calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    MatImageWithHeader_ imageAndHeader = getMatImageWithHeader();
    buff_[0] = mat_to_image(imageAndHeader.image);
    headerBuff_[0] = imageAndHeader.header;
  }
  buff_[1] = copy_image(buff_[0]);
  buff_[2] = copy_image(buff_[0]);
  headerBuff_[1] = headerBuff_[0];
  headerBuff_[2] = headerBuff_[0];
  buffLetter_[0] = letterbox_image(buff_[0], yolo_res, yolo_res);
  buffLetter_[1] = letterbox_image(buff_[0], yolo_res, yolo_res);
  buffLetter_[2] = letterbox_image(buff_[0], yolo_res, yolo_res);
  mat_ = cv::Mat(cv::Size(buff_[0].w, buff_[0].h), CV_MAKETYPE(CV_8U, buff_[0].c));

  int count = 0;

  if (!demoPrefix_ && viewImage_) {
    cv::namedWindow("YOLO V3", cv::WINDOW_NORMAL);
    if (fullScreen_) {
      cv::setWindowProperty("YOLO V3", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    } else {
      cv::moveWindow("YOLO V3", 0, 0);
      cv::resizeWindow("YOLO V3", 640, 480);
    }
  }

  demoTime_ = what_time_is_it_now();
  initDone_ = true;
  while (!demoDone_) {
    // ROS_INFO_THROTTLE("While loop", 5);
    if (buffRdInd_ != buffWrtInd_) {
      // fetch_thread = std::thread(&YoloObjectDetectorTrt::fetchInThread, this);
      // detect_thread = std::thread(&YoloObjectDetectorTrt::detectInThread, this);
      detectInThread();
      if (!demoPrefix_) {
        fps_ = 1./(what_time_is_it_now() - demoTime_);
        demoTime_ = what_time_is_it_now();
        if (viewImage_) {
          displayInThread(0);
        }
        publishInThread();
      } else {
        char name[256];
        sprintf(name, "%s_%08d", demoPrefix_, count);
        save_image(buff_[buffRdInd_], name);
      }
      // fetch_thread.join();
      // detect_thread.join();
      ++count;
      buffRdInd_ = (buffRdInd_ + 1) % 3;
    }

    if (!isNodeRunning()) {
      demoDone_ = true;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

}


MatImageWithHeader_ YoloObjectDetectorTrt::getMatImageWithHeader()
{
  MatImageWithHeader_ ImageWithHeader = {.image = camImageCopy_, .header = imageHeader_};
  return ImageWithHeader;
}

bool YoloObjectDetectorTrt::getImageStatus(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool YoloObjectDetectorTrt::isNodeRunning(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

void *YoloObjectDetectorTrt::publishInThread()
{
  // Publish image.
  if (!publishDetectionImage(mat_)) {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }
  if(publishDetectionImage(mat_)){
    ROS_INFO("detection image has been broadcasted");
  }

  // Publish bounding boxes and detection result.
  int num = roiBoxes_[0].num;
  if (num > 0 && num <= 100) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < numClasses_; j++) {
        if (roiBoxes_[i].Class == j) {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    std_msgs::Int8 msg;
    msg.data = num;
    objectPublisher_.publish(msg);

    for (int i = 0; i < numClasses_; i++) {
      if (rosBoxCounter_[i] > 0) {
        darknet_ros_msgs::BoundingBox boundingBox;

        for (int j = 0; j < rosBoxCounter_[i]; j++) {
          int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
          int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

          boundingBox.Class = classLabels_[i];
          boundingBox.probability = rosBoxes_[i][j].prob;
          boundingBox.xmin = xmin;
          boundingBox.ymin = ymin;
          boundingBox.xmax = xmax;
          boundingBox.ymax = ymax;
          boundingBoxesResults_.bounding_boxes.push_back(boundingBox);
        }
      }
    }
    boundingBoxesResults_.header = headerBuff_[buffRdInd_];
    boundingBoxesResults_.image_header = headerBuff_[buffRdInd_];
    boundingBoxesPublisher_.publish(boundingBoxesResults_);
  } else {
    std_msgs::Int8 msg;
    msg.data = 0;
    objectPublisher_.publish(msg);
  }
  if (isCheckingForObjects()) {
    ROS_DEBUG("[YoloObjectDetectorTrt] check for objects in image.");
    darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
    objectsActionResult.id = buffId_[0];
    objectsActionResult.bounding_boxes = boundingBoxesResults_;
    checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
  }
  boundingBoxesResults_.bounding_boxes.clear();
  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return 0;
}

void YoloObjectDetectorTrt::interpretOutput(xt::xarray<float> &cpu_reshape, xt::xarray<float> &anchors_tensor, xt::xarray<float> &boxes, xt::xarray<float> &box_class_scores, xt::xarray<int> &box_classes)
{
   
        auto sigmoid_v = xt::vectorize(sigmoid_f);
	   auto exponential_v = xt::vectorize(exponential_f);
       
	   xt::xarray<float> box_xy = sigmoid_v(xt::strided_view(cpu_reshape,{xt::all(),xt::all(),xt::all(),xt::range(0,2)}));
        xt::xarray<float> box_wh = exponential_v(xt::strided_view(cpu_reshape,{xt::all(),xt::all(),xt::all(),xt::range(2,4)})) * anchors_tensor; 
	   xt::xarray<float> box_confidence = sigmoid_v(xt::strided_view(cpu_reshape,{xt::all(),xt::all(),xt::all(),4}));
	   xt::xarray<float> box_class_probs = sigmoid_v(xt::strided_view(cpu_reshape,{xt::all(),xt::all(),xt::all(),xt::range(5,9)}));
        box_confidence = box_confidence.reshape({box_confidence.shape(0),box_confidence.shape(1),box_confidence.shape(2),1});
          
	   int grid_h = cpu_reshape.shape(0); 
	   int grid_w = cpu_reshape.shape(1);
	   xt::xarray<int> aux_arange = xt::arange(0, grid_w);
	   xt::xarray<int> aux_tile = xt::tile(aux_arange, grid_w);
	   xt::xarray<int> col = aux_tile.reshape({-1, grid_w});
        aux_arange = xt::arange(0, grid_h).reshape({-1,1}); 	
	   xt::xarray<int> row = xt::tile(aux_arange, grid_h);
        row = row.reshape({grid_h,grid_h});
        row = xt::transpose(row,{1,0});

	   col = col.reshape({grid_h,grid_w,1,1});
	   col = xt::repeat(col,3,2);
	   row = row.reshape({grid_h,grid_w,1,1});
	   row = xt::repeat(row,3,2);
	  
	   xt::xarray<int> grid = xt::concatenate(xtuple(col, row), 3);
	   box_xy += grid;
	   box_xy /= (grid_w, grid_h); 
	   box_wh /= (yolo_res,yolo_res); //input resolution yolo
	   box_xy -= (box_wh / 2.);

	   boxes = xt::concatenate(xtuple(box_xy, box_wh), 3);
	   xt::xarray<float> box_scores = box_confidence * box_class_probs;
	   box_classes = xt::argmax(box_scores,3);
	   box_class_scores = xt::amax(box_scores,3);
}

void YoloObjectDetectorTrt::postprocessResults(std::vector<void*> gpu_output, const std::vector<nvinfer1::Dims> &dims, int batch_size, std::vector<std::vector<int>> &yolo_masks, std::vector<std::vector<float>> &yolo_anchors, const cv::Size orig_dims, float threshold, float nms_threshold, std::vector<cv::Rect> &boxes_return, std::vector<int> &classes_return, std::vector<std::pair<float,int>> &scores_return)
{
    xt::xarray<float> boxes_final = xt::empty<float>({1, 4});
    std::vector<std::vector<std::size_t>> shapes = { {27, 19, 19}, {27,38,38 }, {27,76,76}};
    std::vector<int> classes_final;
    std::vector<float> scores_final;
    for(int i = 0; i < dims.size(); i++)
    {
	   std::vector<float> cpu_output(getSizeByDim(dims[i]) * batch_size);
	   
        cudaMemcpy(cpu_output.data(), (float *) gpu_output[i+1], cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

        xt::xarray<float> cpu_reshape = xt::adapt(cpu_output,shapes[i]);
        xt::xarray<float> boxes; 
	   xt::xarray<int> box_classes; 
	   xt::xarray<float> box_class_scores;

        reshapeOutput(cpu_reshape);

        std::vector<std::size_t> shape_anchor = {1, 1, yolo_anchors[i].size()/2, 2};
        xt::xarray<float> anchors_tensor = xt::adapt(yolo_anchors[i], shape_anchor);

        interpretOutput(cpu_reshape, anchors_tensor, boxes, box_class_scores, box_classes);

        auto pos_aux = xt::where(box_class_scores >= threshold);
	    xt::xarray<int> pos = xt::from_indices(pos_aux);

        for(int k =0; k < pos_aux[0].size(); k++){
            int indx1 = pos(0,k)*boxes.shape(1)*boxes.shape(2)+pos(1,k) * boxes.shape(2)+pos(2,k);
            scores_final.push_back(box_class_scores(indx1));
            classes_final.push_back(box_classes(indx1));
            xt::xarray<float> a = xt::view(boxes, pos(0,k), pos(1,k), pos(2,k), xt::range(0, 4));
            a = a.reshape({1,boxes.shape(3)});
            boxes_final = xt::concatenate(xtuple(boxes_final,a)); 
       }
    }

    // NMS for each class
    boxes_final = xt::view(boxes_final, xt::range(1,_), xt::all());
    xt::xarray<int> image_dims = {orig_dims.width, orig_dims.height, orig_dims.width, orig_dims.height};
    boxes_final = boxes_final * image_dims;
    std::vector<int> unique_classes;
    unique_classes = classes_final;
    std::sort(unique_classes.begin(), unique_classes.end());
    auto  last = std::unique(unique_classes.begin(), unique_classes.end());
    unique_classes.erase(last, unique_classes.end()); 
    for (int i : unique_classes)
    {   
        std::vector<int> ind;
        std::vector<cv::Rect> boxes_selected;
        std::vector<float> scores;
        for(int j=0;j<classes_final.size();j++)
        {
            if(classes_final[j]==i)
            {
                scores.push_back(scores_final[j]);
                boxes_selected.push_back(cv::Rect(boxes_final(j,0),boxes_final(j,1),boxes_final(j,2),boxes_final(j,3)));
            }
        }
        cv::dnn::NMSBoxes(boxes_selected, scores, threshold, nms_threshold,ind);
        for(int k=0;k<ind.size();k++)
        {
            boxes_return.push_back(boxes_selected[k]);
            scores_return.push_back(std::make_pair(scores[k],k));
            classes_return.push_back(i);
        }
    }    
}

void YoloObjectDetectorTrt::rectifyBox(Rect& bbox)
{
     if(bbox.x < 0)
         bbox.x = 0;
     if(bbox.y < 0)
         bbox.y = 0;
     if(bbox.x + bbox.width > original_dims_.width)
         bbox.width = original_dims_.width - bbox.x;
     if(bbox.y + bbox.height > original_dims_.height)
         bbox.height = original_dims_.height - bbox.y;
}

void YoloObjectDetectorTrt::loadEngine(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    std::cout<<"Reading engine from file "<<std::endl;
    
    std::ifstream engineFile(model_path, std::ios::binary);
    if (!engineFile)
    {
        std::cout << "Error opening engine file: " << model_path << std::endl;
        return;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        std::cout << "Error loading engine file: " << model_path << std::endl;
        return;
    }

    TRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLoggerYolo)};
    engine_.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    context_.reset(engine_->createExecutionContext());

}

size_t YoloObjectDetectorTrt::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

double YoloObjectDetectorTrt::sigmoid_f(float val)
{
    return 1.0/(1.0 + std::exp(-val));
}

double YoloObjectDetectorTrt::exponential_f(float val)
{
        return std::exp(val);
}

void YoloObjectDetectorTrt::reshapeOutput(xt::xarray<float>& cpu_reshape)
{
        cpu_reshape = xt::transpose(cpu_reshape, {1,2,0});
        std::size_t dim4 = (4 + 1 + numClasses_);
        std::size_t s1 = cpu_reshape.shape(0);
        std::size_t s2 = cpu_reshape.shape(1);
        cpu_reshape = cpu_reshape.reshape({s1,s2,3,dim4});
}

void YoloObjectDetectorTrt::preprocessImage(cv::Mat frame, float* gpu_input, const nvinfer1::Dims& dims)
{
    auto input_width = dims.d[3];
    auto input_height = dims.d[2];
    auto channels = dims.d[1];
    auto input_size = cv::Size(input_width, input_height);
    

    cv::Mat resize_frame;
    cv::resize(frame, resize_frame, input_size, 0, 0, cv::INTER_NEAREST);
    
    // normalize
    cv::Mat flt_img;
    resize_frame.convertTo(flt_img, CV_32FC3, 1.f / 255.f);

    // upload image to GPU
    cv::cuda::GpuMat gpu_frame_final;
    gpu_frame_final.upload(flt_img);
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(gpu_frame_final, chw); 
}

} /* namespace darknet_ros*/
