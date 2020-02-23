/*
 * YoloObjectDetector.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"

// Check for xServer
#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_ros {

char *cfg;
char *weights;
char *data;
char **detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh)
    : nodeHandle_(nh),
      imageTransport_(nodeHandle_),
      numClasses_(0),
      classLabels_(0),
      rosBoxes_(0),
      rosBoxCounter_(0)
{
  ROS_INFO("[YoloObjectDetector] Node started.");

  // Read parameters from config file.
  if (!readParameters()) {
    ros::requestShutdown();
  }

  init();
}

YoloObjectDetector::~YoloObjectDetector()
{
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
}

bool YoloObjectDetector::readParameters()
{
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

  // Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[YoloObjectDetector] Xserver is running.");
  } else {
    ROS_INFO("[YoloObjectDetector] Xserver is not running.");
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

void YoloObjectDetector::init()
{
  ROS_INFO("[YoloObjectDetector] init().");

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param("yolo_model/threshold/value", thresh, (float) 0.3);

  // Path to weights file.
  nodeHandle_.param("yolo_model/weight_file/name", weightsModel,
                    std::string("yolov2-tiny.weights"));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
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

  // Load network.
  setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_,
                0, 0, 1, 0.5, 0, 0, 0, 0);
  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);


  // Initialize publisher and subscriber.
  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName1;
  std::string boundingBoxesTopicName2;
  std::string boundingBoxesTopicName3;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName1;
  std::string detectionImageTopicName2;
  std::string detectionImageTopicName3;
  int detectionImageQueueSize;
  bool detectionImageLatch;

  timeoflastpushCam1, timeoflastpushCam2, timeoflastpushCam3 = what_time_is_it_now();

  nodeHandle_.param("subscribers/camera_reading/topic1", cameraTopicName1,
                    std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/topic2", cameraTopicName2,
                    std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/topic3", cameraTopicName3,
                    std::string("/camera/image_raw"));

  nodeHandle_.param("subscribers/camera_reading/FPSCam1", FPSCam1, 1.0);
  nodeHandle_.param("subscribers/camera_reading/FPSCam2", FPSCam2, 1.0);
  nodeHandle_.param("subscribers/camera_reading/FPSCam3", FPSCam3, 1.0);


  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName,
                    std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);

  nodeHandle_.param("publishers/bounding_boxes/topic1", boundingBoxesTopicName1,
                    std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/topic2", boundingBoxesTopicName2,
                    std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/topic3", boundingBoxesTopicName3,
                    std::string("bounding_boxes"));

  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);


  nodeHandle_.param("publishers/detection_image/topic1", detectionImageTopicName1,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/topic2", detectionImageTopicName2,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/topic3", detectionImageTopicName3,
                    std::string("detection_image"));


  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

  imageSubscriber_1 = imageTransport_.subscribe(cameraTopicName1, cameraQueueSize,
                                               &YoloObjectDetector::cameraCallback1, this);
  imageSubscriber_2 = imageTransport_.subscribe(cameraTopicName2, cameraQueueSize,
                                               &YoloObjectDetector::cameraCallback2, this);
  imageSubscriber_3 = imageTransport_.subscribe(cameraTopicName3, cameraQueueSize,
                                               &YoloObjectDetector::cameraCallback3, this);

  // objectPublisher_ = nodeHandle_.advertise<std_msgs::Int8>(objectDetectorTopicName,
  //                                                          objectDetectorQueueSize,
  //                                                          objectDetectorLatch);

  boundingBoxesPublisher_1 = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(
      boundingBoxesTopicName1, boundingBoxesQueueSize, boundingBoxesLatch);
  boundingBoxesPublisher_2 = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(
      boundingBoxesTopicName2, boundingBoxesQueueSize, boundingBoxesLatch);
  boundingBoxesPublisher_3 = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(
      boundingBoxesTopicName3, boundingBoxesQueueSize, boundingBoxesLatch);

  detectionImagePublisher_1 = nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName1,
                                                                       detectionImageQueueSize,
                                                                       detectionImageLatch);
  detectionImagePublisher_2 = nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName2,
                                                                       detectionImageQueueSize,
                                                                       detectionImageLatch);
  detectionImagePublisher_3 = nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName3,
                                                                       detectionImageQueueSize,
                                                                       detectionImageLatch);
  // Action servers.
  std::string checkForObjectsActionName;
  nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName,
                    std::string("check_for_objects"));
  checkForObjectsActionServer_.reset(
      new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
  checkForObjectsActionServer_->registerGoalCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
  checkForObjectsActionServer_->registerPreemptCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionPreemptCB, this));
  checkForObjectsActionServer_->start();
}
//************************************/
//**********CAMERA CALLBACKS**********/
//************************************/
void YoloObjectDetector::cameraCallback1(const sensor_msgs::ImageConstPtr& msg)
{

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
      camImageCopy_ = cam_image->image.clone(); // camImageCopy_ IS THE IMAGE WRITTEN TO THE BUFFER!!
      writeImageToQueue(cameraTopicName1);
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
void YoloObjectDetector::cameraCallback2(const sensor_msgs::ImageConstPtr& msg)
{

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
      camImageCopy_ = cam_image->image.clone(); // camImageCopy_ IS THE IMAGE WRITTEN TO THE BUFFER!!
      writeImageToQueue(cameraTopicName2);
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
void YoloObjectDetector::cameraCallback3(const sensor_msgs::ImageConstPtr& msg)
{
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
      camImageCopy_ = cam_image->image.clone(); // camImageCopy_ IS THE IMAGE WRITTEN TO THE BUFFER!!
      writeImageToQueue(cameraTopicName3);
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
//************************************/
//********END CAMERA CALLBACKS********/
//************************************/

void YoloObjectDetector::checkForObjectsActionGoalCB()
{
  ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

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

void YoloObjectDetector::checkForObjectsActionPreemptCB()
{
  ROS_DEBUG("[YoloObjectDetector] Preempt check for objects action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetector::isCheckingForObjects() const
{
  return (ros::ok() && checkForObjectsActionServer_->isActive()
      && !checkForObjectsActionServer_->isPreemptRequested());
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage, const std_msgs::Header& header)
{
  ros::Publisher detectionImagePublisher;
  if(CurrImgTopic == cameraTopicName1)
    {
      detectionImagePublisher = detectionImagePublisher_1;
    }
    else if(CurrImgTopic == cameraTopicName2)
    {
      detectionImagePublisher = detectionImagePublisher_2;
    }
    else
    {
      detectionImagePublisher = detectionImagePublisher_3;
    }

  if (detectionImagePublisher.getNumSubscribers() < 1)
    return false;

  cv_bridge::CvImage cvImage;
  cvImage.header.stamp = header.stamp;
  cvImage.header.frame_id = header.frame_id;
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublisher.publish(*cvImage.toImageMsg());
  return true;
}

// double YoloObjectDetector::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

int YoloObjectDetector::sizeNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      count += l.outputs;
    }
  }
  return count;
}

void YoloObjectDetector::rememberNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
}

detection *YoloObjectDetector::avgPredictions(network *net, int *nboxes)
{
  int i, j;
  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for(j = 0; j < demoFrame_; ++j){
    axpy_cpu(demoTotal_, 1./demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  detection *dets = get_network_boxes(net, Queue.front().img.w, Queue.front().img.h, demoThresh_, demoHier_, 0, 1, nboxes);
  return dets;
}

void *YoloObjectDetector::detectInThread()
{
  if(Queue.empty()) return 0;

    running_ = 1;
    float nms = .4;
    layer l = net_->layers[net_->n - 1];
    float *X = Queue.front().letter.data;
    float *prediction = network_predict(net_, X);

    rememberNetwork(net_);
    detection *dets = 0;
    int nboxes = 0;
    dets = avgPredictions(net_, &nboxes);

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    if (enableConsoleOutput_) {
      printf("\nFPS:%.1f\n",fps_);
      printf("Objects:\n\n");
    }
    image display = Queue.front().img;
    draw_detections(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_);

    // Delete memory of previous ipl_
    cvReleaseImage(&ipl_);
    rgbgr_image(display);
    ipl_ = image_to_ipl(display);

    // extract the bounding boxes and send them to ROS
    int i, j;
    int count = 0;
    for (i = 0; i < nboxes; ++i) {
      float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
      float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
      float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
      float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

      if (xmin < 0)
        xmin = 0;
      if (ymin < 0)
        ymin = 0;
      if (xmax > 1)
        xmax = 1;
      if (ymax > 1)
        ymax = 1;;

      // iterate through possible boxes and collect the bounding boxes
      for (j = 0; j < demoClasses_; ++j) {
        if (dets[i].prob[j]) {
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
            roiBoxes_[count].Class = j;
            roiBoxes_[count].prob = dets[i].prob[j];
            count++;
          }
        }
      }
    }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  if (count == 0) {
    roiBoxes_[0].num = 0;
  } else {
    roiBoxes_[0].num = count;
  }

  free_detections(dets, nboxes);
  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;

  if(Queue.front().imgTopic == cameraTopicName1)
  {
    CurrImgTopic = cameraTopicName1;
  }
  else if (Queue.front().imgTopic == cameraTopicName2)
  {
    CurrImgTopic = cameraTopicName2;
  }
  else
  {
    CurrImgTopic = cameraTopicName3;
    std::cout << std::endl << CurrImgTopic << std::endl;
  }
  {
    boost::unique_lock<boost::shared_mutex> queue_lock(mutexQueue_);
    delete Queue.front().img.data;
    delete Queue.front().letter.data;
    std_msgs::Header currHeader = Queue.front().header; // To use for publishing.
    Queue.pop();
  }

  //********************************/
  /*********PUBLISH SECTION*********/
  //********************************/

  cv::Mat cvImage = cv::cvarrToMat(ipl_);
  if (!publishDetectionImage(cvImage.clone(), currHeader)) {
    ROS_DEBUG("Detection image has not been broadcasted.");
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

    // std_msgs::Int8 msg;
    // msg.data = num;
    // objectPublisher_.publish(msg);

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
    boundingBoxesResults_.header.stamp = currHeader.stamp; //ros::Time::now();
    boundingBoxesResults_.header.frame_id = currHeader.frame_id;
    boundingBoxesResults_.image_header = currHeader;
    if(CurrImgTopic == cameraTopicName1)
    {
      boundingBoxesPublisher_1.publish(boundingBoxesResults_);
    }
    else if(CurrImgTopic == cameraTopicName2)
    {
      boundingBoxesPublisher_2.publish(boundingBoxesResults_);
    }
    else
    {
      boundingBoxesPublisher_3.publish(boundingBoxesResults_);
    }

  }
  else
  {
    // std_msgs::Int8 msg;
    // msg.data = 0;
    // objectPublisher_.publish(msg);
  }
  if (isCheckingForObjects()) {
    ROS_DEBUG("[YoloObjectDetector] check for objects in image.");
    darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
    // objectsActionResult.id = buffId_[0];
    objectsActionResult.bounding_boxes = boundingBoxesResults_;
    checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
  }
  boundingBoxesResults_.bounding_boxes.clear();
  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }
  //********************************/
  /*******END PUBLISH SECTION*******/
  //********************************/


  return 0;
}

void YoloObjectDetector::writeImageToQueue(std::string topic) {
  // if (!initDone_)
  // {
  //   printf("Waiting for init to finish\n");
  //   return;
  // }
  if (topic == cameraTopicName1)
  {
    if ((what_time_is_it_now() - timeoflastpushCam1) < 1/FPSCam1)
    {
      return;
    }
    timeoflastpushCam1 = what_time_is_it_now();
  }
  else if (topic == cameraTopicName2)
  {
    if ((what_time_is_it_now() - timeoflastpushCam2) < 1/FPSCam2)
    {
      return;
    }
    timeoflastpushCam2 = what_time_is_it_now();
  }
  else
  {
    if ((what_time_is_it_now() - timeoflastpushCam3) < 1/FPSCam3)
    {
      return;
    }
    timeoflastpushCam3 = what_time_is_it_now();
  }

  IplImage* ROS_img = new IplImage(camImageCopy_);

  {
    boost::unique_lock<boost::shared_mutex> queue_lock(mutexQueue_);
    while(Queue.size() > MAXQUEUESIZE - 1)
    {
      delete Queue.front().img.data;
      // delete Queue.front().letter.data;
      Queue.pop();

    }
    QueueType QE;
    QE.img = ipl_to_image(ROS_img); // this create new memory.
    QE.header = imageHeader_;
    QE.imgTopic = topic;


    rgbgr_image(QE.img);
    QE.letter = letterbox_image(QE.img, net_->w, net_->h);
    letterbox_image_into(QE.img, net_->w, net_->h, QE.letter);

    Queue.push(QE);
    std::cout << "Queue size: " << Queue.size() << "\n";
  }
}

void YoloObjectDetector::setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                                      char **names, int classes,
                                      int delay, char *prefix, int avg_frames, float hier, int w, int h,
                                      int frames, int fullscreen)
{
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image **alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
  printf("YOLO V3\n");
  net_ = load_network(cfgfile, weightfile, 0);
  set_batch_network(net_, 1);
}

void YoloObjectDetector::yolo()
{
  const auto wait_duration = std::chrono::milliseconds(2000);
  while (!imageStatus_) {
    printf("Waiting for image.\n");
    if (!isNodeRunning()) {
      printf("SHIT");
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  std::thread detect_thread;

  srand(2222222);

  int i;
  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float **) calloc(demoFrame_, sizeof(float*));
  for (i = 0; i < demoFrame_; ++i){
      predictions_[i] = (float *) calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float *) calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_ *) calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));


  while(Queue.size() == 0) // Must loop until something is added to Queue.
  {
    printf("Queue is empty\n");
  }

  ipl_ = cvCreateImage(cvSize(Queue.front().img.w, Queue.front().img.h), IPL_DEPTH_8U, Queue.front().img.c);
  int count = 0;

  demoTime_ = what_time_is_it_now();
  initDone_ = true;
  while (!demoDone_) {
    detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
    //detectInThread();
    if (!demoPrefix_) {
      fps_ = 1./(what_time_is_it_now() - demoTime_);
      demoTime_ = what_time_is_it_now();
    } else {
      char name[256];
      sprintf(name, "%s_%08d", demoPrefix_, count);
      save_image(Queue.front().img, name);
    }

    detect_thread.join();
    ++count;

    if (!isNodeRunning()) {
      demoDone_ = true;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

IplImageWithHeader_ YoloObjectDetector::getIplImageWithHeader()
{
  IplImage* ROS_img = new IplImage(camImageCopy_);
  IplImageWithHeader_ header = {.image = ROS_img, .header = imageHeader_};
  return header;
}

bool YoloObjectDetector::getImageStatus(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool YoloObjectDetector::isNodeRunning(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}



} /* namespace darknet_ros*/
