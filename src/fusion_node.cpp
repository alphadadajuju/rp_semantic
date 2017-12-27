// ROS
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

// ROS messages
#include <std_msgs/Int16.h>
#include "std_msgs/String.h"
#include "visualization_msgs/Marker.h"
#include "std_msgs/Float64MultiArray.h"

#include "rp_semantic/Cluster.h"
#include "rp_semantic/LabelClusters.h"
//#include "rp_semantic/Frame.h"

#include "rp_semantic/RGB2LabelProb.h"

// Headers C_plus_plus
#include <iostream>
#include <fstream>
#include <string>
#include <csignal>
#include <sstream>
#include <cmath>
#include <limits>
#include <float.h>
#include "boost/filesystem.hpp"

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

#include <pcl/filters/frustum_culling.h>
#include <pcl/visualization/common/common.h>

// PCL CLustering
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// OpenCV 
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Geometry>


using namespace std;
//typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

void read_poses(const std::string& name, vector<Eigen::Matrix4f> &v){

    ifstream inFile;
    inFile.open(name);
    if (!inFile)
        ROS_ERROR_STREAM("Unable to open file datafile.txt");


    float val;
    while(inFile >> val){
        Eigen::Matrix4f pose;

        pose(0,0) = val;
        for (int i = 0; i < 3; ++i) { // row
            for (int j = 0; j < 4; ++j) { //col
                if (i == 0 && j == 0) continue;

                inFile >> val;
                pose(i,j) = val;
            }
        }
        // Fill in last row
        pose(3,0) = 0; pose(3,1) = 0; pose(3,2) = 0; pose(3,3) = 1;

        v.push_back(pose);
    }

}

void read_directory(const std::string& name, vector<string> &v) {

    try{
        boost::filesystem::path p(name);
        boost::filesystem::directory_iterator start(p);
        boost::filesystem::directory_iterator end;

        //GET files in directory
        vector<boost::filesystem::path> paths;
        std::copy(start, end, back_inserter(paths));

        //SORT them according to criteria
        struct sort_functor {
            //Return true if b bigger than a, false if equal or smaller
            bool operator()(const boost::filesystem::path &a, const boost::filesystem::path &b) {
                if(a.string().size() == b.string().size())
                    return a.compare(b) < 0;
                else
                    return a.string().size() < b.string().size();
            }
        };
        std::sort(paths.begin(), paths.end(), sort_functor());

        //OUTPUT vector of ordered strings
        for (vector<boost::filesystem::path>::const_iterator it(paths.begin()); it != paths.end(); ++it)
            v.push_back(it->string());

    }catch (const boost::filesystem::filesystem_error& ex)
    {
        cout << ex.what() << '\n';
    }

}

class SemanticFusion{

private:
    ros::NodeHandle nh;         // node handler
    ros::Publisher marker_pub ;   // Clusters node Message Publisher
    ros::Publisher pc_display_pub ;   // Clusters node Message Publisher
    ros::Publisher image_pub;
    ros::Publisher fused_pc_pub ;   // Clusters node Message Publisher
    ros::ServiceClient segnet_client;


    void displayCameraMarker(Eigen::Matrix4f cam_pose);
    void displayPointcloud(const pcl::PointCloud<pcl::PointXYZRGB> &pc);

public:
    SemanticFusion();

    bool debug_mode;

    bool loadPlaceData(const string &base_path, pcl::PointCloud<pcl::PointXYZRGB> &pc, vector<Eigen::Matrix4f> &poses, vector<string> &images);
    void createFusedSemanticMap(const string &base_path, const pcl::PointCloud<pcl::PointXYZRGB> &pc, const vector<Eigen::Matrix4f> &poses, const vector<string> &images);

    void loadAndPublishLabelledPointcloud(string path);
    void publishBoWPTestPointcloud();

    void loadAndPublishMonoClassPointcloud(string path);
};


SemanticFusion::SemanticFusion(){
    debug_mode = true;

    pc_display_pub = nh.advertise<sensor_msgs::PointCloud2>("rp_semantic/fusion_pointcloud", 10); // Publisher
    marker_pub = nh.advertise<visualization_msgs::Marker>("rp_semantic/camera_pose", 10); // Publisher
    fused_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("rp_semantic/semantic_fused_pc", 10); // Publisher
    image_pub = nh.advertise<sensor_msgs::Image>("rp_semantic/camera_image", 10);

    ros::service::waitForService("rgb_to_label_prob", 10);
    segnet_client = nh.serviceClient<rp_semantic::RGB2LabelProb>("rgb_to_label_prob");
    ros::Duration(4).sleep();
}

bool
SemanticFusion::loadPlaceData(const string &base_path, pcl::PointCloud<pcl::PointXYZRGB> &pc, vector<Eigen::Matrix4f> &poses,
                              vector<string> &images) {

    ROS_INFO_STREAM("fusion_node: " << " processing directory " << base_path);

    boost::filesystem::path p(base_path);
    if(!boost::filesystem::exists(base_path))
        ROS_ERROR_STREAM("fusion_node: Invalid base-path for loading place data");


    string pc_path = base_path + "cloud.ply";
    pcl::io::loadPLYFile(pc_path, pc);

    string poses_path = base_path + "poses.txt";
    read_poses(poses_path, poses);

    string rgb_path = base_path + "rgb";
    read_directory(rgb_path, images);

    ROS_INFO_STREAM("fusion_node: " << " loaded cloud with " << pc.points.size() << " points.");
    ROS_INFO_STREAM("fusion_node: " << " loaded " << poses.size() << " poses.");
    ROS_INFO_STREAM("fusion_node: " << " loaded " << images.size() << " image's path.");

    return true;
}

void SemanticFusion::createFusedSemanticMap(const string &base_path, const pcl::PointCloud<pcl::PointXYZRGB> &pc,
                                            const vector<Eigen::Matrix4f> &poses, const vector<string> &images) {

    // Initialize label probability structure
    vector<float> label_prob_init(37, 1.0f/37.0f);
    std::vector<vector<float> > label_prob(pc.points.size(), label_prob_init);

    // Initialize other objects
    pcl::FrustumCulling<pcl::PointXYZRGB> fc;
    fc.setInputCloud ( pc.makeShared() );
    // Set frustrum according to Kinect specifications
    fc.setVerticalFOV (45);
    fc.setHorizontalFOV (58);
    fc.setNearPlaneDistance (0.5);
    fc.setFarPlaneDistance (6); //Should be 4m. but we bump it up a little

    // Initialization of P
    Eigen::Matrix4f P;
    //P << 525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    P << 570.34222412109375, 0., 319.5, 0., 0.0, 570.34222412109375, 239.5, 0., 0., 0.0, 1., 0.0, 0.0, 0.0, 0.0, 1.0;

    // For each pose & image
    for (int i = 0; i < poses.size(); ++i) {
        ROS_INFO_STREAM_THROTTLE(5, "Processing node " << i << "/" << poses.size());

        if(!ros::ok()) break;

        // FrustrumCulling from poses-> indices of visible points
        Eigen::Matrix3f rot_xtion;
        rot_xtion = Eigen::AngleAxisf( -0.5f*M_PI, Eigen::Vector3f::UnitZ())
            * Eigen::AngleAxisf( 0.0f*M_PI, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf( -0.5f*M_PI, Eigen::Vector3f::UnitX());

        Eigen::Matrix4f rot4 = Eigen::Matrix4f::Identity(4,4);
        rot4.topLeftCorner(3,3) = rot_xtion;

        Eigen::Matrix4f xtion_pose =  poses[i] * rot4.inverse();
        //xtion_pose.topLeftCorner(3,3) = rot_xtion * xtion_pose.topLeftCorner(3,3);


        fc.setCameraPose(xtion_pose);
        std::vector<int> inside_indices; // Indices of points in pc_gray inside camera frustrum at pose[i]
        fc.filter(inside_indices);

        // SRV: request image labelling through segnet
        cv::Mat cv_img = cv::imread(images[i], CV_LOAD_IMAGE_COLOR);
        sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_img).toImageMsg();

        rp_semantic::RGB2LabelProb srv_msg;
        srv_msg.request.rgb_image = *image_msg.get();
        segnet_client.call(srv_msg);

        //Display RGB image to match server's time of publication
        sensor_msgs::ImagePtr rgb_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_img).toImageMsg();
        image_pub.publish(rgb_img_msg);

        std_msgs::Float64MultiArray frame_label_probs(srv_msg.response.image_class_probability);
        int dim_stride_1 = frame_label_probs.layout.dim[1].stride;
        int dim_stride_2 = frame_label_probs.layout.dim[2].stride;


        //Create matrix to go from RVIZ depth camera frame to CV rgb camera frame
        Eigen::Matrix4f rviz2cv = Eigen::Matrix4f::Identity(4,4);
        Eigen::Matrix3f r;
        r = Eigen::AngleAxisf(0.0f*M_PI, Eigen::Vector3f::UnitZ())
            * Eigen::AngleAxisf( -0.5f*M_PI, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf( 0.5f*M_PI, Eigen::Vector3f::UnitX());
        rviz2cv.topLeftCorner(3,3) = r;

        Eigen::Matrix4f depth2optical = Eigen::Matrix4f::Identity(4,4);
        depth2optical(0,3) = -0.045; //-0.025

        // Get camera matrix from poses_i and K
        Eigen::Matrix4f pose_inv = xtion_pose.inverse();
        Eigen::Matrix4f world2pix = P * depth2optical * rviz2cv * pose_inv;

        //Create "Z buffer" emulator
        int pixel_point_proj[480][640];
        float pixel_point_dist[480][640];
        fill_n(&pixel_point_proj[0][0], sizeof(pixel_point_proj) / sizeof(**pixel_point_proj), -1);
        fill_n(&pixel_point_dist[0][0], sizeof(pixel_point_dist) / sizeof(**pixel_point_dist), FLT_MAX);

        for(std::vector<int>::iterator it = inside_indices.begin(); it != inside_indices.end(); it++){
            // Backproject points using camera matrix (discard out of range)
            Eigen::Vector4f point_w(pc.points[*it].x, pc.points[*it].y, pc.points[*it].z, 1.0);
            Eigen::Vector4f point_px = world2pix * point_w;
            if(point_px(2) == 0){ point_px(2) +=1; } // Adapt homogenous for 4x4 efficient multiplication

            int pix_x = std::round(point_px(0)/point_px(2));
            int pix_y = std::round(point_px(1)/point_px(2));

            if(pix_x < 0 || pix_x >= 640 || pix_y < 0 || pix_y >= 480)
                continue;

            //Compute distance from camera position to point position
            Eigen::Vector3f cam_position(xtion_pose(0,3), xtion_pose(1,3), xtion_pose(2,3));
            Eigen::Vector3f point_position(pc.points[*it].x, pc.points[*it].y, pc.points[*it].z);
            Eigen::Vector3f cam_point_vec = cam_position - point_position;
            float cam_point_dist = cam_point_vec.norm();

            //Check against tables and keep closest point's index
            if(cam_point_dist < pixel_point_dist[pix_y][pix_x]){
                pixel_point_proj[pix_y][pix_x] = *it;
                pixel_point_dist[pix_y][pix_x] = cam_point_dist;
            }
        }


        //Get associated distributions, multiply  distributions together and renormalize
        for (int y = 0; y < 480; ++y) {
            for (int x = 0; x < 640; ++x) {
                if(pixel_point_proj[y][x] == -1){
                    continue;
                }

                //Get idx of nearest projected point from pointcloud
                int pc_idx = pixel_point_proj[y][x];

                // Multiply each element of the distribution and Get the sum of the final distribution
                float prob_dist_sum = 0.0;
                for (int cl = 0; cl < 37; ++cl) {
                    label_prob[pc_idx][cl] *= frame_label_probs.data[dim_stride_1*y + dim_stride_2*x + cl];
                    prob_dist_sum += label_prob[pc_idx][cl];
                }

                // Divide each element by the sum
                for (int cl = 0; cl < 37; ++cl) {
                    label_prob[pc_idx][cl] *= 1/prob_dist_sum;
                }
            }
        }

        if(debug_mode){
            // Build XYZL pointcloud
            pcl::PointCloud<pcl::PointXYZL> labelled_pc;
            for(int i = 0; i < pc.points.size(); i++) {
                pcl::PointXYZL p;
                p.x = pc.points[i].x;
                p.y = pc.points[i].y;
                p.z = pc.points[i].z;

                uint16_t max_label = 38;
                float max_label_prob = 1.0f/37.0f;
                for (uint16_t cl = 0; cl < 37; ++cl) {
                    if(label_prob[i][cl] > max_label_prob){
                        max_label = cl;
                        max_label_prob = label_prob[i][cl];
                    }
                }

                p.label = max_label;

                labelled_pc.push_back(p);
            }

            cv::Mat label2bgr = cv::imread("/home/albert/rp_data/sun.png", CV_LOAD_IMAGE_COLOR);

            // DEBUG:: Build XYZRGB pointcloud
            pcl::PointCloud<pcl::PointXYZRGB> label_rgb_pc;
            for(int i = 0; i < labelled_pc.points.size(); i++) {
                pcl::PointXYZRGB p;
                p.x = labelled_pc.points[i].x;
                p.y = labelled_pc.points[i].y;
                p.z = labelled_pc.points[i].z;

                if(labelled_pc.points[i].label > 37){
                    p.b = 0;
                    p.g = 0;
                    p.r = 0;
                }else{
                    cv::Vec3b color = label2bgr.at<cv::Vec3b>(cv::Point(labelled_pc.points[i].label+1, 0));
                    p.b = color.val[0];
                    p.g = color.val[1];
                    p.r = color.val[2];
                }

                label_rgb_pc.push_back(p);
            }


            displayCameraMarker(xtion_pose);
            displayPointcloud(label_rgb_pc);
        }
    }// For each pose

    // Build XYZL pointcloud
    pcl::PointCloud<pcl::PointXYZL> labelled_pc;
    for(int i = 0; i < pc.points.size(); i++) {
        pcl::PointXYZL p;
        p.x = pc.points[i].x;
        p.y = pc.points[i].y;
        p.z = pc.points[i].z;

        uint16_t max_label = 37;
        float max_label_prob = 1.0f/37.0f;
        for (uint16_t cl = 0; cl < 37; ++cl) {
            if(label_prob[i][cl] > max_label_prob){
                max_label = cl;
                max_label_prob = label_prob[i][cl];
            }
        }

        p.label = max_label;

        labelled_pc.push_back(p);
    }


    cv::Mat label2bgr = cv::imread("/home/albert/rp_data/sun.png", CV_LOAD_IMAGE_COLOR);
    sensor_msgs::ImagePtr rgb2label_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", label2bgr).toImageMsg();
    image_pub.publish(rgb2label_msg);

    // DEBUG:: Build XYZRGB pointcloud
    pcl::PointCloud<pcl::PointXYZRGB> label_rgb_pc;
    for(int i = 0; i < labelled_pc.points.size(); i++) {
        pcl::PointXYZRGB p;
        p.x = labelled_pc.points[i].x;
        p.y = labelled_pc.points[i].y;
        p.z = labelled_pc.points[i].z;

        if(labelled_pc.points[i].label > 37){
            p.b = 0;
            p.g = 0;
            p.r = 0;
        }else{
            cv::Vec3b color = label2bgr.at<cv::Vec3b>(cv::Point(labelled_pc.points[i].label+1, 0));
            p.b = color.val[0];
            p.g = color.val[1];
            p.r = color.val[2];
        }

        label_rgb_pc.push_back(p);
    }

    //Store pointcloud
    string labelled_cloud_path = base_path + "labelled_cloud.ply";
    string labelled_cloud_rgb_path = base_path + "labelled_cloud_rgb.ply";

    pcl::io::savePLYFile(labelled_cloud_path, labelled_pc, true);
    pcl::io::savePLYFile(labelled_cloud_rgb_path, label_rgb_pc, true);

    sensor_msgs::PointCloud2 pc_fused_msg;
    pcl::toROSMsg(labelled_pc, pc_fused_msg);
    pc_fused_msg.header.frame_id = "map";
    fused_pc_pub.publish(pc_fused_msg);
}

void SemanticFusion::displayCameraMarker(Eigen::Matrix4f cam_pose) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time();
    marker.ns = "rp_semantic";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = cam_pose(0, 3);
    marker.pose.position.y = cam_pose(1, 3);
    marker.pose.position.z = cam_pose(2, 3);

    Eigen::Matrix3f mat = cam_pose.topLeftCorner(3,3);
    Eigen::Quaternionf q(mat);

    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();

    marker.scale.x = 0.5;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 0.5;
    marker.color.g = 0.5;
    marker.color.b = 0.5;

    marker_pub.publish( marker );
}

void SemanticFusion::displayPointcloud(const pcl::PointCloud<pcl::PointXYZRGB> &pc) {
    sensor_msgs::PointCloud2 pc_disp_msg;
    pcl::toROSMsg(*pc.makeShared(), pc_disp_msg);
    pc_disp_msg.header.frame_id = "map";

    pc_display_pub.publish(pc_disp_msg);
}

void SemanticFusion::publishBoWPTestPointcloud() {
    pcl::PointCloud<pcl::PointXYZL> labelled_pc;

    // Add cube 1
    int points_l0 = 0;
    int points_l1 = 0;

    float step = 0.01;
    for (float x = -1.0f; x <= 1.0f; x += step) {
        for (float y = -0.1f; y <= 0.1f; y += step) {
            for (float z = -0.1f; z <= 0.1f; z += step) {
                pcl::PointXYZL p;
                p.x = x; p.y = y; p.z = z;
                p.label = 5;

                labelled_pc.push_back(p);

                points_l0++;
            }
        }
    }

    // Add cube 2
    for (float x = -0.1f; x <= 0.1f; x += step) {
        for (float y = -0.1f; y <= 0.1f; y += step) {
            for (float z = -0.1f; z <= 0.1f; z += step) {
                pcl::PointXYZL p;
                p.x = x; p.y = y; p.z = z + 0.3f;
                p.label = 6;

                labelled_pc.push_back(p);

                points_l1++;
            }
        }
    }

    ROS_INFO_STREAM("Points of label 0: " << points_l0 << ", label 1: " << points_l1);

    sensor_msgs::PointCloud2 pc_fused_msg;
    pcl::toROSMsg(labelled_pc, pc_fused_msg);
    pc_fused_msg.header.frame_id = "map";
    fused_pc_pub.publish(pc_fused_msg);
}

void SemanticFusion::loadAndPublishLabelledPointcloud(string path) {
    pcl::PointCloud<pcl::PointXYZL> labelled_pc;

    pcl::io::loadPLYFile(path, labelled_pc);

    ROS_INFO_STREAM("Loaded labelled pointcoud with " << labelled_pc.points.size());

    // DEBUG:: Build XYZRGB pointcloud
    cv::Mat label2bgr = cv::imread("/home/albert/rp_data/sun.png", CV_LOAD_IMAGE_COLOR);

    pcl::PointCloud<pcl::PointXYZRGB> label_rgb_pc;
    for(int i = 0; i < labelled_pc.points.size(); i++) {
        pcl::PointXYZRGB p;
        p.x = labelled_pc.points[i].x;
        p.y = labelled_pc.points[i].y;
        p.z = labelled_pc.points[i].z;

        if(labelled_pc.points[i].label > 37){
            p.b = 0;
            p.g = 0;
            p.r = 0;
        }else{
            cv::Vec3b color = label2bgr.at<cv::Vec3b>(cv::Point(labelled_pc.points[i].label+1, 0));
            p.b = color.val[0];
            p.g = color.val[1];
            p.r = color.val[2];
        }

        label_rgb_pc.push_back(p);
    }

    displayPointcloud(label_rgb_pc);

    sensor_msgs::PointCloud2 pc_fused_msg;
    pcl::toROSMsg(labelled_pc, pc_fused_msg);
    pc_fused_msg.header.frame_id = "map";
    fused_pc_pub.publish(pc_fused_msg);
}

void SemanticFusion::loadAndPublishMonoClassPointcloud(string path) {
    int class_id = 25;

    pcl::PointCloud<pcl::PointXYZL> labelled_pc;
    pcl::io::loadPLYFile(path, labelled_pc);

    ROS_INFO_STREAM("Loaded labelled pointcoud with " << labelled_pc.points.size());

    // DEBUG:: Build XYZRGB pointcloud
    cv::Mat label2bgr = cv::imread("/home/albert/rp_data/sun.png", CV_LOAD_IMAGE_COLOR);

    pcl::PointCloud<pcl::PointXYZRGB> label_rgb_pc;
    for(int i = 0; i < labelled_pc.points.size(); i++) {
        if(labelled_pc.points[i].label != class_id){
            continue;
        }

        pcl::PointXYZRGB p;
        p.x = labelled_pc.points[i].x;
        p.y = labelled_pc.points[i].y;
        p.z = labelled_pc.points[i].z;

        if(labelled_pc.points[i].label > 37){
            p.b = 0;
            p.g = 0;
            p.r = 0;
        }else{
            cv::Vec3b color = label2bgr.at<cv::Vec3b>(cv::Point(labelled_pc.points[i].label+1, 0));
            p.b = color.val[0];
            p.g = color.val[1];
            p.r = color.val[2];
        }

        label_rgb_pc.push_back(p);
    }

    displayPointcloud(label_rgb_pc);

    pcl::PointCloud<pcl::PointXYZL> labelled_filtered_pc;
    for(int i = 0; i < labelled_pc.points.size(); i++) {
        if(labelled_pc.points[i].label != class_id){
            continue;
        }

        labelled_filtered_pc.points.push_back(labelled_pc.points[i]);
    }

    sensor_msgs::PointCloud2 pc_fused_msg;
    pcl::toROSMsg(labelled_filtered_pc, pc_fused_msg);
    pc_fused_msg.header.frame_id = "map";
    fused_pc_pub.publish(pc_fused_msg);
}


// heigh = 480, width = 640
// [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
//XTION
// [ 570.34222412109375, 0., 319.5, 0., 570.34222412109375, 239.5, 0., 0., 1. ]

/*
 # Given a 3D point [X Y Z]', the projection (x, y) of the point onto
#  [u v w]' = P * [X Y Z 1]'
#         x = u / w
#         y = v / w

for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = pc1->begin(); it != pc1->end(); it++){
    //cout << it->x << ", " << it->y << ", " << it->z << endl;
}
*/

int main(int argc, char **argv)
{
    ros::init(argc, argv, "fusion_node");
    SemanticFusion sem_fusion;
    ros::param::get("debug", sem_fusion.debug_mode);

    //string base_path = "/home/albert/Desktop/room_test_processed/kitchen_with_floor/";
    string base_path;
    ros::param::get("dir", base_path);
    if(base_path.back() != '/') base_path += '/';

    bool load_existing = false;
    ros::param::get("load_labelled", load_existing);

    if(load_existing){
        string pc_path = base_path + "labelled_cloud.ply";
        sem_fusion.loadAndPublishLabelledPointcloud(pc_path);
    }else{
        pcl::PointCloud<pcl::PointXYZRGB> pc;
        vector<Eigen::Matrix4f> poses;
        vector<string> images;

        sem_fusion.loadPlaceData(base_path, pc, poses, images);
        ros::Duration(1.0).sleep();
        sem_fusion.createFusedSemanticMap(base_path, pc, poses, images);
    }

    ros::spin();
    return 0 ;
}
