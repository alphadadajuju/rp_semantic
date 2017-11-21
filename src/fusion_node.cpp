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
#include "rp_semantic/Frame.h"

#include "rp_semantic/RGB2LabelProb.h"

// Headers C_plus_plus
#include <iostream>
#include <fstream>
#include <string>
#include <csignal>
#include <sstream>
#include <cmath>
#include <limits>
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

    bool loadPlaceData(const string &base_path, pcl::PointCloud<pcl::PointXYZRGB> &pc, vector<Eigen::Matrix4f> &poses, vector<string> &images);
    void createFusedSemanticMap(const pcl::PointCloud<pcl::PointXYZRGB> &pc, const vector<Eigen::Matrix4f> &poses, const vector<string> &images);
};


SemanticFusion::SemanticFusion(){

    pc_display_pub = nh.advertise<sensor_msgs::PointCloud2>("rp_semantic/fusion_pointcloud", 10); // Publisher
    marker_pub = nh.advertise<visualization_msgs::Marker>("rp_semantic/camera_pose", 10); // Publisher
    fused_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("rp_semantic/semantic_fused_pc", 10); // Publisher
    image_pub = nh.advertise<sensor_msgs::Image>("rp_semantic/camera_image", 10);

    segnet_client = nh.serviceClient<rp_semantic::RGB2LabelProb>("rgb_to_label_prob");

    // Initialization of variables
    //num_labels = 37 ;
    //ros::param::get("rp_semantic/clusters_node/num_labels", num_labels) ; // we can optimize it later 


    // Subscribers and Publisher // Topic subscribe to : rp_semantic/labels_pointcloud
    //segnet_msg_sub = nh.subscribe("/rp_semantic/labels_pointcloud", 10, &SemanticFusion::frameCallback , this ); // Subscriber
    //clusters_msg_pub = nh.advertise<rp_semantic::LabelClusters>("rp_semantic/labels_clusters", 10); // Publisher
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

void SemanticFusion::createFusedSemanticMap(const pcl::PointCloud<pcl::PointXYZRGB> &pc,
                                            const vector<Eigen::Matrix4f> &poses, const vector<string> &images) {

    // Initialize label probability structure
    vector<float> label_prob_init(37, 1.0f/37.0f);
    std::vector<vector<float> > label_prob(pc.points.size(), label_prob_init);

    // Initialize other objects
    pcl::FrustumCulling<pcl::PointXYZRGB> fc;
    fc.setInputCloud ( pc.makeShared() );
    // Set frustrum according to Kinect specifications
    fc.setVerticalFOV (46.6);
    fc.setHorizontalFOV (58.5);
    fc.setNearPlaneDistance (0.8);
    fc.setFarPlaneDistance (8); //Should be 4m. but we bump it up a little

    // Initialization of P
    Eigen::Matrix4f P;
    P << 525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

    // For each pose & image
    cv::Mat label2bgr = cv::imread("/home/albert/rp_data/sun.png", CV_LOAD_IMAGE_COLOR);
    sensor_msgs::ImagePtr rgb2label_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", label2bgr).toImageMsg();
    image_pub.publish(rgb2label_msg);

    for (int i = 0; i < poses.size(); ++i) {
        ROS_INFO_STREAM_THROTTLE(3, "Processing node " << i << "/" << poses.size());

        if(!ros::ok()) break;

        // FrustrumCulling from poses-> indices of visible points
        fc.setCameraPose(poses[i]);
        std::vector<int> inside_indices; // Indices of points in pc_gray inside camera frustrum at pose[i]
        fc.filter(inside_indices);

        // SRV: request image labelling through segnet
        cv::Mat cv_img = cv::imread(images[i], CV_LOAD_IMAGE_COLOR);
        sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_img).toImageMsg();

        rp_semantic::RGB2LabelProb srv_msg;
        srv_msg.request.rgb_image = *image_msg.get();
        segnet_client.call(srv_msg);

        std_msgs::Float64MultiArray frame_label_probs(srv_msg.response.image_class_probability);
        int dim_stride_1 = frame_label_probs.layout.dim[1].stride;
        int dim_stride_2 = frame_label_probs.layout.dim[2].stride;


        Eigen::Matrix4f rviz2cv = Eigen::Matrix4f::Identity(4,4);
        Eigen::Matrix3f r;
        r = Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitZ())
            * Eigen::AngleAxisf( -0.5f*M_PI, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf( 0.5f*M_PI, Eigen::Vector3f::UnitX());
        rviz2cv.topLeftCorner(3,3) = r;

        // Get camera matrix from poses_i and K
        Eigen::Matrix4f pose_inv = poses[i].inverse();
        Eigen::Matrix4f world2pix = P * rviz2cv * pose_inv;

        //Create "Z buffer" emulator
        bool has_been_projected[640][480] = { false };

        for(std::vector<int>::iterator it = inside_indices.begin(); it != inside_indices.end(); it++){
            // Backproject points using camera matrix (discard out of range)
            Eigen::Vector4f point_w(pc.points[*it].x, pc.points[*it].y, pc.points[*it].z, 1.0);
            Eigen::Vector4f point_px = world2pix * point_w;
            if(point_px(2) == 0){ point_px(2) +=1; } // Adapt homogenous for 4x4 efficient multiplication

            int pix_x = std::round(point_px(0)/point_px(2));
            int pix_y = std::round(point_px(1)/point_px(2));

            if(pix_x < 0 || pix_x >= 640 || pix_y < 0 || pix_y >= 480)
                continue;

            if(!has_been_projected[pix_x][pix_y]){
                // Get associated distributions, multiply distributions together and renormalize
                //multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]

                // Multiply each element of the distribution and Get the sum of the final distribution
                float dist_sum = 0.0;
                for (int cl = 0; cl < 37; ++cl) {
                    label_prob[*it][cl] *= frame_label_probs.data[dim_stride_1*cl + dim_stride_2*pix_x + pix_y];
                    dist_sum += label_prob[*it][cl];
                }

                // Divide each element by the sum
                for (int cl = 0; cl < 37; ++cl) {
                    label_prob[*it][cl] *= 1/dist_sum;
                }
            }
        }

        // Build XYZL pointcloud
        pcl::PointCloud<pcl::PointXYZL> labelled_pc;
        for(int i = 0; i < pc.points.size(); i++) {
            pcl::PointXYZL p;
            p.x = pc.points[i].x;
            p.y = pc.points[i].y;
            p.z = pc.points[i].z;

            uint16_t max_label = 0;
            float max_label_prob = 0.0;
            for (uint16_t cl = 0; cl < 37; ++cl) {
                if(label_prob[i][cl] > max_label_prob){
                    max_label = cl;
                    max_label_prob = label_prob[i][cl];
                }
            }

            p.label = max_label;

            labelled_pc.push_back(p);
        }


        // DEBUG:: Build XYZRGB pointcloud
        pcl::PointCloud<pcl::PointXYZRGB> label_rgb_pc;
        for(int i = 0; i < labelled_pc.points.size(); i++) {
            pcl::PointXYZRGB p;
            p.x = labelled_pc.points[i].x;
            p.y = labelled_pc.points[i].y;
            p.z = labelled_pc.points[i].z;

            cv::Vec3b color = label2bgr.at<cv::Vec3b>(cv::Point(labelled_pc.points[i].label, 0));
            p.b = color.val[0];
            p.g = color.val[1];
            p.r = color.val[2];

            label_rgb_pc.push_back(p);
        }

        displayPointcloud(label_rgb_pc);
        displayCameraMarker(poses[i]);
    }// For each pose



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

    marker.scale.x = 1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    marker_pub.publish( marker );
}

void SemanticFusion::displayPointcloud(const pcl::PointCloud<pcl::PointXYZRGB> &pc) {
    sensor_msgs::PointCloud2 pc_disp_msg;
    pcl::toROSMsg(*pc.makeShared(), pc_disp_msg);
    pc_disp_msg.header.frame_id = "map";

    pc_display_pub.publish(pc_disp_msg);
}

// heigh = 480, width = 640
// [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]

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

    string base_path = "/home/albert/Desktop/kinect2/";
    pcl::PointCloud<pcl::PointXYZRGB> pc;
    vector<Eigen::Matrix4f> poses;
    vector<string> images;

    sem_fusion.loadPlaceData(base_path, pc, poses, images);
    ros::Duration(1.0).sleep();
    sem_fusion.createFusedSemanticMap(pc, poses, images);

    ros::spin();
    return 0 ;
}
