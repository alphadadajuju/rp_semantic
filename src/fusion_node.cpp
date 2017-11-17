// ROS
#include <ros/ros.h>

// ROS messages
#include <std_msgs/Int16.h>
#include "std_msgs/String.h"
#include "visualization_msgs/Marker.h"
#include "rp_semantic/Cluster.h"
#include "rp_semantic/LabelClusters.h"
#include "rp_semantic/Frame.h"


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
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



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

public:
    SemanticFusion();
    bool loadPlaceData(const string &base_path, pcl::PointCloud<pcl::PointXYZRGB> &pc, vector<Eigen::Matrix4f> &poses, vector<string> &images);
    void createFusedSemanticMap(const pcl::PointCloud<pcl::PointXYZRGB> &pc, const vector<Eigen::Matrix4f> &poses, const vector<string> &images);
    void displayCameraMarker(Eigen::Matrix4f cam_pose);
    void displayPointcloud(const pcl::PointCloud<pcl::PointXYZRGB> &pc);

    //void frameCallback(const rp_semantic::Frame &msg);
    void testworld2pixel();
    void testFrustrum(const pcl::PointCloud<pcl::PointXYZRGB> &pc, const Eigen::Matrix4f &cam_pose);
};


SemanticFusion::SemanticFusion(){

    pc_display_pub = nh.advertise<sensor_msgs::PointCloud2>("rp_semantic/fusion_pointcloud", 10); // Publisher
    marker_pub = nh.advertise<visualization_msgs::Marker>("rp_semantic/camera_pose", 10); // Publisher


    // Initialization of variables
    //num_labels = 37 ;
    //ros::param::get("rp_semantic/clusters_node/num_labels", num_labels) ; // we can optimize it later 


    // Subscribers and Publisher // Topic subscribe to : rp_semantic/labels_pointcloud
    //segnet_msg_sub = nh.subscribe("/rp_semantic/labels_pointcloud", 10, &SemanticFusion::frameCallback , this ); // Subscriber
    //clusters_msg_pub = nh.advertise<rp_semantic::LabelClusters>("rp_semantic/labels_clusters", 10); // Publisher
}




void SemanticFusion::testFrustrum(const pcl::PointCloud<pcl::PointXYZRGB> &pc, const Eigen::Matrix4f &cam_pose){

    // Filter points with frustrum object
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc2(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::FrustumCulling<pcl::PointXYZRGB> fc;
    fc.setVerticalFOV (46.6);
    fc.setHorizontalFOV (58.5);
    fc.setNearPlaneDistance (0.8);
    fc.setFarPlaneDistance (4);

    fc.setInputCloud (pc.makeShared());
    fc.setCameraPose(cam_pose);
    fc.filter(*pc2);

    //Send
    displayPointcloud(*pc2);
}


void SemanticFusion::testworld2pixel(){
    // Create uniformly spreaded pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc1(new pcl::PointCloud<pcl::PointXYZ>);

    float step = 0.1; //10cm.

    for (float j = -5; j < 5; j += step)
        for (float k = -5; k < 5; k += step)
            pc1->points.push_back(pcl::PointXYZ(j,k,5));


    std::vector<int> inside_indices;

    pcl::FrustumCulling<pcl::PointXYZ> fc;
    fc.setInputCloud (pc1);
    fc.setVerticalFOV (46.6);
    fc.setHorizontalFOV (58.5);
    fc.setNearPlaneDistance (0.8);
    fc.setFarPlaneDistance (10);
    fc.filter(inside_indices);

    // Initialize intrinsic camera matrix from kinect camera_info
    Eigen::Matrix4f P;
    P << 525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

    Eigen::Matrix3f K;
    K << 525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0;

/*
         # Given a 3D point [X Y Z]', the projection (x, y) of the point onto
        #  the rectified image is given by:
        #  [u v w]' = P * [X Y Z 1]'
        #         x = u / w
        #         y = v / w
        #  This holds for both images of a stereo pair.
         */

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc2(new pcl::PointCloud<pcl::PointXYZRGB>);

    for(pcl::PointCloud<pcl::PointXYZ>::iterator it = pc1->begin(); it != pc1->end(); it++){
        Eigen::Vector3f point_world(it->x, it->y, it->z);

        Eigen::Vector3f point_pixel = K * point_world;
        point_pixel /= point_pixel(2);

        int pix_x = point_pixel(0);
        int pix_y = point_pixel(1);

        pcl::PointXYZRGB p;
        p.x = it->x;
        p.y = it->y;
        p.z = it->z;
        p.r = (uchar) std::min(std::max(0, (int) 255*pix_y/480), 255); //y
        p.g = 0; //(uchar) std::min(std::max(0, (int) 255*pix_x/640), 255); //255.0*point_pixel(0)/640.0; //x
        p.b = 0;

        pc2->points.push_back(p);
    }

    //Send
    sensor_msgs::PointCloud2 pc_disp_msg;
    pcl::toROSMsg(*pc2, pc_disp_msg);
    //pcl_conversions::copyPCLPointCloud2MetaData(pcl_pc2, pc_disp_msg);
    pc_disp_msg.header.frame_id = "map";

    while(ros::ok()){
        pc_display_pub.publish(pc_disp_msg);
        ros::Duration(0.5).sleep();
    }

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


    // For each pose & image
    for (int i = 0; i < poses.size(); ++i) {
        // FrustrumCulling from poses-> indices of visible points
        fc.setCameraPose(poses[i]);

        std::vector<int> inside_indices;
        fc.filter(inside_indices);

        // SRV: request image labelling through segnet

        // Get camera matrix from poses_i and K

        for (int j = 0; j < inside_indices.size(); ++j) {
            // Backproject points using camera matrix (discard out of range)

            // Get associated distributions, multiply distributions together and renormalize


        }

    }


}

/*
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


    // For each pose & image
    for (int i = 0; i < poses.size(); ++i) {
        // FrustrumCulling from poses-> indices of visible points
        fc.setCameraPose(poses[i]);

        std::vector<int> inside_indices;
        fc.filter(inside_indices);

        // SRV: request image labelling through segnet

        // Get camera matrix from poses_i and K

        for (int j = 0; j < inside_indices.size(); ++j) {
            // Backproject points using camera matrix (discard out of range)

            // Get associated distributions, multiply distributions together and renormalize


        }

    }


}
*/

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
#  the rectified image is given by:
#  [u v w]' = P * [X Y Z 1]'
#         x = u / w
#         y = v / w
#  This holds for both images of a stereo pair.

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

    //sem_fusion.displayPointcloud(pc);
    for (int i = 0; i < poses.size(); ++i) {
        sem_fusion.testFrustrum(pc, poses[i]);
        sem_fusion.displayCameraMarker(poses[i]);
        ros::Duration(0.6).sleep();
    }

    ros::spin();
    return 0 ;
}
