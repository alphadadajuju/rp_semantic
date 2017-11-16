// ROS
#include <ros/ros.h>

// ROS messages
#include <std_msgs/Int16.h>
#include "std_msgs/String.h"
#include "rp_semantic/Cluster.h"
#include "rp_semantic/LabelClusters.h"
#include "rp_semantic/Frame.h"

// Headers C_plus_plus
#include <iostream>
#include <string>
#include <csignal>
#include <sstream>
#include <cmath>
#include <limits>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
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

class SemanticFusion{

private:
    ros::NodeHandle nh;         // node handler
    ros::Publisher pc_display_pub ;   // Clusters node Message Publisher

public:
    SemanticFusion();
    void testFrustrum();
    void testworld2pixel();
    //void frameCallback(const rp_semantic::Frame &msg);
};


SemanticFusion::SemanticFusion(){

    pc_display_pub = nh.advertise<sensor_msgs::PointCloud2>("rp_semantic/fusion_pointcloud", 10); // Publisher

    // Initialization of variables
    //num_labels = 37 ;
    //ros::param::get("rp_semantic/clusters_node/num_labels", num_labels) ; // we can optimize it later 


    // Subscribers and Publisher // Topic subscribe to : rp_semantic/labels_pointcloud
    //segnet_msg_sub = nh.subscribe("/rp_semantic/labels_pointcloud", 10, &SemanticFusion::frameCallback , this ); // Subscriber
    //clusters_msg_pub = nh.advertise<rp_semantic::LabelClusters>("rp_semantic/labels_clusters", 10); // Publisher
}




void SemanticFusion::testFrustrum(){
    // Create uniformly spreaded pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc1(new pcl::PointCloud<pcl::PointXYZ>);

    float step = 0.1; //10cm.

    for (float i = -15; i < 15; i += step)
        for (float j = -15; j < 15; j += step)
            for (float k = -15; k < 15; k += step)
                pc1->points.push_back(pcl::PointXYZ(i,j,k));

    // Filter points with frustrum object
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc2(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::FrustumCulling<pcl::PointXYZ> fc;
    fc.setInputCloud (pc1);
    fc.setVerticalFOV (46.6);
    fc.setHorizontalFOV (58.5);
    fc.setNearPlaneDistance (0.8);
    fc.setFarPlaneDistance (4);
    fc.filter(*pc2);

    //Send
    sensor_msgs::PointCloud2 pc_disp_msg;
    pcl::toROSMsg(*pc2, pc_disp_msg);
    //pcl_conversions::copyPCLPointCloud2MetaData(pcl_pc2, pc_disp_msg);
    pc_disp_msg.header.frame_id = "map";

    pc_display_pub.publish(pc_disp_msg);
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

    sem_fusion.testworld2pixel();

    ros::spin();
    return 0 ;
}
