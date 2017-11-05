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

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>

using namespace std;

class ClustersPointClouds{

private:

    ros::NodeHandle cl_handle;         // node handler
    ros::Subscriber segnet_msg_sub ;    // SegNet Message Subscriber
    ros::Publisher clusters_msg_pub ;   // Clusters node Message Publisher

public:

    ClustersPointClouds(); 
    void frameCallback(const rp_semantic::Frame &msg);       
    //ProcessInputPointCloud(const pcl::PointCloud<pcl::PointXYZ> cloud_in , pcl::PointCloud<pcl::PointXYZL>::Ptr cloud, pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud_filtered ) ;       // Create Structured Point cloud 
    //Euclidean_ClusterExtraction(pcl::PointCloud<pcl::PointXYZ> &cloud) ; // Eucleadean Clusters of Point cloud 
};


ClustersPointClouds::ClustersPointClouds(){  
 
    // Subscribers and Publisher
    segnet_msg_sub = cl_handle.subscribe("rp_semantic/labels_pointcloud", 1000, &ClustersPointClouds::frameCallback , this ); // Subscriber
    clusters_msg_pub = cl_handle.advertise<rp_semantic::LabelClusters>("rp_semantic/labels_clusters", 1000); // Publisher
}

void ClustersPointClouds::frameCallback(const rp_semantic::Frame &msg){

    pcl::PointCloud<pcl::PointXYZRGB> cloud_in ; // Pointcloud in of type PointXYZRGB
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud ; // Creating Pointcloud structured of type pcl::PointXYZL
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_filtered ; // Filtered point cloud of type pcl::PointXYZL

    // 






}


/*
ClustersPointClouds::ProcessInputPointCloud(const pcl::PointCloud<pcl::PointXYZ> cloud_in ) { // pcl::PointCloud<pcl::PointXYZL>::Ptr cloud, pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud_filtered  ){
// Fill in the cloud data
    cloud.width    = 512  ;
    cloud.height   = 512  ;
    cloud.is_dense = true ;
    cloud.points.resize (cloud.width * cloud.height);

    for (size_t i = 0; i < cloud->points.size (); ++i)
    {
        cloud->points[i].x = cloud_in->points[i].x;
        cloud->points[i].y = cloud_in->points[i].y;
        cloud->points[i].z = cloud_in->points[i].z;
        cloud->points[i].label = cloud_in->points[i].label;
    }

    std::cerr << "Cloud before filtering: " << std::endl;
    for (size_t i = 0; i < cloud->points.size (); ++i){
        std::cerr << "    " << cloud->points[i].x << " " 
        << cloud->points[i].y << " " 
        << cloud->points[i].z << std::endl;
    }

    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZL> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("label");
    pass.setFilterLimits (0.9, 1.1);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*cloud_filtered);

    std::cerr << "Cloud after filtering: " << std::endl;
    for (size_t i = 0; i < cloud_filtered->points.size (); ++i){
        std::cerr << "    " << cloud_filtered->points[i].x << " " 
        << cloud_filtered->points[i].y << " " 
        << cloud_filtered->points[i].z << std::endl ;
    }


}


ClustersPointClouds::Euclidean_ClusterExtraction(pcl::PointCloud<pcl::PointXYZ> &cloud){






  }
*/



int main(int argc, char **argv)
{
     ros::init(argc, argv, "clusters_node");
     ClustersPointClouds cluster_point_cloud;
     ros::spin();
     return 0 ;
}
