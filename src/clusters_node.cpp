// ROS
#include <ros/ros.h>

// ROS messages
#include <std_msgs/Int16.h>
#include <std_msgs/Float64.h>
#include "std_msgs/String.h"
#include "rp_semantic/Cluster.h"
#include "rp_semantic/LabelClusters.h"
#include "rp_semantic/Frame.h"
#include "rp_semantic/BoWP.h"
#include "rp_semantic/BoWPDescriptors.h"

// Headers C_plus_plus
#include <iostream>
#include <string>
#include <csignal>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

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
// Struct { rp_semantic::float64 rp } // later create a structure of it 


class ClustersPointClouds{

private:
    bool visualize_clusters;

    int num_labels ;
    int min_cluster_size ;
    int max_cluster_size ;
    float cluster_tolerance ;
    ros::NodeHandle cl_handle;            // node handler  
    ros::Subscriber segnet_msg_sub ;      // SegNet Message Subscriber
    ros::Subscriber pointCloud2_msg_sub ; // PointCloud2 message Subscriber
    ros::Publisher clusters_msg_pub ;    // Clusters node Message Publisher
    ros::Publisher descriptors_msg_pub ;  // Clusters node descriptors message

public:

    ClustersPointClouds();
    void frameCallback(const rp_semantic::Frame msg);
    void clusteringPointcloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud , std::vector<rp_semantic::Cluster> &clusters);
    void bowCallback(const sensor_msgs::PointCloud2 pc_msg );
    void computeBoW( const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_in,  std::vector<float> &boW_descriptor );
    void computeBoWP(const std::vector<rp_semantic::Cluster> clusters, std::vector<float> &boWP_descriptor );
};



ClustersPointClouds::ClustersPointClouds(){

    visualize_clusters = true;
    // Initialization of variables
    num_labels = 37 ;

    min_cluster_size = -1 ;
    max_cluster_size = -1 ;
    cluster_tolerance = -1 ;

    ros::param::get("rp_semantic/clusters_node/num_labels", num_labels) ; // we can optimize it later 
    ros::param::get("rp_semantic/clusters_node/cluster_tolerance", cluster_tolerance) ; // we can optimize it later 
    ros::param::get("rp_semantic/clusters_node/min_cluster_size", min_cluster_size) ; // we can optimize it later 
    ros::param::get("rp_semantic/clusters_node/max_cluster_size", max_cluster_size) ; // we can optimize it later 


    // Subscribers  // Topic subscribe to : rp_semantic/labels_pointcloud
    segnet_msg_sub = cl_handle.subscribe("/semantic_frame", 10, &ClustersPointClouds::frameCallback , this ); // Subscriber
    pointCloud2_msg_sub = cl_handle.subscribe("rp_semantic/semantic_fused_pc", 10, &ClustersPointClouds::bowCallback , this ); // Subscriber

    // Publishers
    clusters_msg_pub = cl_handle.advertise<rp_semantic::LabelClusters>("rp_semantic/labels_clusters", 10); // Publisher  
    descriptors_msg_pub = cl_handle.advertise<rp_semantic::BoWPDescriptors>("rp_semantic/place_descriptors", 10); // Publisher
}


void ClustersPointClouds::frameCallback(const rp_semantic::Frame msg){

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL> );          // Creating Pointcloud structured of type pcl::PointXYZL
    pcl::PCLPointCloud2 pcl_pc2;
    cv::Mat label_img;

    // Unpack sensor_msgs/Pointcloud2 in msg.raw_pointcloud into a pointcloud
    pcl_conversions::toPCL( msg.raw_pointcloud , pcl_pc2 );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in( new pcl::PointCloud<pcl::PointXYZRGB> );  // Pointcloud in of type PointXYZRGB
    pcl::fromPCLPointCloud2(pcl_pc2,*cloud_in);

    // Unpack sensor_msgs/Image into cv::Mat type
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg.label, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR_STREAM( ": cv_bridge exception, " << e.what());
        return;
    }

    // Convert image from chars to floats
    cv_ptr->image.convertTo(label_img, CV_8UC1);

    // Resizing label image to the size of cloud cloud_in->width and cloud_in->height
    cv::resize(label_img , label_img, cv::Size(cloud_in->width, cloud_in->height), 0 , 0 , cv::INTER_NEAREST ) ;


    // Convert structured XYZRGB pointcloud + cv::Mat into dense XYZL pointcloud
    ROS_DEBUG_STREAM( "label_img.cols  = "<< label_img.cols   << endl);
    ROS_DEBUG_STREAM( "label_img.rows  = "<< label_img.rows   << endl);
    ROS_DEBUG_STREAM( "Cloud In width  = "<< cloud_in->width  << endl);
    ROS_DEBUG_STREAM( "Cloud in height = "<< cloud_in->height << endl);

    int pc_idx  = 0 ; // Index of the labelled rgb image row wise .
    for(int x = 0 ; x < label_img.cols ; x++ ){
        for( int y = 0 ; y < label_img.rows ; y++ ){
            pc_idx = x + label_img.cols*y;  // i = x + width*y;

            if(  std::isnan(cloud_in->points[pc_idx].x)  && std::isnan(cloud_in->points[pc_idx].y) && std::isnan(cloud_in->points[pc_idx].z ) ){
                continue;
            }

            pcl::PointXYZL p;

            p.x = cloud_in->points[pc_idx].x ;
            p.y = cloud_in->points[pc_idx].y ;
            p.z = cloud_in->points[pc_idx].z ;
            p.label = label_img.at<uchar>(y,x) ;
            cloud->points.push_back(p);
        }
    }

    ROS_DEBUG_STREAM("finsished making label cloud" << endl);
    //msg_out.clusters.push_back(cluster);
    rp_semantic::LabelClusters msg_out;

    // Clustring 
    std::vector<rp_semantic::Cluster> clusters ;
    clusteringPointcloud( cloud , clusters);

    // Filling LabelClusters message 
    std::copy( clusters.begin(), clusters.end() , back_inserter(msg_out.clusters) ) ; //back_inserter(msg_out.clusters);
    msg_out.node_id  = msg.node_id ;
    msg_out.labels   = msg.label ; // TODO : Need to correct naming label to labels in Frame.msg
    msg_out.raw_rgb  = msg.raw_rgb;
    msg_out.raw_pointcloud = msg.raw_pointcloud ;
    clusters_msg_pub.publish(msg_out) ;

    cout<< "frameCallback function ended sucessfully " <<endl ;

} // end of callback function



void ClustersPointClouds::bowCallback(sensor_msgs::PointCloud2 pc_msg ){

    // Initialize pointcloud for msg to pcl conversion
    pcl::PCLPointCloud2 pcl_pc2;

    // Unpack sensor_msgs/Pointcloud2 in pc_msg.raw_pointcloud into a pointcloud
    pcl_conversions::toPCL( pc_msg , pcl_pc2 );
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_in( new pcl::PointCloud<pcl::PointXYZL> );  // Pointcloud in of type PointXYZRGB
    pcl::fromPCLPointCloud2(pcl_pc2,*cloud_in);

    ROS_INFO_STREAM("Finished conversion to pcl pointcloud");

    // Compute boW of the clusters  
    std::vector<float>  boW( num_labels ,0 ) ;
    computeBoW( cloud_in, boW );

    // Compute clusters of the Point cloud 
    std::vector<rp_semantic::Cluster>  clusters;
    clusteringPointcloud( cloud_in, clusters );

    //
    if(visualize_clusters){
        rp_semantic::LabelClusters lc_msg;

        std::copy(clusters.begin(), clusters.end(), back_inserter(lc_msg.clusters));

        clusters_msg_pub.publish(lc_msg);
    }

    // Compute boWP of the cluster
    std::vector<float>  boWp(num_labels*num_labels, 0) ;
    computeBoWP( clusters, boWp );

    // Publishing BoW and boWP 
    rp_semantic::BoWPDescriptors msg_descriptors;

    // Filling the boW and boWP descripto message
    std::copy( boW.begin(), boW.end() , back_inserter(msg_descriptors.bow) ) ;
    std::copy( boWp.begin(), boWp.end() , back_inserter(msg_descriptors.bowp) ) ;

    descriptors_msg_pub.publish( msg_descriptors );

    cout<< "bowCallback function ended sucessfully " <<endl ;

} // end of bowcallback function



void ClustersPointClouds::clusteringPointcloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud , std::vector<rp_semantic::Cluster> &clusters ){
    // Extracting cluster for each labels
    for(int cl = 0 ; cl < num_labels ; cl ++ ){
        if(cl == 0 || cl == 1 || cl == 21   )
            continue;

        // Create the filtering object
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_labels(new pcl::PointCloud<pcl::PointXYZ> ); // Filtered point cloud of type pcl::PointXYZL

        for (size_t i = 0; i < cloud->points.size (); ++i)
        {
            if(cloud->points[i].label == cl){
                pcl::PointXYZ p(cloud->points[i].x , cloud->points[i].y, cloud->points[i].z ) ;
                cloud_filtered_labels->points.push_back(p) ;
            }
        }

        if( cloud_filtered_labels->points.empty()){
            ROS_DEBUG_STREAM( "No points of label  " << cl << ": " << std::endl);
            continue ;
        }


        ROS_DEBUG_STREAM("Clustering label  " << cl << ": " << std::endl);

        // Creating the KdTree object for the search method of the extraction
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud (cloud_filtered_labels);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance (cluster_tolerance); // 2cm
        ec.setMinClusterSize (min_cluster_size);
        ec.setMaxClusterSize (max_cluster_size);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cloud_filtered_labels);
        ec.extract (cluster_indices);

        int j = 0;
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {

            //find mean
            pcl::PointXYZ p(0,0,0);
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
                p.x += cloud_filtered_labels->points[*pit].x ;
                p.y += cloud_filtered_labels->points[*pit].y ;
                p.z += cloud_filtered_labels->points[*pit].z ;
            }

            p.x /= (float) it->indices.size();
            p.y /= (float) it->indices.size();
            p.z /= (float) it->indices.size();

            //compute dist
            float dist_max = 0 ;
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
                float delta_x = p.x - cloud_filtered_labels->points[*pit].x;
                float delta_y = p.y - cloud_filtered_labels->points[*pit].y;
                float delta_z = p.z - cloud_filtered_labels->points[*pit].z;

                float dist = delta_x * delta_x + delta_y*delta_y + delta_z*delta_z ;
                if (dist > dist_max){
                    dist_max = dist ;
                }
            }

            rp_semantic::Cluster cluster;
            cluster.label = cl ;
            cluster.x = p.x ;
            cluster.y = p.y ;
            cluster.z = p.z ;
            cluster.radius = std::sqrt(dist_max);

            clusters.push_back(cluster);
        } //cl clusters
    } // cl
}// end Cluster Function


void ClustersPointClouds::computeBoW(const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_in, std::vector<float> &boW_descriptor ){

    for( int i = 0 ; i < cloud_in->points.size() ; i++ ){
        if( cloud_in->points[i].label > 36) continue;

        boW_descriptor[ cloud_in->points[i].label ]++ ;
    }


}// end bow Function


void ClustersPointClouds::computeBoWP( const std::vector<rp_semantic::Cluster> clusters, std::vector<float> &boWP_descriptor){


    int num_cluster = clusters.size() ;

    ROS_INFO_STREAM("Making BoWP with " << num_cluster << " clusters");

    vector<float>  array1D_init(num_labels ,0);
    vector<vector<float> >  cluster_pairs( num_labels , array1D_init );

    // Computing BOWP histogram by using eucleadean distance
    for( int i = 0 ; i < num_cluster ; ++i ){
        for( int j = i ; j < num_cluster ; ++j ){
            if (i != j ){
                float delta_x = clusters[i].x - clusters[j].x ;
                float delta_y = clusters[i].y - clusters[j].y ;
                float delta_z = clusters[i].z - clusters[j].z ;
                float dist = std::sqrt( delta_x * delta_x + delta_y*delta_y + delta_z*delta_z );

                if( clusters[i].radius > dist )
                    cluster_pairs[ clusters[i].label ][ clusters[j].label] += 1; // Incrementing previous value at that index of that vector by 1

                if( clusters[j].radius > dist )
                    cluster_pairs[ clusters[j].label ][ clusters[i].label] += 1; // Incrementing previous value at that index of that vector by 1

            } //end if main
        } // end i loop
    } // end j loop


    // Converting cluster_pairs into array1D
    for (int row = 0; row < num_labels; row++)
        for (int col = 0; col < num_labels ; col++)
            boWP_descriptor[ row*num_labels+col ] = cluster_pairs[row][col] ;

}// end boWP Function    


// Main clusters_node
int main(int argc, char **argv)
{
    ros::init( argc, argv, "clusters_node" );
    ClustersPointClouds cluster_point_cloud;
    ros::spin();
    return 0 ;
}
