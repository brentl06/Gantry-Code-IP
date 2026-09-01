/*
 * @Author: Ryoma Liu -- ROBOLAND 
 * @Date: 2021-11-21 21:58:00 
 * @Last Modified by: Ryoma Liu
 * @Last Modified time: 2021-11-28 14:38:09
 */

#include "proxy/upperproxy.h"
#include "controller/trajectories_parser.h"


/**
 * upperproxy - class to collect robot's information and trajectories from path
 * planning and decision making part. 
 * agile taur.
 */

namespace turtle_namespace{
namespace control{

upperproxy::upperproxy(std::string name) : Node(name){
    std::cout<<"Traveler Upper Proxy established"
                <<std::endl;
    GUI_publisher = this->create_publisher<std_msgs::msg::Float64MultiArray>
        ("/drag_times", 10);
    trajectory_complete_publisher = this->create_publisher<std_msgs::msg::Bool>
        ("/trajectory_complete", 10);
    GUI_subscriber = this->create_subscription<std_msgs::msg::Float64MultiArray>
        ("/Gui_information", 10, std::bind(&upperproxy::handle_gui, this, _1));
    trajectory_subscriber = this->create_subscription<std_msgs::msg::Float64MultiArray>
        ("/trajectory_points", 10, std::bind(&upperproxy::handle_trajectory_points, this, _1)); 
    RCLCPP_INFO(this->get_logger(), "Subscribed to /trajectory_points (Float64MultiArray)");
    std::cout << "[upperproxy] ctor reached, trajectory sub created\n";

}

void upperproxy::handle_trajectory_points(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
    std::cout << "Received waypoints:\n";
    // Save into shared state so other parts of the system can use them later
    auto &td = turtle_inter_.traj_data;
    td.waypoints_x.clear();
    td.waypoints_y.clear();
    td.waypoints_v.clear();
    td.num_waypoints = 0;

    for (size_t i = 0; i + 2 < msg->data.size(); i += 3) {
        const double x = msg->data[i];
        const double y = msg->data[i + 1];
        const double v = msg->data[i + 2];
        std::cout << " (" << x << ", " << y << "), vel: " << v << std::endl;
        td.waypoints_x.push_back(static_cast<float>(x));
        td.waypoints_y.push_back(static_cast<float>(y));
        td.waypoints_v.push_back(static_cast<float>(v));
        td.num_waypoints++;
    }
    
    td.trajectory_version++;
    
    RCLCPP_INFO(this->get_logger(), "Stored %d waypoints", td.num_waypoints);
    
    // If not currently running, auto-start the trajectory
    if (turtle_inter_.turtle_gui.start_flag == 0) {
        std::cout << "Auto-starting trajectory execution\n";
        turtle_inter_.turtle_gui.start_flag = 1;
    }
}


void upperproxy::handle_gui
    (const std_msgs::msg::Float64MultiArray::SharedPtr msg){
        // int len = msg->data.size();
        turtle_inter_.turtle_gui.start_flag = msg->data[0]; 
        turtle_inter_.turtle_gui.drag_traj = msg->data[1];
        // turtle_inter_.traj_data.lateral_angle_range = msg->data[2]; // sweeping range
        // turtle_inter_.traj_data.drag_speed = msg->data[3]; // insertion
        // turtle_inter_.traj_data.wiggle_time = msg->data[4]; //penetration velocity
        // turtle_inter_.traj_data.servo_speed = msg->data[5]; // sweeping velocity 
        // turtle_inter_.traj_data.extraction_angle = msg->data[6]; // extraction velocity
        // turtle_inter_.traj_data.wiggle_frequency = msg->data[7]; // swing velocity
        // turtle_inter_.traj_data.insertion_depth = msg->data[8]; // 
        // turtle_inter_.traj_data.wiggle_amptitude = msg->data[9];
        
    }

void upperproxy::UpdateGuiCommand(turtle& turtle_){
    turtle_.turtle_gui = turtle_inter_.turtle_gui;
    turtle_.traj_data = turtle_inter_.traj_data;
}

void upperproxy::PublishTrajectoryComplete(){
    auto &traj_parser = turtle_namespace::control::TrajectoriesParser::getTrajParser();
    const bool is_complete = traj_parser.trajComplete();
    if (is_complete != last_traj_complete_) {
        auto message = std_msgs::msg::Bool();
        message.data = is_complete;
        trajectory_complete_publisher->publish(message);
        last_traj_complete_ = is_complete;
    }
}
void upperproxy::PublishStatusFeedback(turtle& turtle_){
    if(turtle_.turtle_gui.status_update_flag == true){
        auto message = std_msgs::msg::Float64MultiArray();
        // std::cout <<  message.data[message.data.size() - 1] << std::endl;
        GUI_publisher->publish(message);
        turtle_.turtle_gui.status_update_flag = false;
    }
    
}

} //namespace control
} //namespace turtle_namespace
