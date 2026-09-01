#include "can_driver/can_driver.hpp"

can_driver::can_driver() : Node("can_driver"),
                              //  socket_channel0_axis0_read_(odrive_can::Msg::MSG_ODRIVE_HEARTBEAT | odrive_can::AXIS::AXIS_0_ID,0, 0x7FF, 500),
                              //  socket_channel0_axis1_read_(odrive_can::Msg::MSG_ODRIVE_HEARTBEAT | odrive_can::AXIS::AXIS_1_ID,0, 0x7FF, 110000),
                              //  socket_channel1_axis0_read_(odrive_can::Msg::MSG_ODRIVE_HEARTBEAT | odrive_can::AXIS::AXIS_0_ID,1, 0x7FF, 500),
                              //  socket_channel1_axis1_read_(odrive_can::Msg::MSG_ODRIVE_HEARTBEAT | odrive_can::AXIS::AXIS_1_ID,1, 0x7FF, 110000),
                               socket_channel0_get_iq_0(odrive_can::Msg::MSG_GET_IQ | odrive_can::AXIS::AXIS_0_ID,0, 0x7FF, 500),
                               socket_channel0_get_iq_1(odrive_can::Msg::MSG_GET_IQ | odrive_can::AXIS::AXIS_1_ID,0, 0x7FF, 110000),
                               socket_channel1_get_iq_0(odrive_can::Msg::MSG_GET_IQ | odrive_can::AXIS::AXIS_3_ID,1, 0x7FF, 500),
                               socket_channel1_get_iq_1(odrive_can::Msg::MSG_GET_IQ | odrive_can::AXIS::AXIS_2_ID,1, 0x7FF, 110000), 
                               socket_get_encoder_estimates_0_axis0(odrive_can::Msg::MSG_GET_ENCODER_ESTIMATES | odrive_can::AXIS::AXIS_0_ID,0, 0x7FF, 500),
                               socket_get_encoder_estimates_0_axis1(odrive_can::Msg::MSG_GET_ENCODER_ESTIMATES | odrive_can::AXIS::AXIS_1_ID,0, 0x7FF, 110000),
                               socket_get_encoder_estimates_1_axis0(odrive_can::Msg::MSG_GET_ENCODER_ESTIMATES | odrive_can::AXIS::AXIS_3_ID,1, 0x7FF, 500),
                               socket_get_encoder_estimates_1_axis1(odrive_can::Msg::MSG_GET_ENCODER_ESTIMATES | odrive_can::AXIS::AXIS_2_ID,1, 0x7FF, 110000),
                               socket_topic_set_position_0_axis0(odrive_can::Msg::MSG_SET_INPUT_POS | odrive_can::AXIS::AXIS_0_ID,0, 0x7FF, 500),
                               socket_topic_set_position_0_axis1(odrive_can::Msg::MSG_SET_INPUT_POS | odrive_can::AXIS::AXIS_1_ID,0, 0x7FF, 500),
                               socket_topic_set_position_1_axis3(odrive_can::Msg::MSG_SET_INPUT_POS | odrive_can::AXIS::AXIS_3_ID,1, 0x7FF, 500),
                               socket_topic_set_position_1_axis2(odrive_can::Msg::MSG_SET_INPUT_POS | odrive_can::AXIS::AXIS_2_ID,1, 0x7FF, 500),
                               socket_topic_set_state_0_axis0(odrive_can::Msg::MSG_SET_AXIS_REQUESTED_STATE | odrive_can::AXIS::AXIS_0_ID,0, 0x7FF, 500),
                               socket_topic_set_state_0_axis1(odrive_can::Msg::MSG_SET_AXIS_REQUESTED_STATE | odrive_can::AXIS::AXIS_1_ID,0, 0x7FF, 500),
                               socket_topic_set_state_1_axis3(odrive_can::Msg::MSG_SET_AXIS_REQUESTED_STATE | odrive_can::AXIS::AXIS_3_ID,1, 0x7FF, 500),
                               socket_topic_set_state_1_axis2(odrive_can::Msg::MSG_SET_AXIS_REQUESTED_STATE | odrive_can::AXIS::AXIS_2_ID,1, 0x7FF, 500)

                               {   
   // count = 0;
    now = std::chrono::system_clock::now();
   //  publisher_0_axis0= this->create_publisher<traveler_msgs::msg::OdriveStatus>("/odrive/odrive_status", 100);
   //  publisher_0_axis1= this->create_publisher<traveler_msgs::msg::OdriveStatus>("/odrive/odrive_status", 100);
   //  publisher_1_axis0= this->create_publisher<traveler_msgs::msg::OdriveStatus>("/odrive/odrive_status", 100);
   //  publisher_1_axis1= this->create_publisher<traveler_msgs::msg::OdriveStatus>("/odrive/odrive_status", 100);
    //timer_ = this->create_wall_timer(std::chrono::microseconds(1000), std::bind(&can_driver::updateChannel1StatusCallback, this));
    //timer_ = this->create_wall_timer(std::chrono::microseconds(1000), std::bind(&can_driver::updateChannel2StatusCallback, this));
    
}

can_driver::~can_driver()
{

}

void can_driver::updateChannel1StatusCallback_0()
{   
    odrive_status_msg_0_axis0.can_channel = 0;
    odrive_status_msg_0_axis0.axis = 0;
    
   //  can_frame st_recv_frame_0_axis0;
   //  if (socket_channel0_axis0_read_.readFrame(&st_recv_frame_0_axis0) < 0) 
   //   {
   //      // RCLCPP_INFO(this->get_logger(), "No Encoder Response Received channel0_axis0");
   //   }
   //    else{
   //      odrive_status_msg_0_axis0.axis_state = odrive_can::can_getSignal<float>(st_recv_frame_0_axis0, 32, 8, true);
   //    }
    

    can_frame recv_frame_0_axis0;
    if (socket_get_encoder_estimates_0_axis0.readFrame(&recv_frame_0_axis0) < 0) 
     {
        // RCLCPP_INFO(this->get_logger(), "No Encoder Response Received channel0_axis0");
     }
      else{
        odrive_status_msg_0_axis0.pos_estimate = odrive_can::can_getSignal<float>(recv_frame_0_axis0, 0, 32, true);
        odrive_status_msg_0_axis0.vel_estimate = odrive_can::can_getSignal<float>(recv_frame_0_axis0, 32, 32, true);
    }

    can_frame iq_recv_frame_0_axis0;
    if (socket_channel0_get_iq_0.readFrame(&iq_recv_frame_0_axis0) < 0)
     {
         // RCLCPP_INFO(this->get_logger(), "No Current Response Received channel0_axis0");
     }
     else{
        odrive_status_msg_0_axis0.iq_setpoint = odrive_can::can_getSignal<float>(iq_recv_frame_0_axis0, 0, 32, true);
        odrive_status_msg_0_axis0.iq_measured = odrive_can::can_getSignal<float>(iq_recv_frame_0_axis0, 32, 32, true);
     }
    
   //  publisher_0_axis0->publish(odrive_status_msg_0_axis0);
} 

void can_driver::updateChannel1StatusCallback_1()
{   
    odrive_status_msg_0_axis1.can_channel = 0;
    odrive_status_msg_0_axis1.axis = 1;

   //      can_frame st_recv_frame_0_axis1;
   //  if (socket_channel0_axis1_read_.readFrame(&st_recv_frame_0_axis1) < 0) 
   //   {
   //      // RCLCPP_INFO(this->get_logger(), "No Encoder Response Received channel0_axis0");
   //   }
   //    else{
   //      odrive_status_msg_0_axis1.axis_state = odrive_can::can_getSignal<float>(st_recv_frame_0_axis1, 32, 8, true);
   //    }
    

    can_frame recv_frame_0_axis1;
    if (socket_get_encoder_estimates_0_axis1.readFrame(&recv_frame_0_axis1) < 0) 
     {
       // RCLCPP_INFO(this->get_logger(), "No Encoder Response Received channel0_axis1");
     }
      else{
        odrive_status_msg_0_axis1.pos_estimate = odrive_can::can_getSignal<float>(recv_frame_0_axis1, 0, 32, true);
        odrive_status_msg_0_axis1.vel_estimate = odrive_can::can_getSignal<float>(recv_frame_0_axis1, 32, 32, true);
    }

    can_frame iq_recv_frame_0_axis1;
    if (socket_channel0_get_iq_1.readFrame(&iq_recv_frame_0_axis1) < 0)
     {
       // RCLCPP_INFO(this->get_logger(), "No Current Response Received channel0_axis1");
     }
     else{
        odrive_status_msg_0_axis1.iq_setpoint = odrive_can::can_getSignal<float>(iq_recv_frame_0_axis1, 0, 32, true);
        odrive_status_msg_0_axis1.iq_measured = odrive_can::can_getSignal<float>(iq_recv_frame_0_axis1, 32, 32, true);
     }
    
   //  publisher_0_axis1->publish(odrive_status_msg_0_axis1);
}   

void can_driver::updateChannel2StatusCallback_0(){
     //count = count + 1;
    odrive_status_msg_1_axis0.axis = 0;
    odrive_status_msg_1_axis0.can_channel = 1;

    // the same reason for channel 1

    // can_frame axis0_frame_channel1;
   //      can_frame st_recv_frame_1_axis0;
   //  if (socket_channel1_axis0_read_.readFrame(&st_recv_frame_1_axis0) < 0) 
   //   {
   //      // RCLCPP_INFO(this->get_logger(), "No Encoder Response Received channel0_axis0");
   //   }
   //    else{
   //      odrive_status_msg_1_axis0.axis_state = odrive_can::can_getSignal<float>(st_recv_frame_1_axis0, 32, 8, true);
   //    }
    
    
    can_frame recv_frame_1_axis0;


    if (socket_get_encoder_estimates_1_axis0.readFrame(&recv_frame_1_axis0) < 0) 
     {
        // RCLCPP_INFO(this->get_logger(), "No Encoder Response Received channel1_axis0");
     }
     else{
        odrive_status_msg_1_axis0.pos_estimate = odrive_can::can_getSignal<float>(recv_frame_1_axis0, 0, 32, true);
        odrive_status_msg_1_axis0.vel_estimate = odrive_can::can_getSignal<float>(recv_frame_1_axis0, 32, 32, true);
    
     }
 
    can_frame iq_recv_frame_1_axis0;
    if (socket_channel1_get_iq_0.readFrame(&iq_recv_frame_1_axis0) < 0)
     {
         // RCLCPP_INFO(this->get_logger(), "No Current Response Received channel1_axis0");
     }
     else{
        odrive_status_msg_1_axis0.iq_setpoint = odrive_can::can_getSignal<float>(iq_recv_frame_1_axis0, 0, 32, true);
        odrive_status_msg_1_axis0.iq_measured = odrive_can::can_getSignal<float>(iq_recv_frame_1_axis0, 32, 32, true);
 
     }
   //  publisher_1_axis0->publish(odrive_status_msg_1_axis0);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = end - now; // duration in seconds as a double
    // float time_cucrr = static_cast<float>(diff.count());
    // std::cout<< "time: " << time_cucrr << "count: " << count << std::endl;
}

void can_driver::updateChannel2StatusCallback_1(){
    count = count + 1;
    odrive_status_msg_1_axis1.axis = 1;
    odrive_status_msg_1_axis1.can_channel = 1;

    // the same reason for channel 1

    // can_frame axis0_frame_channel1;
   // can_frame st_recv_frame_1_axis1;
   //  if (socket_channel1_axis1_read_.readFrame(&st_recv_frame_1_axis1) < 0) 
   //   {
   //      // RCLCPP_INFO(this->get_logger(), "No Encoder Response Received channel0_axis0");
   //   }
   //    else{
   //      odrive_status_msg_1_axis1.axis_state = odrive_can::can_getSignal<float>(st_recv_frame_1_axis1, 32, 8, true);
   //    }
    
    can_frame recv_frame_1_axis1;
    if (socket_get_encoder_estimates_1_axis1.readFrame(&recv_frame_1_axis1) < 0) 
     {
      // RCLCPP_INFO(this->get_logger(), "No Encoder Response Received channel1_axis1");
     }
     else{
        odrive_status_msg_1_axis1.pos_estimate = odrive_can::can_getSignal<float>(recv_frame_1_axis1, 0, 32, true);
        odrive_status_msg_1_axis1.vel_estimate = odrive_can::can_getSignal<float>(recv_frame_1_axis1, 32, 32, true);
    
     }
    can_frame iq_recv_frame_1_axis1;
    if (socket_channel1_get_iq_1.readFrame(&iq_recv_frame_1_axis1) < 0)
     {
         //RCLCPP_INFO(this->get_logger(), "No Current Response Received channel1_axis1");
     }
     else{
        odrive_status_msg_1_axis1.iq_setpoint = odrive_can::can_getSignal<float>(iq_recv_frame_1_axis1, 0, 32, true);
        odrive_status_msg_1_axis1.iq_measured = odrive_can::can_getSignal<float>(iq_recv_frame_1_axis1, 32, 32, true);
 
     }
   // publisher_1_axis1->publish(odrive_status_msg_1_axis1);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = end - now; // duration in seconds as a double
    // float time_cucrr = static_cast<float>(diff.count());
    // std::cout<< "time: " << time_cucrr << "count: " << count << std::endl;
}


void can_driver::get_motor_status(turtle& turtle_){
      // updateChannel1StatusCallback_0();
      // updateChannel1StatusCallback_1();
      updateChannel2StatusCallback_0();
      updateChannel2StatusCallback_1();
   
      // turtle_.turtle_chassis.left_adduction = odrive_status_msg_0_axis1;
      // turtle_.turtle_chassis.left_sweeping = odrive_status_msg_0_axis0;
      turtle_.turtle_chassis.right_adduction = odrive_status_msg_1_axis1;
      turtle_.turtle_chassis.right_sweeping = odrive_status_msg_1_axis0;


}

void can_driver::setControl(turtle& turtle_){
   // turtle_.turtle_control.left_adduction.set_input_position_radian.can_channel = 0;
   // turtle_.turtle_control.left_sweeping.set_input_position_radian.can_channel = 0;
   turtle_.turtle_control.right_adduction.set_input_position_radian.can_channel = 1;
   turtle_.turtle_control.right_sweeping.set_input_position_radian.can_channel = 1;
   std::cout<<"ifcontrol:" << turtle_.turtle_control.if_control<<std::endl;
   
   if(turtle_.turtle_control.if_control){
      // setPosition_left_adduction(turtle_.turtle_control.left_adduction.set_input_position_radian);
      // setPosition_left_sweeping(turtle_.turtle_control.left_sweeping.set_input_position_radian);
      setPosition_right_adduction(turtle_.turtle_control.right_adduction.set_input_position_radian);
      setPosition_right_sweeping(turtle_.turtle_control.right_sweeping.set_input_position_radian);
   }
}

void can_driver::change_odrive_state(turtle &turtle_)
{  
   // std::cout<<"idle"<<turtle_gui.if_idle<<std::endl;
    if (turtle_.turtle_gui.start_flag==1){
      // std::cout << turtle_.turtle_chassis.if_idle_count << std::endl;
        if (turtle_.turtle_chassis.if_idle_count > 0)
        {   
            
            turtle_.turtle_control.left_adduction.set_state.can_channel = 1;
            turtle_.turtle_control.left_sweeping.set_state.can_channel = 1;
            turtle_.turtle_control.right_adduction.set_state.can_channel = 0;
            turtle_.turtle_control.right_sweeping.set_state.can_channel = 0;
            turtle_.turtle_control.left_adduction.set_state.set_state = 8;
            turtle_.turtle_control.left_sweeping.set_state.set_state = 8;
            turtle_.turtle_control.right_adduction.set_state.set_state = 8;
            turtle_.turtle_control.right_sweeping.set_state.set_state = 8;
            // setstate_left_adduction(turtle_.turtle_control.left_adduction.set_state);
            // setstate_left_sweeping(turtle_.turtle_control.left_sweeping.set_state);
            setstate_right_adduction(turtle_.turtle_control.right_adduction.set_state);
            setstate_right_sweeping(turtle_.turtle_control.right_sweeping.set_state);
            std::cout << "set to close loop control" << std::endl;
            turtle_.turtle_chassis.if_idle_count = turtle_.turtle_chassis.if_idle_count - 1;
            // std::cout << turtle_.turtle_chassis.if_idle_count << std::endl;
        }
    
         }
      else 
      {
       if (turtle_.turtle_chassis.if_idle_count < 1)
        {
            turtle_.turtle_control.left_adduction.set_state.can_channel = 1;
            turtle_.turtle_control.left_sweeping.set_state.can_channel = 1;
            turtle_.turtle_control.right_adduction.set_state.can_channel = 0;
            turtle_.turtle_control.right_sweeping.set_state.can_channel = 0;
            turtle_.turtle_control.left_adduction.set_state.set_state = 1;
            turtle_.turtle_control.left_sweeping.set_state.set_state = 1;
            turtle_.turtle_control.right_adduction.set_state.set_state = 1;
            turtle_.turtle_control.right_sweeping.set_state.set_state = 1;
            // setstate_left_adduction(turtle_.turtle_control.left_adduction.set_state);
            // setstate_left_sweeping(turtle_.turtle_control.left_sweeping.set_state);
            setstate_right_adduction(turtle_.turtle_control.right_adduction.set_state);
            setstate_right_sweeping(turtle_.turtle_control.right_sweeping.set_state);
            turtle_.turtle_chassis.if_idle_count = turtle_.turtle_chassis.if_idle_count + 1;
            std::cout << "set to idle" << std::endl;
        }
      }


        
    }


void can_driver::setstate_left_adduction(traveler_msgs::msg::SetState msg)
{
   // RCLCPP_INFO(this->get_logger(), "Do Encoder Response Received");
   // std::cout<<"hahahaha"<<std::endl;
   can_frame send_frame;
   send_frame.can_dlc = 4;
   std::memcpy(&send_frame.data[0], &msg.set_state, sizeof(msg.set_state));


   send_frame.can_id = odrive_can::Msg::MSG_SET_AXIS_REQUESTED_STATE | odrive_can::AXIS::AXIS_1_ID;
   socket_topic_set_state_0_axis1.writeFrame(send_frame);

   // socket_generic_write_.writeFrame(send_frame);
}

void can_driver::setstate_left_sweeping(traveler_msgs::msg::SetState msg)
{
   // RCLCPP_INFO(this->get_logger(), "Do Encoder Response Received");
   // std::cout<<"hahahaha"<<std::endl;
   can_frame send_frame;
   send_frame.can_dlc = 4;
   std::memcpy(&send_frame.data[0], &msg.set_state, sizeof(msg.set_state));


   send_frame.can_id = odrive_can::Msg::MSG_SET_AXIS_REQUESTED_STATE | odrive_can::AXIS::AXIS_0_ID;
   socket_topic_set_state_0_axis0.writeFrame(send_frame);
   // socket_generic_write_.writeFrame(send_frame);
}

void can_driver::setstate_right_adduction(traveler_msgs::msg::SetState msg)
{
   // RCLCPP_INFO(this->get_logger(), "Do Encoder Response Received");
   // std::cout<<"hahahaha"<<std::endl;
   can_frame send_frame;
   send_frame.can_dlc = 4;
   std::memcpy(&send_frame.data[0], &msg.set_state, sizeof(msg.set_state));


   send_frame.can_id = odrive_can::Msg::MSG_SET_AXIS_REQUESTED_STATE | odrive_can::AXIS::AXIS_2_ID;
   socket_topic_set_state_1_axis2.writeFrame(send_frame);

   // socket_generic_write_.writeFrame(send_frame);
}

void can_driver::setstate_right_sweeping(traveler_msgs::msg::SetState msg)
{
   // RCLCPP_INFO(this->get_logger(), "Do Encoder Response Received");
   // std::cout<<"hahahaha"<<std::endl;
   can_frame send_frame;
   send_frame.can_dlc = 4;
   std::memcpy(&send_frame.data[0], &msg.set_state, sizeof(msg.set_state));


   send_frame.can_id = odrive_can::Msg::MSG_SET_AXIS_REQUESTED_STATE | odrive_can::AXIS::AXIS_3_ID;
   socket_topic_set_state_1_axis3.writeFrame(send_frame);
  
   // socket_generic_write_.writeFrame(send_frame);
}

void can_driver::setPosition_left_adduction(traveler_msgs::msg::SetInputPosition msg)
{
   // RCLCPP_INFO(this->get_logger(), "Do Encoder Response Received");
   // std::cout<<"hahahaha"<<std::endl;
   can_frame send_frame;
   send_frame.can_dlc = 8;
   std::memcpy(&send_frame.data[0], &msg.input_position, sizeof(msg.input_position));
   std::memcpy(&send_frame.data[4], &msg.vel_ff, sizeof(msg.vel_ff));
   std::memcpy(&send_frame.data[6], &msg.torque_ff, sizeof(msg.torque_ff));

   send_frame.can_id = odrive_can::Msg::MSG_SET_INPUT_POS | odrive_can::AXIS::AXIS_1_ID;
   socket_topic_set_position_0_axis1.writeFrame(send_frame);
}

void can_driver::setPosition_left_sweeping(traveler_msgs::msg::SetInputPosition msg)
{
   // RCLCPP_INFO(this->get_logger(), "Do Encoder Response Received");
   // std::cout<<"hahahaha"<<std::endl;
   can_frame send_frame;
   send_frame.can_dlc = 8;
   std::memcpy(&send_frame.data[0], &msg.input_position, sizeof(msg.input_position));
   std::memcpy(&send_frame.data[4], &msg.vel_ff, sizeof(msg.vel_ff));
   std::memcpy(&send_frame.data[6], &msg.torque_ff, sizeof(msg.torque_ff));

   send_frame.can_id = odrive_can::Msg::MSG_SET_INPUT_POS | odrive_can::AXIS::AXIS_0_ID;
   socket_topic_set_position_0_axis0.writeFrame(send_frame);
}
void can_driver::setPosition_right_adduction(traveler_msgs::msg::SetInputPosition msg)
{
   // RCLCPP_INFO(this->get_logger(), "Do Encoder Response Received");
   // std::cout<<"hahahaha"<<std::endl;
   can_frame send_frame;
   send_frame.can_dlc = 8;
   std::memcpy(&send_frame.data[0], &msg.input_position, sizeof(msg.input_position));
   std::memcpy(&send_frame.data[4], &msg.vel_ff, sizeof(msg.vel_ff));
   std::memcpy(&send_frame.data[6], &msg.torque_ff, sizeof(msg.torque_ff));

   send_frame.can_id = odrive_can::Msg::MSG_SET_INPUT_POS | odrive_can::AXIS::AXIS_2_ID;
   socket_topic_set_position_1_axis2.writeFrame(send_frame);
}
void can_driver::setPosition_right_sweeping(traveler_msgs::msg::SetInputPosition msg)
{
   // RCLCPP_INFO(this->get_logger(), "Do Encoder Response Received");
   // std::cout<<"hahahaha"<<std::endl;
   can_frame send_frame;
   send_frame.can_dlc = 8;
   std::memcpy(&send_frame.data[0], &msg.input_position, sizeof(msg.input_position));
   std::memcpy(&send_frame.data[4], &msg.vel_ff, sizeof(msg.vel_ff));
   std::memcpy(&send_frame.data[6], &msg.torque_ff, sizeof(msg.torque_ff));

   send_frame.can_id = odrive_can::Msg::MSG_SET_INPUT_POS | odrive_can::AXIS::AXIS_3_ID;
   socket_topic_set_position_1_axis3.writeFrame(send_frame);
}


