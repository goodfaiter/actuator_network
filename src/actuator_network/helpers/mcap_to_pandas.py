import pandas as pd
from mcap_ros2.reader import read_ros2_messages

def process_field(topic, field):
    """Process a ROS message field and return a dictionary with meaningful names"""
    if field._type == 'sensor_msgs/Imu':
        return {
            # Orientation
            f"{topic}_orientation_x": field.orientation.x,
            f"{topic}_orientation_y": field.orientation.y,
            f"{topic}_orientation_z": field.orientation.z,
            f"{topic}_orientation_w": field.orientation.w,
            
            # Angular velocity
            f"{topic}_angular_velocity_x": field.angular_velocity.x,
            f"{topic}_angular_velocity_y": field.angular_velocity.y,
            f"{topic}_angular_velocity_z": field.angular_velocity.z,
            
            # Linear acceleration
            f"{topic}_linear_acceleration_x": field.linear_acceleration.x,
            f"{topic}_linear_acceleration_y": field.linear_acceleration.y,
            f"{topic}_linear_acceleration_z": field.linear_acceleration.z,
        }
    if field._type == 'std_msgs/Float32':
        return {
            # Orientation
            f"{topic}_data": field.data,
        }
    return None

def read_mcap_to_dataframe(file_path: str) -> pd.DataFrame:
    """Read MCAP file to pandas DataFrame"""
    data = []

    msgs = read_ros2_messages(file_path, topics=["/imu/data_raw", 
                                                 "/weight_kg",
                                                 "/desired_position_rad",
                                                 "/measured_position_rad",
                                                 "/measured_velocity_rad_per_sec",])

    for msg in msgs:
        # Convert message to dictionary
        msg_dict = {}
        msg_dict["timestamp"] = msg.log_time_ns

        ros_msg_processed = process_field(msg.channel.topic[1:].replace("/", "_"), msg.ros_msg)

        if ros_msg_processed is not None:
            # Add all processed data to msg_dict
            msg_dict.update(ros_msg_processed)

        data.append(msg_dict)

    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")
        df = df.sort_values("timestamp")

    df.set_index('timestamp', inplace=True)

    return df
