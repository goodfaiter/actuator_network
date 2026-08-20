import pandas as pd
from mcap_ros2.reader import read_ros2_messages


def process_field(topic, field):
    """Process a ROS message field and return a dictionary with meaningful names"""
    if field._type == "sensor_msgs/Imu":
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
    if field._type == "std_msgs/Float32":
        return {
            # Orientation
            f"{topic}_data": field.data,
        }
    if field._type == "geometry_msgs/WrenchStamped":
        return {
            f"{topic}_force_x": field.wrench.force.x,
            f"{topic}_force_y": field.wrench.force.y,
            f"{topic}_force_z": field.wrench.force.z,
            f"{topic}_torque_x": field.wrench.torque.x,
            f"{topic}_torque_y": field.wrench.torque.y,
            f"{topic}_torque_z": field.wrench.torque.z,
        }
    return None


def read_mcap_to_dataframe(file_path: str, topics: list = None) -> pd.DataFrame:
    """Read MCAP file to pandas DataFrame with maximum performance."""
    if topics is None:
        topics = [
            "/imu/data_raw",
            "/weight_kg",
            "/desired_position_rad",
            "/measured_position_rad",
            "/measured_velocity_rad_per_sec",
            "/bota/wrench_N_and_Nm",
        ]

    # Cache sanitized topic names to avoid repeated string manipulation per message.
    topic_name_map = {topic: topic[1:].replace("/", "_") for topic in topics}

    msgs = read_ros2_messages(file_path, topics=topics)

    timestamps = []
    data_dicts = []

    # Single pass: process and filter messages in one loop.
    for msg in msgs:
        data = process_field(topic_name_map[msg.channel.topic], msg.ros_msg)
        if data is not None:
            timestamps.append(msg.log_time_ns)
            data_dicts.append(data)

    # Create DataFrame with timestamp as index in one operation
    df = pd.DataFrame(
        data_dicts,
        index=pd.to_datetime(timestamps, unit="ns"),
    ).sort_index()

    return df
