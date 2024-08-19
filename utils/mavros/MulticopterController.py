import math
import threading
from functools import partial
from threading import Thread

import numpy as np
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, WaypointList, State, Mavlink, WaypointReached, \
    ParamValue, GPSRAW
from mavros_msgs.srv import ParamGet, ParamSet, CommandBool, SetMode, WaypointClear, WaypointPush
from pymavlink import mavutil
from sensor_msgs.msg import NavSatFix, Imu

import rospy
from mavros import mavlink

from .BaseROSController import BaseROSController
from .subscriber_callbacks import *


__all__ = ['MultiCopterController']


class MultiCopterController(BaseROSController):
    # TODO increase loop freq if simulator is running at faster rate
    def __init__(self, node_name='mavros', sim_speed: int = 1):
        super(MultiCopterController, self).__init__()
        self._init_messages()
        self._init_service(node_name)
        self._init_topics(node_name)
        self._init_heartbeat()
        self.sim_speed = sim_speed

    def _init_messages(self):
        # Crucial topics
        self.altitude = Altitude()
        self.state = State()
        self.extended_state = ExtendedState()
        self.home_position = HomePosition()
        self.local_position = PoseStamped()
        self.global_position = NavSatFix()
        self.imu_data = Imu()
        self.mission_items = WaypointList()
        # self.hil_state_quaternion = HilStateQuaternion()
        self.gps1_raw = GPSRAW()
        self.sub_topics_ready = {key: False for key, _ in self.named_messages()}
        rospy.loginfo("Registered Message: {0}".format(tuple(self.sub_topics_ready.keys())))

        # Non-crucial states
        self.mission_item_reached = -1  # first mission item is 0
        self.mission_name = ""

    def _init_service(self, node_name):
        # ROS services
        service_timeout = 30
        rospy.loginfo("waiting for ROS services")
        try:
            rospy.wait_for_service(f'{node_name}/param/get', service_timeout)
            rospy.wait_for_service(f'{node_name}/param/set', service_timeout)
            rospy.wait_for_service(f'{node_name}/cmd/arming', service_timeout)
            rospy.wait_for_service(f'{node_name}/mission/push', service_timeout)
            rospy.wait_for_service(f'{node_name}/mission/clear', service_timeout)
            rospy.wait_for_service(f'{node_name}/set_mode', service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException as e:
            raise e

        # Initialize service proxies after services are all up.
        self.get_param_srv = rospy.ServiceProxy(f'{node_name}/param/get', ParamGet)
        self.set_param_srv = rospy.ServiceProxy(f'{node_name}/param/set', ParamSet)
        self.set_arming_srv = rospy.ServiceProxy(f'{node_name}/cmd/arming', CommandBool)
        self.set_mode_srv = rospy.ServiceProxy(f'{node_name}/set_mode', SetMode)
        self.wp_clear_srv = rospy.ServiceProxy(f'{node_name}/mission/clear', WaypointClear)
        self.wp_push_srv = rospy.ServiceProxy(f'{node_name}/mission/push', WaypointPush)

    def _init_topics(self, node_name):
        # Wrap the callback function with self
        def get_sub(path, msg_cls, callback, *args, **kwargs):
            return rospy.Subscriber(path, msg_cls, partial(callback, self), *args, **kwargs)
        
        # ROS subscribers
        self.alt_sub = get_sub(f'{node_name}/altitude', Altitude, update_altitude)
        self.state_sub = get_sub(f'{node_name}/state', State, update_state)
        self.ext_state_sub = get_sub(f'{node_name}/extended_state', ExtendedState, update_extended_state)
        self.home_pos_sub = get_sub(f'{node_name}/home_position/home', HomePosition, update_home_position)
        self.local_pos_sub = get_sub(f'{node_name}/local_position/pose', PoseStamped, update_local_position)
        self.global_pos_sub = get_sub(f'{node_name}/global_position/global', NavSatFix, update_global_position)
        self.imu_data_sub = get_sub(f'{node_name}/imu/data', Imu, update_imu_data)
        self.gps1_raw_sub = get_sub(f'{node_name}/gpsstatus/gps1/raw', GPSRAW, update_gps1_raw)

        self.mission_wp_sub = get_sub(f'{node_name}/mission/waypoints', WaypointList, update_mission_wp)
        self.mission_item_reached_sub = get_sub(
            f'{node_name}/mission/reached', WaypointReached, update_mission_item_reached
        )

        # ROS publishers
        self.mavlink_pub = rospy.Publisher('mavlink/to', Mavlink, queue_size=1)

    def _init_heartbeat(self):
        # need to simulate heartbeat to prevent datalink loss detection
        hb_mav_msg = mavutil.mavlink.MAVLink_heartbeat_message(mavutil.mavlink.MAV_TYPE_GCS, 0, 0, 0, 0, 0)
        hb_mav_msg.pack(mavutil.mavlink.MAVLink('', 2, 1))
        hb_ros_msg = mavlink.convert_to_rosmsg(hb_mav_msg)

        self.stop_heartbeat = threading.Event()

        def send_heartbeat():
            rate = rospy.Rate(2)  # Hz
            while not rospy.is_shutdown() and not self.stop_heartbeat.is_set():
                self.mavlink_pub.publish(hb_ros_msg)
                try:  # prevent garbage in console output when thread is killed
                    rate.sleep()
                except rospy.ROSInterruptException:
                    pass

        self.heartbeat_thread = Thread(target=send_heartbeat, args=())
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()

    def close(self):
        if self.__dict__.get('heartbeat_thread'):
            # Stop the thread by event.
            self.stop_heartbeat.set()
            self.heartbeat_thread.join(5)
        super(MultiCopterController, self).close()

    #
    # Helper methods
    #
    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        loop_freq = 1  # Hz
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        old_arm = self.state.armed
        arm_set = False
        for i in range(timeout * loop_freq):
            if self.state.armed == arm:
                arm_set = True
                rospy.loginfo(f"set arm success | seconds: {i / loop_freq} of {timeout}")
                break
            else:
                try:
                    res = self.set_arming_srv(arm)
                    if not res.success:
                        rospy.logerr("failed to send arm command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            rate.sleep()

        if not arm_set:
            rospy.logerr(f"failed to set arm | "
                         f"new arm: {arm}, old arm: {old_arm} | "
                         f"timeout(seconds): {timeout}")

        return arm_set

    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        old_mode = self.state.mode
        loop_freq = 1  # Hz
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        mode_set = False
        for i in range(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            rate.sleep()

        if not mode_set:
            rospy.logerr(f"failed to set mode | "
                         f"new mode: {mode}, old mode: {old_mode} | "
                         f"timeout(seconds): {timeout}")

        return mode_set

    def get_param(self, param_id: str, timeout):
        rospy.loginfo(f'waiting for {param_id}')
        loop_freq = 1
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        param_received = False
        recv_value = None
        for i in range(timeout * loop_freq):
            try:
                recv_value = self._get_param_once(param_id)
                if recv_value is not None:
                    param_received = True
                    rospy.loginfo(f"{param_id} received | "
                                  f"param_id: {param_id}, param_value: {recv_value} | "
                                  f"seconds: {i / loop_freq} of {timeout}")
                    break
            except rospy.ServiceException as e:
                rospy.logerr(e)
            rate.sleep()

        if not param_received:
            rospy.logerr(f"{param_id} param get failed | timeout(seconds): {timeout}")

        return recv_value

    def _get_param_once(self, param_id: str):
        recv_value = None
        res = self.get_param_srv(param_id)
        if res.success:
            recv_value = res.value.real
            if recv_value == 0.0:
                # By mavros ParamValue convention, there are two case when float is 0:
                #   - The param is a real number zero
                #   - The param is integer
                # So we return integer 0 instead.
                recv_value = res.value.integer

        return recv_value

    def set_param(self, param_id: str, param_value, timeout):
        rospy.loginfo(f'setting {param_id} to value {param_value}')
        loop_freq = 1
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        param_setted = False
        # TODO Do something if wrongly interpret float as int or int as float
        target_value = ParamValue(
            integer=param_value if isinstance(param_value, int) else 0,
            real=param_value if isinstance(param_value, float) else 0.0
        )
        for i in range(timeout * loop_freq):
            try:
                res = self.set_param_srv(param_id, target_value)
                if res.success:
                    param_setted = True
                    current_value = res.value.integer if res.value.real == 0.0 else res.value.real
                    if current_value == 0 and isinstance(param_value, float):
                        current_value = float(current_value)

                    rospy.loginfo(f'{param_id} has changed to {current_value}')
                    if not np.isclose(current_value, param_value):
                        rospy.logwarn(f'expect {param_id} change to {param_value}, got {current_value} instead')
                    break
            except rospy.ServiceException as e:
                rospy.logerr(e)
            rate.sleep()

        if not param_setted:
            rospy.logerr(f"{param_id} param set failed | timeout(seconds): {timeout}")

        return param_setted

    def wait_for_topics(self, timeout):
        """wait for simulation to be ready, make sure we're getting topic info
        from all topics by checking dictionary of flag values set in callbacks,
        timeout(int): seconds"""
        rospy.loginfo("waiting for subscribed topics to be ready")
        loop_freq = 1  # Hz
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        simulation_ready = False
        for i in range(timeout * loop_freq):
            if all(value for value in self.sub_topics_ready.values()):
                simulation_ready = True
                rospy.loginfo("simulation topics ready | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break
            rate.sleep()

        if not simulation_ready:
            rospy.logerr(f"failed to hear from all subscribed simulation topics | "
                         f"topic ready flags: {self.sub_topics_ready} | "
                         f"timeout(seconds): {timeout}")

        return simulation_ready

    def get_simulation_rate(self, loop_freq):
        hz = int(loop_freq * self.sim_speed)
        return rospy.Rate(hz), hz

    def wait_for_landed_state(self, desired_landed_state, timeout, index):
        state_name = mavutil.mavlink.enums['MAV_LANDED_STATE'][desired_landed_state].name
        rospy.loginfo(f"waiting for landed state | state: {state_name}, index: {index}")
        loop_freq = 10  # Hz
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        landed_state_confirmed = False
        for i in range(timeout * loop_freq):
            if self.extended_state.landed_state == desired_landed_state:
                landed_state_confirmed = True
                rospy.loginfo("landed state confirmed | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break
            rate.sleep()

        if not landed_state_confirmed:
            current_state = mavutil.mavlink.enums['MAV_LANDED_STATE'][self.extended_state.landed_state]
            rospy.logerr(f"landed state not detected | "
                         f"desired: {state_name}, current: {current_state.name} | "
                         f"index: {index}, timeout(seconds): {timeout}")

        return landed_state_confirmed

    def clear_waypoints(self, timeout):
        """timeout(int): seconds"""
        loop_freq = 1  # Hz
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        wps_cleared = False
        for i in range(timeout * loop_freq):
            if not self.mission_items.waypoints:
                wps_cleared = True
                rospy.loginfo(f"clear waypoints success | seconds: {i / loop_freq} of {timeout}")
                break
            else:
                try:
                    res = self.wp_clear_srv()
                    if not res.success:
                        rospy.logerr("failed to send waypoint clear command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            rate.sleep()

        if not wps_cleared:
            rospy.logerr(f"failed to clear waypoints | timeout(seconds): {timeout}")

        return wps_cleared

    def send_waypoints(self, waypoints, timeout):
        """waypoints, timeout(int): seconds"""
        rospy.loginfo("sending mission waypoints")
        if self.mission_items.waypoints:
            rospy.loginfo("FCU already has mission waypoints")

        loop_freq = 1  # Hz
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        wps_sent = False
        wps_verified = False
        for i in range(timeout * loop_freq):
            if not wps_sent:
                try:
                    res = self.wp_push_srv(start_index=0, waypoints=waypoints)
                    wps_sent = res.success
                    if wps_sent:
                        rospy.loginfo("waypoints successfully transferred")
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            else:
                if len(waypoints) == len(self.mission_items.waypoints):
                    rospy.loginfo(f"number of waypoints transferred: {len(waypoints)}")
                    wps_verified = True

            if wps_sent and wps_verified:
                rospy.loginfo(f"send waypoints success | seconds: {i / loop_freq} of {timeout}")
                break
            rate.sleep()

        if not (wps_sent and wps_verified):
            rospy.logerr(f"mission could not be transferred and verified | timeout(seconds): {timeout}")

        return wps_sent and wps_verified

    def wait_for_mav_type(self, timeout):
        """Wait for MAV_TYPE parameter, timeout(int): seconds"""
        rospy.loginfo("waiting for MAV_TYPE")
        loop_freq = 1  # Hz
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        mav_type_received = False
        for i in range(timeout * loop_freq):
            try:
                mav_type = self._get_param_once('MAV_TYPE')
                if mav_type is not None:
                    mav_type_received = True
                    self.mav_type = int(mav_type)
                    rospy.loginfo(f"MAV_TYPE received | "
                                  f"type: {mavutil.mavlink.enums['MAV_TYPE'][self.mav_type].name} | "
                                  f"seconds: {i / loop_freq} of {timeout}")
                    break
            except rospy.ServiceException as e:
                rospy.logerr(e)
            rate.sleep()

        if not mav_type_received:
            rospy.logerr(f"MAV_TYPE param get failed | timeout(seconds): {timeout}")

        return mav_type_received

    def distance_to_wp(self, lat, lon, alt):
        """alt(amsl): meters"""
        R = 6371000  # metres
        rlat1 = math.radians(lat)
        rlat2 = math.radians(self.global_position.latitude)

        rlat_d = math.radians(self.global_position.latitude - lat)
        rlon_d = math.radians(self.global_position.longitude - lon)

        a = (math.sin(rlat_d / 2) * math.sin(rlat_d / 2) + math.cos(rlat1) *
             math.cos(rlat2) * math.sin(rlon_d / 2) * math.sin(rlon_d / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        d = R * c
        alt_d = abs(alt - self.altitude.amsl)

        rospy.logdebug(f"d: {d}, alt_d: {alt_d}")
        return d, alt_d

    def distance_to_groundtruth(self):
        latitude = self.gps1_raw.lat * 1e-7
        longitude = self.gps1_raw.lon * 1e-7
        altitude = self.gps1_raw.alt * 1e-3
        return self.distance_to_wp(latitude, longitude, altitude)

    def reach_position(self, lat, lon, alt, timeout, index):
        """alt(amsl): meters, timeout(int): seconds"""
        rospy.loginfo(f"trying to reach waypoint | lat: {lat:.9f}, lon: {lon:.9f}, alt: {alt:.2f}, index: {index}")
        best_pos_xy_d = None
        best_pos_z_d = None
        reached = False
        mission_length = len(self.mission_items.waypoints)
        pos_xy_d, pos_z_d = self.distance_to_wp(lat, lon, alt)  # Record initial location.

        # does it reach the position in 'timeout' seconds?
        loop_freq = 2  # Hz
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        for i in range(timeout * loop_freq):
            pos_xy_d, pos_z_d = self.distance_to_wp(lat, lon, alt)

            # remember best distances
            if not best_pos_xy_d or best_pos_xy_d > pos_xy_d:
                best_pos_xy_d = pos_xy_d
            if not best_pos_z_d or best_pos_z_d > pos_z_d:
                best_pos_z_d = pos_z_d

            # FCU advanced to the next mission item, or finished mission
            reached = ((index < self.mission_items.current_seq)
                       or (index == (mission_length - 1) and self.mission_item_reached == index))
            if reached:
                rospy.loginfo(f"position reached | "
                              f"pos_xy_d: {pos_xy_d:.2f}, pos_z_d: {pos_z_d:.2f}, index: {index} | "
                              f"seconds: {i / loop_freq} of {timeout}")
                break
            elif i == 0 or ((i / loop_freq) % 10) == 0:
                # log distance first iteration and every 10 sec
                rospy.loginfo(f"current distance to waypoint | "
                              f"pos_xy_d: {pos_xy_d:.2f}, pos_z_d: {pos_z_d:.2f}, index: {index}")

            rate.sleep()

        if not reached:
            # Failed to reach mission item is not so critical.
            # So I use warn message instead of error.
            rospy.logwarn(f"position not reached | "
                          f"lat: {lat:.9f}, lon: {lon:.9f}, alt: {alt:.2f}, "
                          f"current pos_xy_d: {pos_xy_d:.2f}, current pos_z_d: {pos_z_d:.2f}, "
                          f"best pos_xy_d: {best_pos_xy_d:.2f}, best pos_z_d: {best_pos_z_d:.2f}, index: {index} | "
                          f"timeout(seconds): {timeout}")

        return reached

    def wait_till_deviated(self, max_distance, timeout, index, minimum_height: float = 2.0):
        """alt(amsl): meters, timeout(int): seconds"""
        rospy.loginfo(f"Waiting until deviated | max_distance: {max_distance:.3f}, index: {index}")
        deviated = False

        # does it reach the position in 'timeout' seconds?
        loop_freq = 10  # Hz
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        for i in range(timeout * loop_freq):
            dev_xy_d, dev_z_d = self.distance_to_groundtruth()
            cur_dev = np.sqrt(dev_xy_d ** 2 + dev_z_d ** 2)

            # remember the maximum distance
            if cur_dev >= max_distance:
                deviated = True
                rospy.logwarn(f"vehicle is deviated | "
                              f"current dev_xy_d: {dev_xy_d:.3f}, current dev_z_d: {dev_z_d:.3f}, "
                              f"current_deviation: {cur_dev:.3f}, max_allowed_deviation: {max_distance:.3f}"
                              f"index: {index} | timeout(seconds): {timeout}")
                break

            rel_hgt = self.local_position.pose.position.z
            if np.isfinite(rel_hgt) and rel_hgt < minimum_height:
                deviated = True
                rospy.logwarn(f"vehicle is below minimum height | "
                              f"current_height: {rel_hgt:.3f}, minimum_allowed_height: {minimum_height:.3f}"
                              f"index: {index} | timeout(seconds): {timeout}")
                break

            if i == 0 or ((i / loop_freq) % 10) == 0:
                # log distance first iteration and every 10 sec
                rospy.loginfo(f"current deviation from true position | "
                              f"dev_xy_d: {dev_xy_d:.2f}, dev_z_d: {dev_z_d:.2f},"
                              f"current_deviation: {cur_dev:.3f}, max_allowed_deviation: {max_distance:.3f}")
                rospy.loginfo(f"current_height: {rel_hgt:.3f}, min_allowed_height: {minimum_height:.3f} "
                              f"index: {index} | timeout(seconds): {timeout}")

            rate.sleep()

        return deviated

    def wait_for_next_wp(self, index, timeout):
        """timeout(int): seconds"""
        rospy.loginfo(f"trying to reach waypoint | index: {index}")
        reached = False
        mission_length = len(self.mission_items.waypoints)

        # does it reach the waypoint in 'timeout' seconds?
        loop_freq = 2  # Hz
        rate, loop_freq = self.get_simulation_rate(loop_freq)
        for i in range(timeout * loop_freq):
            # FCU advanced to the next mission item, or finished mission
            reached = ((index < self.mission_items.current_seq)
                       or (index == (mission_length - 1) and self.mission_item_reached == index))
            if reached:
                rospy.loginfo(f"waypoint reached! | index: {index} | seconds: {i / loop_freq} of {timeout}")
                break

            rate.sleep()

        if not reached:
            # Failed to reach mission item is not so critical.
            # So I use warn message instead of error.
            rospy.logwarn(f"waypoint not reached | index: {index} | timeout(seconds): {timeout}")

        return reached

    def log_topic_vars(self):
        """log the state of topic variables"""
        rospy.loginfo("========================")
        rospy.loginfo("===== topic values =====")
        rospy.loginfo("========================")
        rospy.loginfo("altitude:\n{}".format(self.altitude))
        rospy.loginfo("========================")
        rospy.loginfo("extended_state:\n{}".format(self.extended_state))
        rospy.loginfo("========================")
        rospy.loginfo("global_position:\n{}".format(self.global_position))
        rospy.loginfo("========================")
        rospy.loginfo("home_position:\n{}".format(self.home_position))
        rospy.loginfo("========================")
        rospy.loginfo("local_position:\n{}".format(self.local_position))
        rospy.loginfo("========================")
        rospy.loginfo("mission_items:\n{}".format(self.mission_items))
        rospy.loginfo("========================")
        rospy.loginfo("state:\n{}".format(self.state))
        rospy.loginfo("========================")
