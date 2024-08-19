import math
import rospy
from pymavlink import mavutil

__all__ = ['update_altitude', 'update_state', 'update_extended_state', 'update_home_position', 'update_local_position',
           'update_global_position', 'update_imu_data', 'update_mission_wp', 'update_mission_item_reached',
           'update_hil_state_quaternion', 'update_gps1_raw']


def update_altitude(self, data):
    self.altitude = data
    # amsl has been observed to be nan while other fields are valid
    if not self.sub_topics_ready.get('altitude') and not math.isnan(data.amsl):
        self.sub_topics_ready['altitude'] = True


def update_state(self, data):
    if self.state.armed != data.armed:
        rospy.loginfo("armed state changed from {0} to {1}".format(self.state.armed, data.armed))

    if self.state.connected != data.connected:
        rospy.loginfo("connected changed from {0} to {1}".format(self.state.connected, data.connected))

    if self.state.mode != data.mode:
        rospy.loginfo("mode changed from {0} to {1}".format(self.state.mode, data.mode))

    if self.state.system_status != data.system_status:
        old_state = mavutil.mavlink.enums['MAV_STATE'][self.state.system_status].name
        new_state = mavutil.mavlink.enums['MAV_STATE'][data.system_status].name
        rospy.loginfo("system_status changed from {0} to {1}".format(old_state, new_state))

    self.state = data

    # mavros publishes a disconnected state message on init
    if not self.sub_topics_ready.get('state') and data.connected:
        self.sub_topics_ready['state'] = True

    self.mav_type = None


def update_extended_state(self, data):
    if self.extended_state.landed_state != data.landed_state:
        old_state = mavutil.mavlink.enums['MAV_LANDED_STATE'][self.extended_state.landed_state].name
        new_state = mavutil.mavlink.enums['MAV_LANDED_STATE'][data.landed_state].name
        rospy.loginfo("landed state changed from {0} to {1}".format(old_state, new_state))
    self.extended_state = data
    if not self.sub_topics_ready.get('extended_state'):
        self.sub_topics_ready['extended_state'] = True


def update_home_position(self, data):
    self.home_position = data
    if not self.sub_topics_ready.get('home_position'):
        self.sub_topics_ready['home_position'] = True


def update_local_position(self, data):
    self.local_position = data
    if not self.sub_topics_ready.get('local_position'):
        self.sub_topics_ready['local_position'] = True


def update_global_position(self, data):
    self.global_position = data
    if not self.sub_topics_ready.get('global_position'):
        self.sub_topics_ready['global_position'] = True


def update_imu_data(self, data):
    self.imu_data = data

    if not self.sub_topics_ready.get('imu_data'):
        self.sub_topics_ready['imu_data'] = True


def update_mission_wp(self, data):
    if self.mission_items.current_seq != data.current_seq:
        rospy.loginfo("current mission waypoint sequence updated: {0}".format(data.current_seq))

    self.mission_items = data

    if not self.sub_topics_ready['mission_items']:
        self.sub_topics_ready['mission_items'] = True


def update_mission_item_reached(self, data):
    if self.mission_item_reached != data.wp_seq:
        rospy.loginfo("mission item reached: {0}".format(data.wp_seq))
        self.mission_item_reached = data.wp_seq


def update_hil_state_quaternion(self, data):
    self.hil_state_quaternion = data

    if not self.sub_topics_ready['hil_state_quaternion']:
        self.sub_topics_ready['hil_state_quaternion'] = True


def update_gps1_raw(self, data):
    self.gps1_raw = data

    if not self.sub_topics_ready['gps1_raw']:
        self.sub_topics_ready['gps1_raw'] = True