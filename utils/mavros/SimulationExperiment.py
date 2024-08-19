import json
import os
from math import isfinite

import rosgraph
import rospy
import pexpect
from pymavlink import mavutil

from mavros_msgs.msg import Waypoint

from .MulticopterController import MultiCopterController

__all__ = ['SimulationExperiment']


class SimulationExperiment:
    def __init__(self, mission_path, px4_firmware_path, node_name='mavros', headless=False, sim_spd: int = 1,
                 core_port=11311, gcs_port=14540, mavlink_port=14580, flight_params: dict = None):
        self.AP_Worker = None
        self.SITL_Worker = None
        self.ROS_Master: MultiCopterController = None  # noqa
        self.waypoints = None

        self.mission_path = mission_path
        self.px4_path = px4_firmware_path
        self.node_name = node_name
        self.instance_id = None
        self.headless = headless
        self.sim_spd = sim_spd
        self.core_port = core_port
        self.gcs_port = gcs_port
        self.mavlink_port = mavlink_port
        self.flight_params = flight_params

        self._original_params = None
        self._is_running = False
        self._is_teardown = False
        self._early_termination = False

    def read_mission(self, file_path=None):
        if file_path is None:
            file_path = self.mission_path
        else:
            self.mission_path = file_path
        rospy.loginfo("reading mission {0}".format(file_path))

        wps = []
        with open(file_path, 'r') as f:
            for waypoint in read_plan_file(f):
                wps.append(waypoint)
                rospy.logdebug(waypoint)
        # set first item to current
        if wps:
            wps[0].is_current = True

        self.waypoints = wps

    def setup_experiment(self):
        init_ros(self.instance_id, self.core_port, self.gcs_port, self.mavlink_port)

        if self.waypoints is None:
            self.read_mission()

        home_lat = None
        home_lon = None
        for wp in self.waypoints:
            if wp.command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
                break
            if isfinite(wp.x_lat):
                home_lat = wp.x_lat
                rospy.loginfo(f"Set PX4_HOME_LAT={home_lat:.4f}")
            if isfinite(wp.y_long):
                home_lon = wp.y_long
                rospy.loginfo(f"Set PX4_HOME_LON={home_lon:.4f}")

        self.ROS_Master = MultiCopterController(self.node_name, self.sim_spd)
        self.AP_Worker, self.SITL_Worker = init_px4_jmavsim(self.px4_path, self.instance_id, self.headless,
                                                            self.sim_spd, home_lat, home_lon)
        self._is_running = True

    def teardown_experiment(self):
        self.stop_simulation()
        if not rospy.is_shutdown():
            rospy.signal_shutdown("Teardown Experiment! Good night ROS!")
        self._is_running = False

    def stop_simulation(self):
        if self.ROS_Master is not None:
            self.ROS_Master.close()
        if self.SITL_Worker is not None:
            self.SITL_Worker.close()
        if self.AP_Worker is not None:
            self.AP_Worker.close()

    def preflight_check(self):
        # make sure the simulation is ready to start the mission
        sim_ready = self.ROS_Master.wait_for_topics(60) and\
                    self.ROS_Master.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 10, -1) and\
                    self.ROS_Master.wait_for_mav_type(timeout=10)
        can_upload_mission = sim_ready and self.waypoints is not None and self.ROS_Master.clear_waypoints(timeout=5)
        mission_uploaded = can_upload_mission and self.ROS_Master.send_waypoints(self.waypoints, 30)

        if sim_ready and mission_uploaded and self.flight_params:
            for k, v in self.flight_params.items():
                self.ROS_Master.set_param(k, v, 1)

        return sim_ready and mission_uploaded

    def at_flight_begin(self):
        rospy.loginfo("run mission {0}".format(self.ROS_Master.mission_name))

    def reach_waypoint(self, wp_idx, waypoint=None):
        # only check position for waypoints where this makes sense
        if waypoint is None:
            waypoint = self.waypoints[wp_idx]

        if waypoint.command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT and\
                (waypoint.frame == Waypoint.FRAME_GLOBAL_REL_ALT or waypoint.frame == Waypoint.FRAME_GLOBAL):
            # Moving to next waypoint location
            altitude = waypoint.z_alt
            if waypoint.frame == waypoint.FRAME_GLOBAL_REL_ALT:
                altitude_msg = self.ROS_Master.altitude
                altitude += altitude_msg.amsl - altitude_msg.relative

            self.ROS_Master.reach_position(waypoint.x_lat, waypoint.y_long, altitude, 60, wp_idx)
        elif waypoint.command == mavutil.mavlink.MAV_CMD_NAV_LAND or\
                waypoint.command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            # Wait until landed
            self.ROS_Master.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 120, wp_idx)
        else:
            # Fallback to wait for next wp
            self.ROS_Master.wait_for_next_wp(wp_idx, 120)

    def after_waypoint(self, wp_idx, waypoint=None):
        if waypoint is None:
            waypoint = self.waypoints[wp_idx]
        # after reaching position, wait for landing detection if applicable

    def at_flight_end(self):
        if not self._early_termination:
            self.ROS_Master.set_arm(False, timeout=5)
        self.ROS_Master.clear_waypoints(timeout=5)
        self.stop_simulation()

        return True

    def run_experiment(self, set_param_only=False):
        self.setup_experiment()
        if not self.preflight_check():
            rospy.logerr("Failed to initiate Simulation!")
            self.stop_simulation()
            return False

        if set_param_only:
            rospy.loginfo("set_param_only=True, return early")
            self.stop_simulation()
            return True

        # Run auto mode and arm the plane
        if not self.ROS_Master.set_mode("AUTO.MISSION", 5) or not self.ROS_Master.set_arm(True, 5):
            rospy.logerr("Failed to arm the vehicle!")
            self.stop_simulation()
            return False

        # Start Mission
        self.at_flight_begin()

        # For each waypoint
        for index, waypoint in enumerate(self.waypoints):
            self.reach_waypoint(index, waypoint)
            self.after_waypoint(index, waypoint)
            if self._early_termination:
                break
        self.at_flight_end()
        self._early_termination = False
        return True


# Note: One rospy Node per process
def init_ros(instance_id=None, core_port=11311, gcs_port=14540, mavlink_port=14580, simulator_port=4560, **kwargs):
    if instance_id is None:
        instance_id = 0
    core_port += instance_id
    gcs_port += instance_id
    mavlink_port += instance_id

    if not rosgraph.is_master_online(f'http://localhost:{core_port}/'):
        launch_cmd = f"roscore -p {core_port}"
        pexpect.run(" ".join(["gnome-terminal", "-x", launch_cmd]))

    ros_cmd = " ".join([
        "roslaunch",
        '--wait',
        f"-p {core_port}",
        'mavros',
        'px4.launch',
        f"fcu_url:='udp://:{gcs_port}@localhost:{mavlink_port}'"
    ])
    pexpect.run(" ".join(["gnome-terminal", "-x", ros_cmd]))

    rospy.init_node("test_node", anonymous=True)
    node_name_actual = rospy.get_name()
    rospy.logdebug(f"node name: {node_name_actual}")


# TODO Multi instance
def init_px4_jmavsim(px4_path, instance_id=None, headless=False, sim_spd=None, home_lat=None, home_lon=None):
    env_var = os.environ.copy()
    if headless:
        env_var.update(HEADLESS=str(1))
        rospy.loginfo("SITL Headless Mode Enabled!")
    if home_lat is not None:
        env_var.update(PX4_HOME_LAT=str(home_lat))
    if home_lon is not None:
        env_var.update(PX4_HOME_LON=str(home_lon))
    if sim_spd is not None:
        env_var.update(PX4_SIM_SPEED_FACTOR=str(sim_spd))

    fcu_cmd = "gnome-terminal -x make px4_sitl none"
    if instance_id is not None:
        # TODO Running multiple px4 instance requires calling ./Tools/sitl_multiple_run.sh ${num_instance}
        #   - This would require input_batch processing across multiple ROSController
        #   - So I postpone it since running one instance already enough.
        # fcu_cmd += f"-i {instance_id}"
        pass
    else:
        instance_id = 0
    # Default port is 4560
    jmavsim_cmd = f'bash ./Tools/jmavsim_run.sh -p {4560 + instance_id} -l -r 250'
    autopilot_worker = pexpect.spawn(fcu_cmd, cwd=px4_path, encoding='utf-8', env=env_var)
    jmavsim_worker = pexpect.spawn(jmavsim_cmd, cwd=px4_path, encoding='utf-8', env=env_var)

    return autopilot_worker, jmavsim_worker


def read_plan_file(f):
    d = json.load(f)
    if 'mission' in d:
        d = d['mission']

    def nan_or_value(val):
        return 'nan' if val is None else val

    if 'items' in d:
        for wp in d['items']:
            yield Waypoint(
                is_current=False,
                frame=int(wp['frame']),
                command=int(wp['command']),
                param1=float(nan_or_value(wp['params'][0])),
                param2=float(nan_or_value(wp['params'][1])),
                param3=float(nan_or_value(wp['params'][2])),
                param4=float(nan_or_value(wp['params'][3])),
                x_lat=float(nan_or_value(wp['params'][4])),
                y_long=float(nan_or_value(wp['params'][5])),
                z_alt=float(nan_or_value(wp['params'][6])),
                autocontinue=bool(wp['autoContinue'])
            )
    else:
        raise IOError("no mission items")


if __name__ == '__main__':
    mission_file = 'ProjectDSS/data/missions/MC_mission_box.plan'
    px4_path = 'ProjectDSS/Firmwares/PX4-Autopilot'
    exp_instance = SimulationExperiment(mission_file, px4_path)
    exp_instance.setup_experiment()
    exp_instance.run_experiment()
    exp_instance.teardown_experiment()
