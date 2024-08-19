import glob
import os
from os.path import join, split
import argparse

import rospy
import yaml

from utils.mavros import SimulationExperiment
from utils.mav_utils import AttackType

parser = argparse.ArgumentParser(description="Automated Flight Log Collection")
parser.add_argument('px4_path', action='store', help='path to px4 firmware folder')
parser.add_argument('mission_file', action='store', help='path to mission file')
parser.add_argument('--logger_topics', action='store', default=None,
                    help='path to logger topics file, will overwrite logger topics in the px4 folder')
parser.add_argument('-n', '--num_trial', action='store', default=1, type=int)
parser.add_argument('-s', '--sim_spd', action='store', default=1, type=int)
parser.add_argument('--show_simulation', action='store_true',
                    help="Set this option if one want to demonstrate the experiment process. "
                         "WARNING: extra performance cost, would slow down the simulation.")
parser.add_argument('--init_param', action='store', default=None, help="yaml files of initial parameter")
parser.add_argument('--detector_param', action='store', default=None,
                    help="yaml files of detector, will override init_param and attack_param (if exists)")
parser.add_argument('--attack_wp', action='store', default=None, type=int,
                    help="start attack after this waypoint index is reached")
parser.add_argument('--attack_timeout', action='store', default=10, type=int,
                    help="stop attack after timeout reached")
parser.add_argument('--attack_param', action='store', default=None,
                    help="yaml files for sensor attack, "
                         "if specified, it will require attack_wp is not None and ATK_APPLY_TYPE is greater than zero. "
                         "Will override init_param")
parser.add_argument('--max_deviation', action='store', default=5.0, type=float,
                    help="maximum distance deviation to terminate the sensor attack experiment, "
                         "useful for early termination")
parser.add_argument('--retry_count', action='store', default=10, type=int)
parser.add_argument('--log_dir', action='store', default=None, type=str,
                    help="specify the location of firmware flight logs stored")
parser.add_argument('--output_dir', action='store', default=None, type=str,
                    help="move the flight log to this location, "
                         "note: to avoid name clash the log file will rename to YYYY-MM-DD_hhmmss.ulg. \n"
                         "For example: log with path 2022-10-13/06_35_33.ulg will rename to 2022-10-13_063533.ulg")


class SimulationFlightTest(SimulationExperiment):
    def __init__(self, *args, max_deviation: float = 3.0, attack_apply_type: int = 0,
                 attack_wp_idx: int = None, attack_time_out_s: int = 10, **kwargs):
        super(SimulationFlightTest, self).__init__(*args, **kwargs)
        self.max_deviation = max_deviation
        self.attack_apply_type = attack_apply_type
        self.attack_wp_idx = attack_wp_idx
        self.attack_time_out_s = attack_time_out_s

    def after_waypoint(self, wp_idx, waypoint=None):
        if waypoint is None:
            waypoint = self.waypoints[wp_idx]
        if self.attack_wp_idx is not None and wp_idx == self.attack_wp_idx:
            if self.attack_apply_type != AttackType.NoAttack:
                # Execute attack with 5 second waiting
                self.ROS_Master.set_param('ATK_APPLY_TYPE', self.attack_apply_type, 5)
                self.ROS_Master.wait_till_deviated(self.max_deviation, self.attack_time_out_s, wp_idx)
                self.ROS_Master.set_param('ATK_APPLY_TYPE', 0, 5)
            else:
                # End mission immediately
                pass

            self._early_termination = True
        elif self.attack_wp_idx is None and wp_idx+1 >= len(self.waypoints):
            # Do not wait until land, because I can't handle landing crash at this implementation
            self._early_termination = True
        else:
            return super(SimulationFlightTest, self).after_waypoint(wp_idx, waypoint)

    def at_flight_end(self):
        rospy.loginfo("mission completed!")
        rospy.loginfo(f"log file {get_last_log(log_dir)}")
        return super(SimulationFlightTest, self).at_flight_end()


def get_last_log(logging_dir=None):
    if logging_dir is None:
        try:
            logging_dir = os.environ['PX4_LOG_DIR']
        except KeyError:
            try:
                logging_dir = os.path.join(os.environ['ROS_HOME'], 'log')
            except KeyError:
                logging_dir = os.path.join(os.environ['HOME'], '.ros/log')

    try:
        last_log_dir = sorted(glob.glob(os.path.join(logging_dir, '*')))[-1]
        last_log = sorted(glob.glob(os.path.join(last_log_dir, '*.ulg')))[-1]
        return last_log
    except IndexError:
        return ""


if __name__ == '__main__':
    arg_parser = parser.parse_args()
    num_trials = int(arg_parser.num_trial)

    log_dir = arg_parser.log_dir
    if log_dir is None:
        log_dir = join(arg_parser.px4_path, 'build/px4_sitl_default/logs')

    output_dir = arg_parser.output_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    log_topic_file = arg_parser.logger_topics
    LOG_TOPIC_DIR = join(arg_parser.px4_path, 'build/px4_sitl_default/etc/logging')
    if log_topic_file is not None:
        # Remove original logger_topics.txt, and create new symlink to it
        log_topic_file = os.path.abspath(log_topic_file)
        dst_topic_file = join(LOG_TOPIC_DIR, 'logger_topics.txt')
        backup_file = join(LOG_TOPIC_DIR, 'logger_topics-Backup.txt')
        os.makedirs(LOG_TOPIC_DIR, exist_ok=True)
        try:
            os.rename(dst_topic_file, backup_file)
        except FileNotFoundError:
            pass
        os.symlink(log_topic_file, dst_topic_file, False)

    init_params = {}
    if arg_parser.init_param is not None:
        with open(arg_parser.init_param, 'r') as f:
            init_params.update(yaml.load(f, Loader=yaml.SafeLoader))

    ATTACK_APPLY_TYPE = AttackType.NoAttack
    init_params['ATK_APPLY_TYPE'] = AttackType.NoAttack
    attack_timeout = arg_parser.attack_timeout
    if arg_parser.attack_param is not None:
        with open(arg_parser.attack_param, 'r') as f:
            attack_params = yaml.load(f, Loader=yaml.SafeLoader)
            if attack_params.get('ATK_APPLY_TYPE') is not None:
                ATTACK_APPLY_TYPE = attack_params.pop('ATK_APPLY_TYPE')
                init_params.update(attack_params)
            else:
                raise ValueError(f"Cannot get valid ATK_APPLY_TYPE params! Expect non-empty value, "
                                 f"got {attack_params.get('ATK_APPLY_TYPE')} instead.")

    if ATTACK_APPLY_TYPE != AttackType.NoAttack and arg_parser.attack_wp is None:
        raise ValueError("Please specify attack waypoint if launching attack")

    if arg_parser.detector_param is not None:
        with open(arg_parser.detector_param, 'r') as f:
            init_params.update(yaml.load(f, Loader=yaml.SafeLoader))

    headless = not arg_parser.show_simulation
    Exp = SimulationFlightTest(arg_parser.mission_file, arg_parser.px4_path, max_deviation=arg_parser.max_deviation,
                               headless=headless, sim_spd=arg_parser.sim_spd, flight_params=init_params,
                               attack_apply_type=ATTACK_APPLY_TYPE, attack_wp_idx=arg_parser.attack_wp, attack_time_out_s=attack_timeout)
    Exp.run_experiment(set_param_only=True)  # Initialize parameters
    last_log_before_experiment = get_last_log(log_dir)

    retry_count = int(arg_parser.retry_count)
    total_budget = num_trials
    while total_budget > 0:
        completed = Exp.run_experiment()
        total_budget -= 1
        if not completed:
            Exp.run_experiment(set_param_only=True)
            last_log_before_experiment = get_last_log(log_dir)
            if retry_count > 0:
                total_budget += 1
                retry_count -= 1
            continue

        if output_dir is not None:
            # Move the flight log to specified position
            try:
                current_last_log = get_last_log(log_dir)
                if current_last_log != last_log_before_experiment:
                    # We got new log, rename it and move to the new position
                    path_chunks, log_name = split(current_last_log)
                    _, date_name = split(path_chunks)
                    new_log_name = f"{date_name}_{log_name.replace('_', '')}"
                    os.rename(current_last_log, join(output_dir, new_log_name))
            except OSError or FileNotFoundError:
                pass

    Exp.teardown_experiment()
