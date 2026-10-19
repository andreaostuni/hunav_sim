import numpy as np
import sys
import os
from functools import partial
from itertools import combinations

import rclpy
from rclpy.node import Node
from hunav_evaluator import hunav_metrics

from hunav_msgs.msg import Agents
from hunav_msgs.msg import Agent
from hunav_msgs.srv import StartEvaluation
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseStamped
import pandas as pd


class HunavEvaluatorNode(Node):

    def __init__(self):
        super().__init__("hunav_evaluator_node")

        self.agents_list = []
        # One Agents message per time step, each holding every robot's state.
        self.robots_list = []
        self.robot_goal = None
        # Optional per-robot goals, keyed by robot name (fleet experiments).
        self.robot_goals = {}
        self.metrics_to_compute = {}
        self.metrics_lists = {}
        self.number_of_behaviors = 6

        # The user start/stop the recording through the
        #    the services /hunav_start_recording and
        #    /hunav_stop_recording.

        # Indicate the frequency of capturing the data
        # (it must be slower than data publishing).
        # If the value is set to zero, the data is captured
        # at the same frequency than it is published.
        self.freq = (
            self.declare_parameter("frequency", 0.0).get_parameter_value().double_value
        )

        # base name of the result files
        self.declare_parameter("result_file", "metrics")

        # Body radii used by every metric, in place of the radii in the HuNav
        # messages (0.4 m per person from agents.yaml, 0.35 m for every robot in
        # the Gazebo plugin), so this evaluator and social_evaluation measure the
        # same bodies. A robot takes the radius of the longest name prefix in
        # robot_radius_names (turtlebot2_1 -> turtlebot2), else robot_radius.
        self.agent_radius = self.declare_parameter("agent_radius", 0.3).value
        self.robot_radius = self.declare_parameter("robot_radius", 0.27).value
        self.robot_radii = dict(
            zip(
                self.declare_parameter(
                    "robot_radius_names", ["turtlebot2", "jackal"]
                ).value,
                self.declare_parameter("robot_radius_values", [0.178, 0.326]).value,
            )
        )
        # Goal tolerance (m) on the robot centre, for time_to_reach_goal and
        # completed; carried to the metrics in each robot's goal_radius.
        self.goal_tolerance = self.declare_parameter("goal_tolerance", 0.4).value
        # Thresholds under which the robot counts as not moving (time_not_moving).
        self.metric_functions = dict(hunav_metrics.metrics)
        self.metric_functions["time_not_moving"] = partial(
            hunav_metrics.time_not_moving,
            min_velocity=self.declare_parameter("min_velocity", 0.08).value,
            min_angular_velocity=self.declare_parameter("min_angular_velocity", 0.01).value,
        )
        # Filter of the acceleration and jerk metrics (as in social_evaluation).
        filter_params = dict(
            rate_window=self.declare_parameter("rate_window", 0.5).value,
            max_accel=self.declare_parameter("max_accel", 2.5).value,
            smoothing_time=self.declare_parameter("smoothing_time", 0.949).value,
        )
        self.metric_functions["avg_acceleration"] = partial(
            hunav_metrics.avg_acceleration, **filter_params
        )
        self.metric_functions["avg_overacceleration"] = partial(
            hunav_metrics.avg_overacceleration,
            max_jerk=self.declare_parameter("max_jerk", 2.5).value,
            **filter_params,
        )

        self.metrics_to_compute = self.get_metrics_to_compute()

        self.get_logger().info("Hunav evaluator:")
        self.get_logger().info(f"freq: {self.freq:.2f} Hz")
        self.get_logger().info("Metrics:")
        for m in self.metrics_to_compute.keys():
            self.get_logger().info(f"   {m}")

        if self.freq > 0.0:
            self.agents = Agents()
            self.robots = Agents()
            self.record_timer = self.create_timer(
                1 / self.freq, self.timer_record_callback
            )

        self.recording = False
        self.recording_service_start = self.create_service(
            StartEvaluation, "hunav_start_recording", self.recording_service_start
        )
        self.recording_service_stop = self.create_service(
            Empty, "hunav_stop_recording", self.recording_service_stop
        )

        # Queues of 10 so a slow callback drops fewer messages; the two streams
        # are paired by stamp afterwards (see _paired_steps).
        self.agent_sub = self.create_subscription(
            Agents, "human_states", self.human_callback, 10
        )
        # robot_states now carries every robot (one Agent per robot).
        self.robot_sub = self.create_subscription(
            Agents, "robot_states", self.robot_callback, 10
        )

    def recording_service_start(
        self, request: StartEvaluation.Request, response: StartEvaluation.Response
    ):
        if self.recording == True:
            self.get_logger().warn("Hunav evaluator already recording!")
            response.success = False
        else:
            self.get_logger().info("Hunav evaluator started recording!")
            self.recording = True
            self.robot_goal = (
                request.robot_goal if request.robot_goal != PoseStamped() else None
            )
            # Per-robot goals (parallel arrays matched by name). Robots absent
            # from this map fall back to robot_goal.
            self.robot_goals = {
                name: pose
                for name, pose in zip(request.robot_goal_names, request.robot_goals)
            }
            self.exp_tag = request.experiment_tag
            self.run_id = request.run_id
            self.agents_list.clear()
            self.robots_list.clear()
            self.last_time = self.get_clock().now()
            response.success = True
        return response

    def recording_service_stop(self, request, response):
        if self.recording == False:
            self.get_logger().warn("Hunav evaluator not recording!")
            return response

        self.get_logger().info("Hunav evaluator stopping recording!")
        self.recording = False
        self.compute_metrics()
        return response

    def human_callback(self, msg: Agents):
        """Callback for the human agents data.
        If the frequency is set to zero, the data is stored
        at the same frequency than it is published.
        If the frequency is greater than zero, the data is stored
        at the specified frequency.

        Args:
            msg (Agents): The message containing the human agents data."""
        if self.recording:
            for a in msg.agents:
                a.radius = self.agent_radius
            if self.freq == 0.0:
                self.agents_list.append(msg)
            else:
                self.agents = msg

    def robot_callback(self, msg: Agents):
        """Callback for the robots data (one Agent per robot).
        If the frequency is set to zero, the data is stored
        at the same frequency than it is published.
        If the frequency is greater than zero, the data is stored
        at the specified frequency.
        Args:
            msg (Agents): The message containing every robot's state."""
        if self.recording:
            robots_msg = msg
            # Inject each robot's goal: its own goal if provided, else the shared
            # robot_goal. Used by the goal-distance metrics.
            for r in robots_msg.agents:
                r.radius = self.radius_for(r.name)
                goal = self.robot_goals.get(r.name, self.robot_goal)
                if goal is not None:
                    r.goals.clear()
                    r.goals.append(goal.pose)
                    r.goal_radius = float(self.goal_tolerance)

            if self.freq == 0.0:
                self.robots_list.append(robots_msg)
            else:
                self.robots = robots_msg

    def radius_for(self, robot_name: str) -> float:
        """This robot's body radius: the longest matching name prefix, else robot_radius."""
        matches = [key for key in self.robot_radii if robot_name.startswith(key)]
        if matches:
            return float(self.robot_radii[max(matches, key=len)])
        return float(self.robot_radius)

    def timer_record_callback(self):
        """Timer callback to record the data at the specified frequency
        this is only used if the frequency is greater than zero."""
        if self.recording == True:
            self.agents_list.append(self.agents)
            self.robots_list.append(self.robots)

    def _paired_steps(self):
        """The recorded human and robot messages, paired step by step.

        HuNav publishes human_states and robot_states back to back with the same
        stamp, so each robot message is paired with the human message of its
        stamp: a message dropped on either topic then loses one step instead of
        shifting every later pair. Without unique robot stamps (a HuNav plugin
        built before it stamped robot_states) the messages are paired by
        position. With frequency > 0 the timer already stores them together.
        """
        if self.freq > 0.0:
            n = min(len(self.agents_list), len(self.robots_list))
            return self.agents_list[:n], self.robots_list[:n]

        def stamp(msg):
            return (msg.header.stamp.sec, msg.header.stamp.nanosec)

        robot_stamps = [stamp(m) for m in self.robots_list]
        if len(set(robot_stamps)) < len(robot_stamps):
            self.get_logger().warn(
                "robot_states stamps are not unique (HuNav plugin without robot "
                "stamps?): pairing the messages by position"
            )
            n = min(len(self.agents_list), len(self.robots_list))
            return self.agents_list[:n], self.robots_list[:n]

        humans = {stamp(m): m for m in self.agents_list}
        pairs = [
            (humans[key], robots)
            for key, robots in zip(robot_stamps, self.robots_list)
            if key in humans
        ]
        unmatched = len(self.robots_list) - len(pairs)
        if unmatched:
            self.get_logger().warn(
                f"{unmatched} robot_states messages have no human_states message "
                "with the same stamp; leaving them out"
            )
        return [h for h, _ in pairs], [r for _, r in pairs]

    def _robot_series(self):
        """Transpose robots_list (one Agents msg per step) into per-robot series.

        Returns:
            (names, series, agents_for) where
              names      : ordered list of robot names seen in the data;
              series     : {name -> [Agent, ...]} one entry per recorded step;
              agents_for : {name -> [Agents, ...]} the matching human messages
                           (same length as series[name]).
        """
        agents_list, robots_list = self._paired_steps()
        n = len(robots_list)

        names = []
        for ag in robots_list:
            for r in ag.agents:
                if r.name not in names:
                    names.append(r.name)

        series = {name: [] for name in names}
        agents_for = {name: [] for name in names}
        for i in range(n):
            by_name = {r.name: r for r in robots_list[i].agents}
            for name in names:
                if name in by_name:
                    series[name].append(by_name[name])
                    agents_for[name].append(agents_list[i])
        return names, series, agents_for

    def _compute_suite(self, agents_list, robot_list):
        """Run the full metric suite for a single robot trajectory.

        Returns (results, steps): results maps metric name -> overall value,
        steps maps metric name -> per-step list (plus 'time_stamps')."""
        results = {}
        steps = {"time_stamps": hunav_metrics.get_time_stamps(agents_list, robot_list)}
        for m in self.metrics_to_compute.keys():
            metric = self.metric_functions[m](agents_list, robot_list)
            results[m] = metric[0]
            if len(metric) > 1:
                steps[m] = metric[1]
        return results, steps

    def compute_metrics(self):
        """Compute the metrics based on the collected data.
        Runs the metric suite once per robot, writes one row per robot to the
        result file, and (for a fleet) also writes an aggregate and the
        inter-robot metrics."""
        if not self.check_data():
            self.get_logger().warn("Data not collected. Not computing metrics.")
            return

        names, series, agents_for = self._robot_series()
        if not names:
            self.get_logger().warn("No robots in the recorded data.")
            return

        self.get_logger().info(
            f"Hunav evaluator. Collected {len(self.agents_list)} agent messages "
            f"and {len(self.robots_list)} robot messages for {len(names)} robot(s): "
            f"{', '.join(names)}"
        )

        per_robot_results = {}
        for name in names:
            robot_list = series[name]
            agents_list = agents_for[name]
            results, steps = self._compute_suite(agents_list, robot_list)
            per_robot_results[name] = results
            # per-step file is per robot; the summary is one appended row per robot
            self.metrics_lists = steps
            self.store_metrics(self.result_file_path, results, robot=name)
            # behaviors, per robot
            for i in range(1, (self.number_of_behaviors + 1)):
                self.compute_metrics_behavior(i, agents_list, robot_list, name)

        rclpy.logging.get_logger("hunav_evaluator").info(
            f"Metrics computed for robots: {names}"
        )

        # Fleet-only outputs: aggregate across robots + inter-robot metrics.
        if len(names) > 1:
            self.store_aggregate(per_robot_results)
            self.store_inter_robot(names, series)

    def get_metrics_to_compute(self) -> dict:
        """Get the metrics to compute based on parameters.

        Returns:
            dict: A dictionary with the metrics to compute."""
        metrics_to_compute = {}
        for m in hunav_metrics.metrics.keys():
            ok = (
                self.declare_parameter("metrics." + m, True)
                .get_parameter_value()
                .bool_value
            )
            if ok:
                metrics_to_compute[m] = 0.0
        return metrics_to_compute

    def compute_metrics_behavior(self, behavior: int, agents_list, robot_list, robot_name):
        """Compute the metrics for a specific behavior, for one robot.

        Args:
            behavior (int): The behavior to compute the metrics for.
            agents_list (List[Agents]): human messages aligned with robot_list.
            robot_list (List[Agent]): this robot's trajectory.
            robot_name (str): the robot's name (used in the output file name).
        """

        # Keep only the steps where at least one agent shows this behavior.
        # beh_active is aligned to those kept steps (not to all steps), so it
        # matches the length of the per-step metric lists in the steps file.
        beh_agents = []  # list of Agents with the behavior
        beh_robot = []  # list of Agent with the behavior
        beh_active = []  # 1 if any agent is in an active behavior state, per kept step
        for la, lr in zip(agents_list, robot_list):
            ag = Agents()  # create a new Agents message
            ag.header = la.header  # copy the header from the Agents message
            active = 0
            for a in la.agents:  # iterate over the agents in the Agents message
                if a.behavior.type == behavior:  # check if the agent has the behavior
                    ag.agents.append(a)  # add the agent to the Agents message
                if a.behavior.state != a.behavior.BEH_NO_ACTIVE:  # agent is active
                    active = 1
            # Skip (don't abort) the steps with no agent of this behavior, so a
            # single empty step no longer discards the whole behavior's metrics.
            if len(ag.agents) > 0:
                beh_agents.append(ag)
                beh_robot.append(lr)
                beh_active.append(active)

        if not beh_agents:
            rclpy.logging.get_logger("hunav_evaluator").debug(
                f"Behavior {behavior} not present for robot {robot_name}; skipping."
            )
            return

        metrics = {}
        self.metrics_lists = {"behavior_active": beh_active}
        # then, compute the metrics for those agents
        for m in self.metrics_to_compute.keys():
            metric = self.metric_functions[m](beh_agents, beh_robot)
            metrics[m] = metric[0]
            if len(metric) > 1:  # if the metric function returns more than one value,
                self.metrics_lists[m] = metric[1]

        rclpy.logging.get_logger("hunav_evaluator").debug(
            f"Metrics computed for behavior {behavior}, robot {robot_name}"
        )
        store_file = self.result_file_path  # base name of the result file
        if not store_file.endswith(".csv"):
            store_file += ".csv"
        store_file = store_file.replace(".csv", f"_beh_{behavior}.csv")

        self.store_metrics(store_file, metrics, robot=robot_name)

    def store_metrics(self, result_file: str, metrics: dict, robot: str = ""):
        """Store the computed metrics for one robot in a file."""

        if not result_file.endswith(".csv"):
            result_file += ".csv"  # ensure the file has a .csv extension
        # add extension if it does not have it

        file_was_created = os.path.exists(result_file)
        # be sure thath the parent directory exists
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

        df_metrics = pd.DataFrame(metrics, index=[self.exp_tag])
        df_metrics.index.name = "experiment_tag"
        df_metrics["robot"] = robot  # which robot these metrics belong to
        df_metrics["run_id"] = self.run_id  # add the run id to the metrics

        # save the metrics to a CSV file
        df_metrics.to_csv(result_file, mode="a", header=not file_was_created)
        # open and write the second file (metric for each step)

        suffix = f"_{robot}" if robot else ""
        df_steps = pd.DataFrame(self.metrics_lists)
        df_steps.index.name = "time_stamps"
        # save the steps to a CSV file
        steps_csv_file = result_file.replace(
            ".csv", f"_steps_{self.exp_tag}_{self.run_id}{suffix}.csv"
        )
        df_steps.to_csv(steps_csv_file, index=True)
        self.get_logger().info(
            f"Metrics steps stored in {result_file} and {steps_csv_file}"
        )

    def store_aggregate(self, per_robot_results: dict):
        """Write a fleet aggregate (mean/min/max across robots) per metric."""
        agg = {}
        for m in self.metrics_to_compute.keys():
            vals = [float(per_robot_results[name][m]) for name in per_robot_results]
            agg[m + "_mean"] = float(np.mean(vals))
            agg[m + "_min"] = float(np.min(vals))
            agg[m + "_max"] = float(np.max(vals))

        agg_file = self.result_file_path
        if not agg_file.endswith(".csv"):
            agg_file += ".csv"
        agg_file = agg_file.replace(".csv", "_aggregate.csv")

        file_was_created = os.path.exists(agg_file)
        os.makedirs(os.path.dirname(agg_file), exist_ok=True)
        df = pd.DataFrame(agg, index=[self.exp_tag])
        df.index.name = "experiment_tag"
        df["n_robots"] = len(per_robot_results)
        df["run_id"] = self.run_id
        df.to_csv(agg_file, mode="a", header=not file_was_created)
        self.get_logger().info(f"Aggregate metrics stored in {agg_file}")

    def store_inter_robot(self, names, series):
        """Write inter-robot metrics: minimum inter-robot distance and the
        number of robot-on-robot collisions over the run."""
        steps = min(len(series[n]) for n in names)
        per_step_min = []
        min_dist = float("inf")
        collisions = 0
        for i in range(steps):
            step_min = float("inf")
            for a, b in combinations(names, 2):
                ra = series[a][i]
                rb = series[b][i]
                d = hunav_metrics.euclidean_distance(ra.position, rb.position)
                step_min = min(step_min, d)
                if d - ra.radius - rb.radius < 0.02:
                    collisions += 1
            per_step_min.append(step_min if step_min != float("inf") else 0.0)
            min_dist = min(min_dist, step_min)

        results = {
            "min_inter_robot_distance": (min_dist if min_dist != float("inf") else 0.0),
            "robot_on_robot_collisions": collisions,
        }

        inter_file = self.result_file_path
        if not inter_file.endswith(".csv"):
            inter_file += ".csv"
        inter_file = inter_file.replace(".csv", "_inter_robot.csv")

        file_was_created = os.path.exists(inter_file)
        os.makedirs(os.path.dirname(inter_file), exist_ok=True)
        df = pd.DataFrame(results, index=[self.exp_tag])
        df.index.name = "experiment_tag"
        df["run_id"] = self.run_id
        df.to_csv(inter_file, mode="a", header=not file_was_created)

        df_steps = pd.DataFrame({"min_inter_robot_distance": per_step_min})
        df_steps.index.name = "step"
        df_steps.to_csv(
            inter_file.replace(".csv", f"_steps_{self.exp_tag}_{self.run_id}.csv"),
            index=True,
        )
        self.get_logger().info(
            f"Inter-robot metrics stored in {inter_file} "
            f"(min dist {results['min_inter_robot_distance']:.2f} m, "
            f"{collisions} robot-robot collisions)"
        )

    def check_data(self) -> bool:
        """Check that the data is valid for computing the metrics."""
        # check if list is empty
        min_length = min(len(self.agents_list), len(self.robots_list))
        if min_length == 0:
            self.get_logger().error("No data collected. Cannot compute metrics.")
            return False
        # the two lists may differ in length: _paired_steps matches them up
        return True

    @property
    def result_file_path(self):
        """Get the result file path from the parameter."""
        return self.get_parameter("result_file").get_parameter_value().string_value


def main(args=None):
    rclpy.init(args=args)
    node = HunavEvaluatorNode()
    try:
        node.get_logger().info("Hunav evaluator node started.")
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        # Clean up and shutdown
        if rclpy.ok():
            node.get_logger().info("Shutting down Hunav evaluator node...")
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
