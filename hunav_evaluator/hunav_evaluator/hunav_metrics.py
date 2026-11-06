import numpy as np
import pandas as pd
import math
import rclpy
from hunav_msgs.msg import Agents
from hunav_msgs.msg import Agent
from geometry_msgs.msg import Pose
from hunav_evaluator.sfm import SFM
from typing import List, Tuple
from angles import normalize_angle, shortest_angular_distance
from enum import Enum

# Teaching Robot Navigation Behaviors to Optimal RRT Planners
# Noé Pérez-Higueras, Fernando Caballero & Luis Merino


class SpaceType(Enum):
    INTIMATE = 0
    PERSONAL = 1
    SOCIAL = 2
    PUBLIC = 3


def indicator_function_space(norm: float, space_type: SpaceType) -> bool:
    """
    Determines if the distance norm falls within the specified space type.

    Args:
        norm (float): The distance norm to evaluate.
        space_type (SpaceType): The type of space to check against.

    Returns:
        bool: True if the norm is within the specified space type, False otherwise.
    """
    match space_type:
        case SpaceType.INTIMATE:
            return norm < 0.45
        case SpaceType.PERSONAL:
            return 0.45 <= norm < 1.2
        case SpaceType.SOCIAL:
            return 1.2 <= norm < 3.6
        case SpaceType.PUBLIC:
            return norm >= 3.6
    return False


def euclidean_distance(pose: Pose, pose1: Pose):
    return math.sqrt(
        (pose.position.x - pose1.position.x) ** 2
        + (pose.position.y - pose1.position.y) ** 2
    )


def agent_distance(agent: Agent, agent1: Agent):
    return (
        euclidean_distance(agent.position, agent1.position)
        - agent.radius
        - agent1.radius
    )


def get_group_center(agents_i: List[Agent], group_id: int, distance: float) -> Pose:
    """
    Computes the center of a group of agents at a specified distance from their current positions.
    Args:
        agents_i (List[Agent]): List of agents in the current time step.
        group_id (int): The group ID to filter agents.
        distance (float): The distance from the agent's position to compute the center.
    Returns:
        Pose: The computed center pose of the group at the specified distance.
    """
    if len(agents_i) == 0:
        raise ValueError("No agents found in the provided list.")
    group = []
    for agent in agents_i:
        if agent.group_id == group_id:
            pose = Pose()
            pose.position.x = agent.position.position.x + (
                distance * math.cos(agent.yaw)
            )
            pose.position.y = agent.position.position.y + (
                distance * math.sin(agent.yaw)
            )
            group.append(pose)

    interaction_center = Pose()
    for p in group:
        interaction_center.position.x += p.position.x
        interaction_center.position.y += p.position.y

    interaction_center.position.x = float(interaction_center.position.x / len(group))
    interaction_center.position.y = float(interaction_center.position.y / len(group))
    return interaction_center


def get_time_stamps(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """
    Extracts the time stamps from the agents and computes the time difference
    from the first agent's time stamp.
    Args:
        agents (List[Agents]): List of Agents messages containing the time stamps.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list of time differences in seconds from the first agent's time stamp.
    """
    time_list = []
    t0 = rclpy.time.Time.from_msg(agents[0].header.stamp)
    for a in agents:
        t = rclpy.time.Time.from_msg(a.header.stamp)
        dur = (t - t0).to_msg()
        s = float(dur.sec + dur.nanosec / 1e9)
        time_list.append(s)
    return time_list


def at_goal(r: Agent) -> bool:
    """Whether the robot's centre is within goal_radius (the goal tolerance) of a goal."""
    return any(euclidean_distance(r.position, g) < r.goal_radius for g in r.goals)


def time_integral(values: List[float], agents: List[Agents]) -> float:
    """Trapezoidal integral of per-step values over the message time stamps.

    The cumulative metrics integrate over time rather than sum per step, so they
    do not depend on how often the data was recorded.
    """
    t = np.asarray(get_time_stamps(agents, None), dtype=float)
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return 0.0
    return float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(t)))


def total_time(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """
    Computes the time until the robot first reaches its goal (its centre within
    goal_radius), or the whole recording when it never does.
    Args:
        agents (List[Agents]): List of Agents messages containing the time stamps.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the total time in seconds.
    """
    last = next((i for i, r in enumerate(robot) if at_goal(r)), len(agents) - 1)
    t2 = rclpy.time.Time.from_msg(agents[last].header.stamp)
    t1 = rclpy.time.Time.from_msg(agents[0].header.stamp)
    dur = t2 - t1
    secs = dur.nanoseconds / 1e9
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Time_to_reach_goal computed: {secs:.2f} secs"
    )
    return [secs]


def robot_path_length(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """
    Computes the total path length traveled by the robot.
    Args:
        agents (List[Agents]): List of Agents messages containing the robot's state.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the total path length in meters.
    """
    path_length = 0.0
    for i in range(len(robot) - 1):
        path_length += euclidean_distance(robot[i + 1].position, robot[i].position)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Path_length computed: {path_length:.2f} m"
    )
    return [path_length]


def cumulative_heading_changes(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """
    Computes the cumulative heading changes of the robot.
    Args:
        agents (List[Agents]): List of Agents messages containing the robot's state.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the cumulative heading changes in radians.
    """
    chc_list = [0.0]
    chc = 0
    for i in range(len(robot) - 1):
        norm = abs(shortest_angular_distance(robot[i].yaw, robot[i + 1].yaw))
        chc += norm
        chc_list.append(norm)

    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Cumulative_heading_changes: {chc:.2f} rads"
    )
    return [chc, chc_list]


def avg_closest_person(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """
    Computes the average distance to the closest person for each robot.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the average distance to the closest person
        and a list of minimum distances for each robot.
    """
    min_dist_list = []
    for i in range(len(robot)):
        min_dist = np.inf
        for agent in agents[i].agents:
            d = agent_distance(robot[i], agent)
            min_dist = min(min_dist, d)  # negative when the bodies overlap
        # A step without people has no closest person: NaN keeps the per-step
        # list aligned with the time stamps and out of the average.
        min_dist_list.append(min_dist if len(agents[i].agents) > 0 else np.nan)

    avg_dist = (
        float(np.nanmean(min_dist_list))
        if np.isfinite(min_dist_list).any()
        else 0.0
    )
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Average_closest_person: {avg_dist:.2f} m"
    )
    return [avg_dist, min_dist_list]


def minimum_distance_to_people(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """
    Computes the minimum distance to any person for each robot.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the minimum distance to any person for each robot.
    """
    min_distance = list()

    for i in range(len(robot)):
        for agent in agents[i].agents:
            d = agent_distance(robot[i], agent)
            min_distance.append(d)  # negative when the bodies overlap
    min_dist = min(min_distance)

    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Minimum_distance_to_people: {min_dist:.2f} m"
    )
    return [min_dist]


def maximum_distance_to_people(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """
    Computes the maximum distance to any person for each robot.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the maximum distance to any person for each robot.
    """
    max_distance = list()

    for i in range(len(robot)):
        for agent in agents[i].agents:
            max_distance.append(
                euclidean_distance(robot[i].position, agent.position)
                - robot[i].radius
                - agent.radius
            )

    max_dist = max(max_distance)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Maximum_distance_to_people: {max_dist:.2f} m"
    )

    return [max_dist]


def space_intrusions(
    agents: List[Agents], robot: List[Agent], k: SpaceType
) -> Tuple[float, List[int]]:
    """
    This function calculates the percentage of time the robot intrudes into
    the intimate, personal, or social space of other agents.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
        k (SpaceType): The type of space to check for intrusions (intimate, personal, social).
    Returns:
        Tuple[float, List[int]]: A tuple containing the percentage of space intrusions
        and a list indicating which robots had intrusions (1 for intrusion, 0 for no intrusion).
    """

    space_intrusions = 0
    measured = 0
    space_intrusions_list = [np.nan] * len(robot)

    for i in range(len(robot)):
        if not agents[i].agents:
            continue  # nobody to intrude on: the step is left out
        min_dist = min(agent_distance(robot[i], agent) for agent in agents[i].agents)
        indicator = int(indicator_function_space(min_dist, k))
        space_intrusions += indicator
        measured += 1
        space_intrusions_list[i] = indicator

    percentage = space_intrusions / measured * 100.0 if measured else 0.0

    return percentage, space_intrusions_list


def intimate_space_intrusions(
    agents: List[Agents], robot: List[Agent]
) -> Tuple[float, List[int]]:
    """
    Computes the percentage of time the robot intrudes into the intimate space of other agents.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the percentage of intimate space intrusions
        and a list indicating which robots had intrusions (1 for intrusion, 0 for no intrusion).
    """
    percentage, slist = space_intrusions(agents, robot, SpaceType.INTIMATE)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Intimate_space_intrusions: {percentage:.2f} % of the total time"
    )
    return [percentage, slist]


def personal_space_intrusions(
    agents: List[Agents], robot: List[Agent]
) -> Tuple[float, List[int]]:
    """
    Computes the percentage of time the robot intrudes into the personal space of other agents.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the percentage of personal space intrusions
        and a list indicating which robots had intrusions (1 for intrusion, 0 for no intrusion).
    """
    percentage, slist = space_intrusions(agents, robot, SpaceType.PERSONAL)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Personal_space_intrusions: {percentage:.2f} % of the total time"
    )
    return [percentage, slist]


def social_space_intrusions(
    agents: List[Agents], robot: List[Agent]
) -> Tuple[float, List[int]]:
    percentage, slist = space_intrusions(agents, robot, SpaceType.SOCIAL)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Social_space_intrusions: {percentage:.2f} % of the total time"
    )
    return [percentage, slist]


def public_space_intrusions(
    agents: List[Agents], robot: List[Agent]
) -> Tuple[float, List[int]]:
    percentage, slist = space_intrusions(agents, robot, SpaceType.PUBLIC)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Public_space_intrusions: {percentage:.2f} % of the total time"
    )
    return [percentage, slist]


def detect_groups(agents: List[Agents]) -> List[int]:
    """
    Detects unique group IDs from the agents and returns a list of those IDs.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
    Returns:
        List[int]: A list of unique group IDs found in the agents.
    """
    group_ids = []
    for step in agents:
        for a in step.agents:
            if a.group_id != -1 and a.group_id not in group_ids:
                group_ids.append(a.group_id)

    return group_ids


def group_space_intrusions(
    agents: List[Agents], robot: List[Agent], k: SpaceType
) -> Tuple[float, List[int]]:
    """
    Computes the percentage of time the robot intrudes into the intimate, personal, or social space
    of groups of agents.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
        k (SpaceType): The type of space to check for intrusions (intimate, personal, social).
    Returns:
        Tuple[float, List[int]]: A tuple containing the percentage of space intrusions
        and a list indicating which robots had intrusions (1 for intrusion, 0 for no intrusion).
    """
    group_ids = detect_groups(agents)
    if len(group_ids) == 0:
        return [0.0]

    d = 1.5
    space_intrusions = 0
    measured = 0
    group_list = [np.nan] * len(robot)
    for i in range(len(robot)):
        min_dist = np.inf
        for id in group_ids:
            if not any(a.group_id == id for a in agents[i].agents):
                continue  # no member of this group at this step
            group_center = get_group_center(agents[i].agents, id, d)
            dist = euclidean_distance(robot[i].position, group_center) - robot[i].radius
            min_dist = min(min_dist, dist)
        if min_dist == np.inf:
            continue  # no group at this step: it is left out
        indicator = int(indicator_function_space(min_dist, k))
        space_intrusions += indicator
        measured += 1
        group_list[i] = indicator

    percentage = space_intrusions / measured * 100.0 if measured else 0.0

    return [percentage, group_list]


def group_intimate_space_intrusions(
    agents: List[Agents], robot: List[Agent]
) -> List[float]:
    """Computes the percentage of time the robot intrudes into the intimate space of groups of agents.

    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the percentage of intimate space intrusions
        and a list indicating which robots had intrusions (1 for intrusion, 0 for no intrusion).
    """
    r = group_space_intrusions(agents, robot, SpaceType.INTIMATE)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Group_intimate_space_intrusions: {r[0]:.2f} % of the total time"
    )
    return r


def group_personal_space_intrusions(
    agents: List[Agents], robot: List[Agent]
) -> List[float]:
    """Computes the percentage of time the robot intrudes into the personal space of groups of agents.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the percentage of personal space intrusions
        and a list indicating which robots had intrusions (1 for intrusion, 0 for no intrusion).
    """
    r = group_space_intrusions(agents, robot, SpaceType.PERSONAL)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Group_personal_space_intrusions: {r[0]:.2f} % of the total time"
    )
    return r


def group_social_space_intrusions(
    agents: List[Agents], robot: List[Agent]
) -> List[float]:
    """Computes the percentage of time the robot intrudes into the social space of groups of agents.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the percentage of social space intrusions
        and a list indicating which robots had intrusions (1 for intrusion, 0 for no intrusion).
    """
    r = group_space_intrusions(agents, robot, SpaceType.SOCIAL)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Group_social_space_intrusions: {r[0]:.2f} % of the total time"
    )
    return r


def group_public_space_intrusions(
    agents: List[Agents], robot: List[Agent]
) -> List[float]:
    """Computes the percentage of time the robot intrudes into the public space of groups of agents.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the percentage of public space intrusions
        and a list indicating which robots had intrusions (1 for intrusion, 0 for no intrusion).
    """
    r = group_space_intrusions(agents, robot, SpaceType.PUBLIC)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Group_public_space_intrusions: {r[0]:.2f} % of the total time"
    )
    return r


# SEAN 2.0: Formalizing and Generating Social Situations for Robot Navigation
# Nathan Tsoi, Alec Xiang, Peter Yu, Samuel S. Sohn, Greg Schwartz, Subashri Ramesh, Mohamed Hussein, Anjali W. Gupta, Mubbasir Kapadia, and Marynel Vázquez

# The metrics Robot on Person Personal Distance Violation, Person on Robot Personal Distance Violation, Intimate Distance Violation and
# Person on Robot Intimate Distance Violation have already been implemented in the Personal_space_intrusions function.
# Instead of returning the number of times, it returns a percentage of distance violation.


def collisions(agents: List[Agents], robot: List[Agent]) -> List[int]:
    """Counts the collisions between the robot and persons. A step is a contact
    when the bodies are less than 0.02 m apart; a collision is each time the robot
    and a person come into contact (so the count does not depend on the sampling
    rate), attributed at its first contact step by heading.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the number of robot collisions, person collisions,
        and lists indicating which robots and persons had collisions (1 for collision, 0 for no collision).
    """
    robot_coll_list = [0] * len(robot)
    person_coll_list = [0] * len(robot)
    robot_collisions = 0
    person_collisions = 0
    in_contact = set()  # ids of the persons touching the robot at the previous step

    for i in range(len(robot)):
        touching = set()
        for agent in agents[i].agents:

            if (
                euclidean_distance(robot[i].position, agent.position)
                - robot[i].radius
                - agent.radius
                < 0.02
            ):
                touching.add(agent.id)
                if agent.id in in_contact:
                    continue  # the same collision, still going on

                # Robot's angle
                nrx = (
                    robot[i].position.position.x - agent.position.position.x
                ) * math.cos(agent.yaw) + (
                    robot[i].position.position.y - agent.position.position.y
                ) * math.sin(
                    agent.yaw
                )
                nry = -(
                    robot[i].position.position.x - agent.position.position.x
                ) * math.sin(agent.yaw) + (
                    robot[i].position.position.y - agent.position.position.y
                ) * math.cos(
                    agent.yaw
                )
                alpha = math.atan2(nry, nrx)

                # Agent's angle
                nrx = (
                    agent.position.position.x - robot[i].position.position.x
                ) * math.cos(robot[i].yaw) + (
                    agent.position.position.y - robot[i].position.position.y
                ) * math.sin(
                    robot[i].yaw
                )
                nry = -(
                    agent.position.position.x - robot[i].position.position.x
                ) * math.sin(robot[i].yaw) + (
                    agent.position.position.y - robot[i].position.position.y
                ) * math.cos(
                    robot[i].yaw
                )
                alpha2 = math.atan2(nry, nrx)

                # alpha: where the robot lies seen from the person; alpha2: where
                # the person lies seen from the robot. Whoever has the other more
                # in front ran into it; a tie counts for both.
                # (float32 yaws: a tie is equal to about 1e-7 rad)
                tie = math.isclose(abs(alpha), abs(alpha2), abs_tol=1e-6)
                if abs(alpha2) < abs(alpha) or tie:
                    robot_collisions += 1
                    robot_coll_list[i] = 1
                if abs(alpha) < abs(alpha2) or tie:
                    person_collisions += 1
                    person_coll_list[i] = 1
        in_contact = touching

    return robot_collisions, person_collisions, robot_coll_list, person_coll_list


def robot_on_person_collision(agents: List[Agents], robot: List[Agent]) -> List[int]:
    """Computes the number of collisions between the robot and persons.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the number of robot collisions with persons and a list indicating which robots had collisions (1 for collision, 0 for no collision).
    """
    collision = collisions(agents, robot)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Robot_on_person_collision: {collision[0]} times"
    )
    return [collision[0], collision[2]]


def person_on_robot_collision(agents: List[Agents], robot: List[Agent]) -> List[int]:
    """Computes the number of collisions between persons and the robot.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[int]: A list containing the number of person collisions with the robot and a list indicating which robots had collisions (1 for collision, 0 for no collision).
    """

    collision = collisions(agents, robot)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Person_on_robot_collision: {collision[1]} times"
    )
    return [collision[1], collision[3]]


def time_not_moving(
    agents: List[Agents],
    robot: List[Agent],
    min_velocity: float = 0.08,
    min_angular_velocity: float = 0.01,
) -> List[float]:
    """Computes the total time the robot is not moving: the time step before each
    step where its speed is under min_velocity and its angular speed under
    min_angular_velocity (the thresholds of social_evaluation's metrics.yaml).
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
        min_velocity (float): Speed (m/s) under which the robot is not moving.
        min_angular_velocity (float): Angular speed (rad/s) under which it is not turning.
    Returns:
        List[float]: A list containing the total time the robot is not moving in seconds and a list indicating which robots are not moving (1 for not moving, 0 for moving).
    """

    stamps = get_time_stamps(agents, robot)
    not_moving = [
        int(r.linear_vel < min_velocity and abs(r.angular_vel) < min_angular_velocity)
        for r in robot
    ]
    time_stopped = sum(
        stamps[i] - stamps[i - 1] for i in range(1, len(robot)) if not_moving[i]
    )
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Time_not_moving: {time_stopped:.2f} secs"
    )
    return [time_stopped, not_moving]


def goal_reached(agents: List[Agents], robot: List[Agent]) -> List[bool]:
    """Checks if the robot ended the run at its goal (SEAN 2.0): its final
    position within goal_radius (the goal tolerance) of the goal.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
        Returns:
        List[bool]: A list containing a boolean indicating whether the robot has reached its goal.
    """
    return [at_goal(robot[-1])]


def final_goal_distance(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """
    Computes the distance from the robot's final position to its goal.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the minimum distance to the goal for the robot.
    """
    min_dist = np.inf
    if len(robot[-1].goals):
        for g in robot[-1].goals:
            dist = euclidean_distance(robot[-1].position, g)
            min_dist = max(min(min_dist, dist), 0.0)
    return [min_dist]


def minimum_goal_distance(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """Computes the minimum distance from the robot's position to any of its goals.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the minimum distance to any goal for the robot.
    """
    min_dist = np.inf
    for r in robot:
        if len(r.goals):
            for g in r.goals:
                dist = euclidean_distance(r.position, g)
                min_dist = max(min(min_dist, dist), 0.0)  # Ensure non-negative distance
    return [min_dist]


# SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation
# ABHIJAT BISWAS, ALLAN WANG, GUSTAVO SILVERA, AARON STEINFELD, and HENNY AD-MONI, Carnegie Mellon University


def avg_robot_linear_speed(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """Computes the average robot linear speed based on the robot's linear velocities.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the average robot speed and a list of individual speeds.
    """
    speed_list = []
    speed = 0
    for r in robot:
        speed_list.append(r.linear_vel)
        speed += r.linear_vel

    speed = speed / len(robot)

    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Average_robot_speed: {speed:.2f} m/s"
    )
    return [speed, speed_list]


def avg_robot_angular_speed(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """Computes the average robot angular speed based on the robot's angular velocities.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the average robot angular speed and a list of individual angular speeds.
    """
    speed_list = []
    speed = 0
    for r in robot:
        speed_list.append(r.angular_vel)
        speed += np.abs(r.angular_vel)

    speed = speed / len(robot)

    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Average_robot_angular_speed: {speed:.2f} rad/s"
    )
    return [speed, speed_list]


def rate_filter(time: pd.Series, rate_window: float, smoothing_time: float):
    """The sample count and moving-average factor of a rate over ``rate_window``
    seconds smoothed with time constant ``smoothing_time``, at the median sample
    step (as social_evaluation computes them), so acceleration and jerk are
    comparable whatever the recording rate.
    """
    times = np.unique(time.dropna().to_numpy(dtype=float))
    dt = float(np.median(np.diff(times))) if len(times) > 1 else rate_window
    return max(1, int(round(rate_window / dt))), float(1.0 - np.exp(-dt / smoothing_time))


def _filtered_rate(
    time: pd.Series, value: pd.Series, period: int, limit: float, alpha: float
) -> pd.Series:
    """The rate of change of ``value``, filtered as social_evaluation filters it:
    differences over ``period`` samples, values above ``limit`` blanked and
    interpolated (cubic), then an exponential moving average of factor ``alpha``
    (both from rate_filter).
    Indexed like ``value``; samples where it is undefined are left out.
    """
    rate = value.diff(periods=period) / time.diff(periods=period)
    rate[rate.abs() > limit] = np.nan
    try:
        rate = rate.interpolate(method="cubic", order=3)
    except (ValueError, TypeError):  # too few samples for a cubic
        rate = rate.interpolate(method="linear")
    return rate.dropna().ewm(alpha=alpha).mean()


def avg_acceleration(
    agents: List[Agents],
    robot: List[Agent],
    rate_window: float = 0.5,
    max_accel: float = 2.5,
    smoothing_time: float = 0.949,
) -> List[float]:
    """Computes the average magnitude of the robot's acceleration, filtered as in
    social_evaluation (see _filtered_rate).
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
        rate_window (float): Seconds the speed is differenced over.
        max_accel (float): Accelerations (m/s^2) above this are treated as outliers.
        smoothing_time (float): Time constant (s) of the exponential moving average.
    Returns:
        List[float]: A list containing the average robot acceleration and a list of individual acceleration values.
    """
    time = pd.Series(get_time_stamps(agents, robot), dtype=float)
    speed = pd.Series([r.linear_vel for r in robot], dtype=float)
    period, alpha = rate_filter(time, rate_window, smoothing_time)
    acceleration = _filtered_rate(time, speed, period, max_accel, alpha)
    value = float(acceleration.abs().mean()) if len(acceleration) else 0.0

    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Average_robot_acceleration: {value:.5f} m/s^2"
    )
    return [value, acceleration.reindex(range(len(robot))).tolist()]


def avg_overacceleration(
    agents: List[Agents],
    robot: List[Agent],
    rate_window: float = 0.5,
    max_accel: float = 2.5,
    max_jerk: float = 2.5,
    smoothing_time: float = 0.949,
) -> List[float]:
    """Computes the average magnitude of the robot's jerk (over-acceleration): the
    rate of change of the filtered acceleration, filtered the same way, as in
    social_evaluation.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
        rate_window (float): Seconds each rate is differenced over.
        max_accel (float): Accelerations (m/s^2) above this are treated as outliers.
        max_jerk (float): Jerks (m/s^3) above this are treated as outliers.
        smoothing_time (float): Time constant (s) of the exponential moving average.
    Returns:
        List[float]: A list containing the average robot jerk (over-acceleration) and a list of individual jerk values.
    """
    time = pd.Series(get_time_stamps(agents, robot), dtype=float)
    speed = pd.Series([r.linear_vel for r in robot], dtype=float)
    period, alpha = rate_filter(time, rate_window, smoothing_time)
    acceleration = _filtered_rate(time, speed, period, max_accel, alpha)
    # the acceleration's own samples, renumbered as social_evaluation does
    jerk = _filtered_rate(
        time[acceleration.index].reset_index(drop=True),
        acceleration.reset_index(drop=True),
        period,
        max_jerk,
        alpha,
    )
    value = float(jerk.abs().mean()) if len(jerk) else 0.0
    jerk.index = acceleration.index[jerk.index]

    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Average_robot_jerk(over_acceleration): {value:.5f} m/s^3"
    )
    return [value, jerk.reindex(range(len(robot))).tolist()]


# Learning a Group-Aware Policy for Robot Navigation
# Kapil Katyal ∗1,2 , Yuxiang Gao ∗2 , Jared Markowitz 1 , Sara Pohland 3 , Corban Rivera 1 , I-Jeng Wang 1 , Chien-Ming Huang 2


def avg_pedestrian_velocity(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """Computes the average speed of pedestrians in the environment.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the average pedestrian speed and a list of individual pedestrian speeds.
    """
    speed = 0
    samples = 0
    speed_list = []
    for i in range(len(agents)):
        speed2 = 0.0
        for agent in agents[i].agents:
            speed += agent.linear_vel
            speed2 += agent.linear_vel
        samples += len(agents[i].agents)
        if len(agents[i].agents) > 0:
            speed_list.append(speed2 / len(agents[i].agents))
        else:
            speed_list.append(0.0)

    speed = speed / samples if samples else 0.0

    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Average_Pedestrian_speed: {speed:.2f} m/s"
    )
    return [speed, speed_list]


def avg_closest_pedestrian_velocity(agents, robot):
    speed = 0
    speed_list = []
    for i in range(len(robot)):
        min_dist = 10000
        closest = Agent()
        for agent in agents[i].agents:
            d = euclidean_distance(robot[i].position, agent.position)
            if d < min_dist:
                min_dist = d
                closest = agent
                if min_dist < 0.0:
                    min_dist = 0.0

        speed += closest.linear_vel
        speed_list.append(closest.linear_vel)
    if len(robot) > 0:
        speed = speed / len(robot)
    rclpy.logging.get_logger("hunav_evaluator").debug(
        f"Average_Closest_Pedestrian_speed: {speed:.2f} m/s"
    )
    return [speed, speed_list]


def avg_pedestrian_angle(agents, robot):
    pass


# metrics based on social force model


# cumulative modulus of the social force provoked
# by the robot in the agents
def social_force_on_agents(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """Computes the cumulative modulus of the social force provoked by the robot in the agents.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: The social force integrated over time (N s) and the force at each step.
    """
    sfm = SFM()
    sf_list = [sfm.modulusSocialForce2(rb, agts) for agts, rb in zip(agents, robot)]
    return [time_integral(sf_list, agents), sf_list]


# cumulative modulus of the social force provoked
# by the agents in the robot
def social_force_on_robot(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """Computes the cumulative modulus of the social force provoked by the agents in the robot.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: The social force integrated over time (N s) and the force at each step.
    """
    sfm = SFM()
    sf_list = [sfm.modulusSocialForce(rb, agts) for agts, rb in zip(agents, robot)]
    return [time_integral(sf_list, agents), sf_list]


# cumulative social work employed in this planner:
# https://github.com/robotics-upo/social_force_window_planner
def social_work(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """Computes the cumulative social work employed in the planner.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: The social work integrated over time (N s) and the work at each step.
    """
    sfm = SFM()
    sw_list = [sfm.computeSocialWork(rb, agts) for agts, rb in zip(agents, robot)]
    return [time_integral(sw_list, agents), sw_list]


def social_work_per_second(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """Computes the social work per second: its time integral over the recording time.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: The social work per second (N) and the work at each step.
    """
    total, sw_list = social_work(agents, robot)
    duration = get_time_stamps(agents, robot)[-1]
    return [total / duration if duration > 0 else 0.0, sw_list]


def social_work_interaction(
    agents: List[Agents], robot: List[Agent], interaction_distance: float = 3.6
) -> List[float]:
    """Computes the social work per second of interaction: the per-step social work averaged
    over time only on the steps where a person's body is closer than interaction_distance to
    the robot's (default: the social-space boundary, 3.6 m), as in social_evaluation, so steps
    with nobody around do not dilute it. 0 when nobody comes that close.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
        interaction_distance (float): Body gap (m) under which the robot interacts.
    Returns:
        List[float]: The social work per second of interaction (N) and, per step,
        1 when a person is within interaction_distance.
    """
    _, sw_list = social_work(agents, robot)
    stamps = np.asarray(get_time_stamps(agents, robot))
    interacting = np.array([any(agent_distance(r, a) < interaction_distance for a in step.agents)
                            for step, r in zip(agents, robot)])
    # each step stands for the interval before it
    weights = np.diff(stamps, prepend=stamps[0]) * interacting
    total = weights.sum()
    value = float(np.dot(sw_list, weights) / total) if total > 0 else 0.0
    return [value, interacting.astype(int).tolist()]


# cumulative obstacle force on the agents
def obstacle_force_on_agents(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """Computes the cumulative obstacle force on the agents.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the cumulative obstacle force on the agents
        and a list of individual obstacle forces for each agent.
    """
    sfm = SFM()
    of_list = []
    for agts in agents:
        forces = [np.linalg.norm(sfm.computeObstacleForce(a)) for a in agts.agents]
        of_list.append(float(np.mean(forces)) if forces else 0.0)
    return [time_integral(of_list, agents), of_list]


# cumulative obstacle force on the robot
def obstacle_force_on_robot(agents: List[Agents], robot: List[Agent]) -> List[float]:
    """
    Computes the cumulative obstacle force on the robot.
    Args:
        agents (List[Agents]): List of Agents messages containing the agents' states.
        robot (List[Agent]): List of Agent messages representing the robot's state.
    Returns:
        List[float]: A list containing the cumulative obstacle force on the robot
        and a list of individual obstacle forces for each robot.
    """
    sfm = SFM()
    of_list = [float(np.linalg.norm(sfm.computeObstacleForce(r))) for r in robot]
    return [time_integral(of_list, agents), of_list]


# TODO
def path_irregularity(agents, robot):
    pass


# TODO
def path_efficiency(agents, robot):
    pass


# TODO
def static_obs_collision(agents, robot):
    pass


# Evaluation of Socially-Aware Robot Navigation
# Yuxiang Gao * and Chien-Ming Huang
# Department of Computer Science, The Johns Hopkins University, Baltimore, MD, United States

# TODO:
# Average Displacement Error --> Trajectory needed
# Final Displacement Error --> Trajectory needed
# Asymmetric Dynamic Time Warping --> Trajectory needed
# Topological Complexity --> Path needed
# Path irregularity and Path efficiency are similar to the ones in SEAN paper .
# Personal space and o/p/r-space metrics are similar to the ones in Teaching Robot Navigation Behaviors to Optimal RRT Planners paper.


metrics = {
    # N. Perez-Higueras, F. Caballero, and L. Merino, “Teaching Robot Nav-
    # igation Behaviors to Optimal RRT Planners,” International Journal of
    # Social Robotics, vol. 10, no. 2, pp. 235–249, 2018.
    "time_to_reach_goal": total_time,
    "path_length": robot_path_length,
    "cumulative_heading_changes": cumulative_heading_changes,
    "avg_distance_to_closest_person": avg_closest_person,
    "minimum_distance_to_people": minimum_distance_to_people,
    "maximum_distance_to_people": maximum_distance_to_people,
    "intimate_space_intrusions": intimate_space_intrusions,
    "personal_space_intrusions": personal_space_intrusions,
    "social_space_intrusions": social_space_intrusions,
    "public_space_intrusions": public_space_intrusions,
    "group_intimate_space_intrusions": group_intimate_space_intrusions,
    "group_personal_space_intrusions": group_personal_space_intrusions,
    "group_social_space_intrusions": group_social_space_intrusions,
    "group_public_space_intrusions": group_public_space_intrusions,
    # N. Tsoi, A. Xiang, P. Yu, S. S. Sohn, G. Schwartz, S. Ramesh,
    # M. Hussein, A. W. Gupta, M. Kapadia, and M. V ́azquez, “Sean 2.0:
    # Formalizing and generating social situations for robot navigation,”
    # IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 11 047–
    # 11 054, 2022
    #   - 'Total Path Length' (meters): similar to 'path_length'
    #   - 'Path Irregularity': (radians): total rotations in the robot's
    #       traveled path greater than the total rotations in the search-based
    #       path from the starting pose.
    #   - 'Path Efficiency': (meters): ratio between robot's traveled path and
    #       geodesic distance of the search-based path from the starting pose.
    # true when the robot's final pose is within a specified distance of the goal.
    # The final distance threshold is easily adjustable by the user, but defaults
    # to 1.2m.
    "completed": goal_reached,
    # (meters): the closest the robot passes to the target position.
    "minimum_distance_to_target": minimum_goal_distance,
    # (meters): distance between the last robot position and the target position.
    "final_distance_to_target": final_goal_distance,
    #   - 'Robot on Person Personal Distance Violation': number of times a robot
    # approaches a person within the personal distance of the robot.
    # Similar to 'personal_space_intrusions'
    #   - 'Person on Robot Personal Distance Violation': number of times a person
    # approaches the robot within the personal distance of the robot.
    #   - 'Intimate Distance Violation': number of times the robot approached within
    # the intimate distance of a person.
    #   - 'Person on Robot Intimate Distance Violation': number of times a person
    # approaches the robot within the intimate distance of the robot.
    "robot_on_person_collision": robot_on_person_collision,
    "person_on_robot_collision": person_on_robot_collision,
    "time_not_moving": time_not_moving,
    # TODO: 'static_obstacle_collision': static_obs_collision,
    # number of times the robot collides with a static obstacle.
    # SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation
    # ABHIJAT BISWAS, ALLAN WANG, GUSTAVO SILVERA, AARON STEINFELD, and HENNY AD-MONI, Carnegie Mellon University
    "avg_robot_linear_speed": avg_robot_linear_speed,
    "avg_robot_angular_speed": avg_robot_angular_speed,
    "avg_acceleration": avg_acceleration,
    "avg_overacceleration": avg_overacceleration,
    # Learning a Group-Aware Policy for Robot Navigation
    # Kapil Katyal ∗1,2 , Yuxiang Gao ∗2 , Jared Markowitz 1 , Sara Pohland 3 , Corban Rivera 1 , I-Jeng Wang 1 , Chien-Ming Huang 2
    "avg_pedestrian_velocity": avg_pedestrian_velocity,
    "avg_closest_pedestrian_velocity": avg_closest_pedestrian_velocity,
    # metrics based on Social Force Model employed in different papers
    "social_force_on_agents": social_force_on_agents,
    "social_force_on_robot": social_force_on_robot,
    "social_work": social_work,
    "social_work_per_second": social_work_per_second,
    "social_work_interaction": social_work_interaction,
    "obstacle_force_on_robot": obstacle_force_on_robot,
    "obstacle_force_on_agents": obstacle_force_on_agents,
}
