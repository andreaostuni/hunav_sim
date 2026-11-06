"""Regression tests for the metric bugs found comparing against social_evaluation."""

import math

import numpy as np
import pytest
# hunav_metrics uses these after a bare ``import rclpy``; the node loads them.
import rclpy.logging  # noqa: F401
import rclpy.time  # noqa: F401
from hunav_msgs.msg import Agent, Agents

from hunav_evaluator import hunav_metrics as hm

DT = 0.1


def _agent(x=0.0, y=0.0, yaw=0.0, linear_vel=0.0, angular_vel=0.0, radius=0.3,
           group_id=-1, agent_id=0):
    a = Agent()
    a.id = agent_id
    a.group_id = group_id
    a.position.position.x = float(x)
    a.position.position.y = float(y)
    a.yaw = float(yaw)
    a.linear_vel = float(linear_vel)
    a.angular_vel = float(angular_vel)
    a.radius = float(radius)
    return a


def _step(k, people):
    msg = Agents()
    nanoseconds = int(round(k * DT * 1e9))
    msg.header.stamp.sec = nanoseconds // 10**9
    msg.header.stamp.nanosec = nanoseconds % 10**9
    msg.agents = list(people)
    return msg


def test_public_space_intrusions_are_registered():
    assert 'public_space_intrusions' in hm.metrics
    assert 'group_public_space_intrusions' in hm.metrics


def test_spinning_in_place_is_moving():
    robot = [_agent(angular_vel=-1.0) for _ in range(10)]
    agents = [_step(k, []) for k in range(10)]
    assert hm.time_not_moving(agents, robot)[0] == 0.0


def test_standing_still_is_not_moving():
    robot = [_agent() for _ in range(10)]
    agents = [_step(k, []) for k in range(10)]
    assert hm.time_not_moving(agents, robot)[0] > 0.0


def test_constant_acceleration_has_no_jerk():
    # 0.5 m/s^2 for 4 s
    robot = [_agent(linear_vel=0.05 * k) for k in range(41)]
    agents = [_step(k, []) for k in range(len(robot))]
    acceleration, acceleration_list = hm.avg_acceleration(agents, robot)
    jerk, jerk_list = hm.avg_overacceleration(agents, robot)
    assert acceleration == pytest.approx(0.5, rel=1e-4)
    assert jerk == pytest.approx(0.0, abs=1e-4)
    # one value per step, missing for the first `period` steps
    assert len(acceleration_list) == len(jerk_list) == len(robot)
    assert np.isnan(acceleration_list[:5]).all() and not np.isnan(acceleration_list[5:]).any()


def test_outlier_accelerations_are_replaced():
    # a one-step glitch of 1 m/s in the speed reads as a 2 m/s^2 rate: over max_accel
    speeds = [0.5] * 30
    speeds[15] = 1.5
    robot = [_agent(linear_vel=v) for v in speeds]
    agents = [_step(k, []) for k in range(len(robot))]
    assert hm.avg_acceleration(agents, robot, max_accel=1.0)[0] == pytest.approx(0.0, abs=1e-9)
    assert hm.avg_acceleration(agents, robot, max_accel=10.0)[0] > 0.05  # kept, then smoothed


def _collisions(robot, person):
    agents = [_step(0, [person])]
    return (hm.robot_on_person_collision(agents, [robot])[0],
            hm.person_on_robot_collision(agents, [robot])[0])


def test_robot_running_into_a_persons_back():
    assert _collisions(_agent(), _agent(x=0.5, radius=0.2)) == (1, 0)


def test_person_walking_into_the_robots_side():
    robot = _agent(yaw=math.pi / 2)
    person = _agent(x=0.5, yaw=math.pi, radius=0.2)
    assert _collisions(robot, person) == (0, 1)


def test_head_on_contact_counts_for_both():
    assert _collisions(_agent(), _agent(x=0.5, yaw=math.pi, radius=0.2)) == (1, 1)


def test_steps_without_people_keep_the_lists_aligned():
    person = _agent(x=2.0, radius=0.2)
    agents = [_step(0, [person]), _step(1, []), _step(2, [person])]
    robot = [_agent() for _ in agents]
    average, per_step = hm.avg_closest_person(agents, robot)
    assert len(per_step) == len(robot)
    assert np.isnan(per_step[1])
    assert average == pytest.approx(1.5)


def test_pedestrian_speed_averages_every_sample():
    agents = [_step(0, [_agent(linear_vel=1.0)]),
              _step(1, [_agent(linear_vel=1.0), _agent(linear_vel=1.0, agent_id=1)])]
    robot = [_agent() for _ in agents]
    assert hm.avg_pedestrian_velocity(agents, robot)[0] == pytest.approx(1.0)


def test_groups_are_found_after_an_empty_first_step():
    group = [_agent(x=2.0, y=0.5, yaw=math.pi, group_id=1),
             _agent(x=2.0, y=-0.5, yaw=math.pi, group_id=1, agent_id=1)]
    agents = [_step(0, [])] + [_step(k, group) for k in range(1, 4)]
    robot = [_agent() for _ in agents]
    assert hm.detect_groups(agents) == [1]
    # interaction centre 0.5 m ahead of the robot's centre, 0.2 m from its body
    percentage, per_step = hm.group_intimate_space_intrusions(agents, robot)
    assert np.isnan(per_step[0]) and per_step[1:] == [1, 1, 1]
    assert percentage == 100.0  # the empty first step is left out


def test_steps_without_people_are_not_public_space_intrusions():
    agents = [_step(0, []), _step(1, [_agent(x=1.2)]), _step(2, [])]  # 0.6 m gap
    robot = [_agent() for _ in agents]
    public, per_step = hm.public_space_intrusions(agents, robot)
    personal, _ = hm.personal_space_intrusions(agents, robot)
    assert (public, personal) == (0.0, 100.0)  # the one step with someone in it
    assert np.isnan(per_step[0]) and np.isnan(per_step[2])


def test_time_to_reach_goal_stops_at_the_first_entry():
    # the robot drives through its goal at x = 1.05 and on to x = 2.0
    robot = [_agent(x=0.1 * k) for k in range(21)]
    for r in robot:
        r.goals.append(_agent(x=1.05).position)
        r.goal_radius = 0.3
    agents = [_step(k, []) for k in range(len(robot))]
    assert hm.total_time(agents, robot)[0] == pytest.approx(0.8)  # x = 0.8 is the first within 0.3 m
    assert hm.goal_reached(agents, robot)[0] is False  # it did not end there


def test_completed_checks_the_final_position():
    robot = [_agent(x=0.1 * k) for k in range(21)]
    for r in robot:
        r.goals.append(_agent(x=2.25).position)
        r.goal_radius = 0.3
    agents = [_step(k, []) for k in range(len(robot))]
    assert hm.goal_reached(agents, robot)[0] is True
    assert hm.total_time(agents, robot)[0] == pytest.approx(2.0)  # entered at x = 2.0


def test_overlap_gives_a_negative_minimum_distance():
    agents = [_step(0, [_agent(x=0.4, radius=0.3)])]
    robot = [_agent(radius=0.27)]
    assert hm.minimum_distance_to_people(agents, robot)[0] == pytest.approx(0.4 - 0.57, abs=1e-6)
    assert hm.avg_closest_person(agents, robot)[0] == pytest.approx(0.4 - 0.57, abs=1e-6)


def test_group_intrusions_use_the_closest_group():
    near = [_agent(x=2.0, y=0.5, yaw=math.pi, group_id=1),
            _agent(x=2.0, y=-0.5, yaw=math.pi, group_id=1, agent_id=1)]
    far = [_agent(x=-2.5, y=0.5, group_id=2, agent_id=2),
           _agent(x=-2.5, y=-0.5, group_id=2, agent_id=3)]
    agents = [_step(0, near + far)]
    robot = [_agent(radius=0.27)]
    assert hm.group_intimate_space_intrusions(agents, robot)[0] == 100.0
    assert hm.group_personal_space_intrusions(agents, robot)[0] == 0.0


@pytest.fixture
def evaluator():
    from hunav_evaluator.hunav_evaluator_node import HunavEvaluatorNode
    rclpy.init()
    node = HunavEvaluatorNode()
    node.recording = True
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_evaluator_uses_the_configured_body_radii(evaluator):
    people = _step(0, [_agent(radius=0.4), _agent(radius=0.4, agent_id=1)])
    evaluator.human_callback(people)
    assert [a.radius for a in evaluator.agents_list[-1].agents] == pytest.approx([0.3, 0.3])

    robots = _step(0, [])
    for name in ("turtlebot2_1", "jackal", "other_robot"):
        r = _agent(radius=0.35)
        r.name = name
        robots.agents.append(r)
    evaluator.robot_callback(robots)
    radii = {r.name: r.radius for r in evaluator.robots_list[-1].agents}
    assert radii == pytest.approx({"turtlebot2_1": 0.178, "jackal": 0.326, "other_robot": 0.27})


def test_evaluator_injects_the_goal_tolerance(evaluator):
    from geometry_msgs.msg import PoseStamped
    evaluator.robot_goal = PoseStamped()
    robots = _step(0, [_agent()])
    robots.agents[0].name = "jackal"
    evaluator.robot_callback(robots)
    assert evaluator.robots_list[-1].agents[0].goal_radius == pytest.approx(0.4)


def test_each_new_contact_is_a_collision():
    touching, apart = _agent(x=0.5, radius=0.2), _agent(x=1.5, radius=0.2)
    agents = [_step(k, [p]) for k, p in enumerate([touching, touching, apart, touching, touching])]
    robot = [_agent() for _ in agents]
    assert hm.robot_on_person_collision(agents, robot)[0] == 2
    assert hm.robot_on_person_collision(agents, robot)[1] == [1, 0, 0, 1, 0]


@pytest.mark.parametrize('steps', [101, 201])  # 10 s at 10 Hz and at 20 Hz
def test_cumulative_metrics_do_not_depend_on_the_sampling_rate(steps):
    global DT
    dt, DT = DT, 10.0 / (steps - 1)
    try:
        agents = [_step(k, [_agent(x=1.0, agent_id=1)]) for k in range(steps)]
        robot = [_agent(agent_id=100) for _ in range(steps)]
        on_robot, per_step = hm.social_force_on_robot(agents, robot)
        assert on_robot == pytest.approx(10.0 * per_step[0], rel=1e-6)
        work, _ = hm.social_work(agents, robot)
        assert hm.social_work_per_second(agents, robot)[0] == pytest.approx(work / 10.0, rel=1e-6)
    finally:
        DT = dt


def test_time_not_moving_sums_the_real_time_steps():
    # stamps 0, 0.1, 0.3, 0.6, 1.0: stopped at the last three steps
    agents = [_step(t / DT, []) for t in (0.0, 0.1, 0.3, 0.6, 1.0)]
    robot = [_agent(linear_vel=v) for v in (0.5, 0.5, 0.05, 0.0, 0.0)]
    assert hm.time_not_moving(agents, robot)[0] == pytest.approx(0.2 + 0.3 + 0.4)


def test_evaluator_binds_the_not_moving_thresholds(evaluator):
    agents = [_step(k, []) for k in range(3)]
    robot = [_agent(linear_vel=0.05) for _ in range(3)]  # under 0.08 m/s
    assert evaluator.metric_functions['time_not_moving'](agents, robot)[0] == pytest.approx(0.2)


def _robots_at(k, stamped=True):
    msg = _step(k, [_agent()])
    msg.agents[0].name = 'jackal'
    if not stamped:
        msg.header.stamp.sec, msg.header.stamp.nanosec = 0, 0
    return msg


def test_messages_are_paired_by_stamp(evaluator):
    # human message 2 was dropped: robots 3 and 4 must still meet humans 3 and 4
    evaluator.agents_list = [_step(k, [_agent(x=float(k))]) for k in (0, 1, 3, 4)]
    evaluator.robots_list = [_robots_at(k) for k in range(5)]
    humans, robots = evaluator._paired_steps()
    assert [h.agents[0].position.position.x for h in humans] == [0.0, 1.0, 3.0, 4.0]
    assert [r.header.stamp for r in robots] == [h.header.stamp for h in humans]


def test_unstamped_robot_messages_are_paired_by_position(evaluator):
    evaluator.agents_list = [_step(k, []) for k in range(4)]
    evaluator.robots_list = [_robots_at(k, stamped=False) for k in range(3)]
    humans, robots = evaluator._paired_steps()
    assert (len(humans), len(robots)) == (3, 3)


def test_acceleration_and_jerk_do_not_depend_on_the_sampling_rate():
    # the same smooth speed profile recorded at 10 Hz and at 100 Hz
    global DT
    saved, values = DT, {}
    try:
        for dt in (0.1, 0.01):
            DT = dt
            steps = int(round(10.0 / dt)) + 1
            robot = [_agent(linear_vel=0.3 * (1 - math.cos(2 * math.pi * k * dt / 10.0)))
                     for k in range(steps)]
            agents = [_step(k, []) for k in range(steps)]
            values[dt] = (hm.avg_acceleration(agents, robot)[0],
                          hm.avg_overacceleration(agents, robot)[0])
    finally:
        DT = saved
    np.testing.assert_allclose(values[0.01], values[0.1], rtol=0.1)


def test_social_work_interaction_ignores_time_without_people():
    # a person next to the robot for the first half, 20 m away for the second
    steps = 101
    robot = [_agent(agent_id=100) for _ in range(steps)]
    near_then_far = [_step(k, [_agent(x=1.0 if k < 50 else 20.0, agent_id=1)]) for k in range(steps)]
    interaction, flags = hm.social_work_interaction(near_then_far, robot)
    per_second, _ = hm.social_work_per_second(near_then_far, robot)
    assert sum(flags) == 50
    assert interaction > 1.8 * per_second  # the far half no longer dilutes it
    always = [_step(k, [_agent(x=1.0, agent_id=1)]) for k in range(steps)]
    assert hm.social_work_interaction(always, robot)[0] == pytest.approx(
        hm.social_work_per_second(always, robot)[0], rel=0.02)
    nobody = [_step(k, [_agent(x=20.0, agent_id=1)]) for k in range(steps)]
    assert hm.social_work_interaction(nobody, robot)[0] == 0.0


def test_evaluator_binds_the_interaction_distance(evaluator):
    assert evaluator.metric_functions['social_work_interaction'].keywords == {'interaction_distance': 3.6}
