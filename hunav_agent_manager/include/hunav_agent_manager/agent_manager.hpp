
#ifndef HUNAV_BEHAVIORS__AGENT_MANAGER_HPP_
#define HUNAV_BEHAVIORS__AGENT_MANAGER_HPP_

#include "rclcpp/rclcpp.hpp"
// #include <ament_index_cpp/get_package_prefix.hpp>
// #include <ament_index_cpp/get_package_share_directory.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
//#include "nav_msgs/msg/OccupancyGrid.hpp"
// WATCH OUT! ALTOUGH THE NAME OF MESSAGE FILES HAVE
// CAPITAL LETTER (e.g. Agent and IsRobotVisible) THE
// FILES GENERATED FROM MESSAGE GENERATION ARE SNAKE CASE!
// ('Agent' goes to 'agent', and 'IsRobotVisible' goes to 'is_robot_visible')
#include "hunav_msgs/msg/agent.hpp"
#include "hunav_msgs/msg/agents.hpp"
//#include "hunav_msgs/srv/is_robot_visible.hpp"
// #include "hunav_msgs/srv/compute_agents.hpp"

// // Behavior Trees
// #include "behaviortree_cpp_v3/behavior_tree.h"
// #include "behaviortree_cpp_v3/bt_factory.h"
// #include "behaviortree_cpp_v3/loggers/bt_cout_logger.h"
// #include "behaviortree_cpp_v3/loggers/bt_file_logger.h"
// #include "behaviortree_cpp_v3/loggers/bt_minitrace_logger.h"
// #ifdef ZMQ_FOUND
// #include "behaviortree_cpp_v3/loggers/bt_zmq_publisher.h"
// #endif

#include <iostream>
//#include <memory>
#include <chrono>
#include <limits>
#include <math.h> /* fabs */
#include <mutex>
#include <random>
#include <string>
#include <vector>

// Social Force Model
#include <lightsfm/sfm.hpp>

// Pluggable human motion models (SFM / CV / ORCA)
#include "hunav_agent_manager/motion_models.hpp"

namespace hunav
{

struct agentBehavior
{
  int type = 1;
  int state = 0;
  int configuration = 0;  // def, manual, random
  double duration = 40.0;
  bool once = true;
  double vel = 0.0;
  double dist = 0.0;
  double forceFactor = 0.0;
};

struct agent
{
  std::string name;
  int type;
  // int behavior;
  // int behavior_state;
  agentBehavior behavior;
  sfm::Agent sfmAgent;
  // Motion model driving this agent on the regular navigation path.
  MotionModel motionModel = MotionModel::Sfm;
  std::shared_ptr<AgentMotionModel> motion_model;
};

class AgentManager
{
public:
  /**
   * @brief Construct a new Agent Manager object
   *
   */
  AgentManager();
  /**
   * @brief Destroy the Agent Manager object
   *
   */
  ~AgentManager();

  void init();

  // bool running();
  bool canCompute();

  /**
   * @brief method to update the agents
   *
   * @param msg
   */
  void updateAllAgents(const hunav_msgs::msg::Agents::SharedPtr robots_msg, const hunav_msgs::msg::Agents::SharedPtr msg);
  void resetAllAgents(const hunav_msgs::msg::Agents::SharedPtr robots_msg,
                      const hunav_msgs::msg::Agents::SharedPtr msg);
  /**
   * @brief method to update the agents
   *
   * @param msg
   */
  bool updateAgents(const hunav_msgs::msg::Agents::SharedPtr msg);
  /**
   * @brief method to update the robot
   *
   * @param msg
   */
  void updateAgentRobot(const hunav_msgs::msg::Agents::SharedPtr msg);
  /**
   * @brief method to update the robot
   *
   * @param msg
   */
  void updateAgentsAndRobot(const hunav_msgs::msg::Agents::SharedPtr msg);
  /**
   * @brief compute the forces of the sfm_agents_
   *
   */
  void computeForces();
  void computeForces(int id);
  /**
   * @brief initialize the sfm_agents_ based on the agents_ vector
   *
   */
  void initializeAgents(const hunav_msgs::msg::Agents::SharedPtr msg);
  /**
   * @brief initialize the srobot_ based on the agent msg of the robot
   *
   */
  void initializeRobot(const hunav_msgs::msg::Agents::SharedPtr msg);

  /**
   * @brief return the vector of agents in format of sfm lib
   *
   */
  std::vector<sfm::Agent> getSFMAgents();

  /**
   * @brief update the agent position by applying the forces
   * for a short time period
   *
   * @param id identifier of the agent
   * @param dt time to compute the agent's movement
   */
  void updatePosition(int id, double dt);
  /**
   * @brief build a Agents msg based on the info of the sfm_agents_
   * @return hunav_msgs::msg::Agents msg
   */
  hunav_msgs::msg::Agents getUpdatedAgentsMsg();

  /**
   * @brief build a Agent msg based on the id of the agent
   * @param id integer id of the agent
   * @return hunav_msgs::msg::Agent msg
   */
  hunav_msgs::msg::Agent getUpdatedAgentMsg(int id);

  /**
   * @brief return the forces of the agent
   * @param id integer id of the agent
   * @return sfm::Forces of the agent
   */
  sfm::Forces getAgentForces(int id)
  {
    return agents_[id].sfmAgent.forces;
  };

  /**
   * @brief computed the squared distance between the robot and the agent
   * indicated by the parameter id.
   *
   * @param id int value that is the agent id and the vector index of the agent
   * @return float
   */
  float robotSquaredDistance(int id);
  /**
   * @brief return a pointer to the robot closest to the agent indicated by id,
   * or nullptr if there are no robots. Used by the behaviors that must react to
   * "the robot" when more than one is present.
   *
   * @param id int value that is the agent id
   * @return const agent* nearest robot (nullptr if none)
   */
  const agent* nearestRobot(int id);
  /**
   * @brief stop the agent translation and changing its orientation to look at
   * the robot
   *
   * @param id int id of the desired agent (index of the agents_ array too)
   */
  void lookAtTheRobot(int id);
  /**
   * @brief check if the robot is in the field of view of the agent indicated by
   * the parameter id
   *
   * @param id int id of the desired agent (index of the agents_ array too)
   * @param dist maximum distance to detect the robot
   * @return true if the robot is the field of view of the agent
   * @return false otherwise
   */
  bool isRobotVisible(int id, double dist);
  /**
   * @brief check if the robot is in the line of sight of teh agent indicated by
   * the parameter id
   *
   * @param id int id of the desired agent (index of the agents_ array too)
   * @return true if the robot is in the line of sight
   * @return false otherwise
   */
  bool lineOfSight(int id);

  /**
   * @brief change the current navigation goal of the agent and its
   * maximum velocity so the agent approaches the robot
   *
   * @param id identifier of the agent
   * @param dt time to compute the agent's movement
   * @param closest_dist minimum distance between the robot and the agent
   */
  void approximateRobot(int id, double dt, double closest_dist = 1.5, double max_vel = 1.8);

  /**
   * @brief the agent will try to keep more distance from the robot
   *
   * @param id identifier of the agent
   * @param dt time to compute the agent's movement
   */
  void avoidRobot(int id, double dt, double scary_factor_force = 20.0, double max_vel = 0.6);

  /**
   * @brief the agent will try to block the path of the robot
   *
   * @param id identifier of the agent
   * @param dt time to compute the agent's movement
   */
  void blockRobot(int id, double dt, double front_dist = 1.4);

  bool goalReached(int id);
  bool updateGoal(int id);

  /**
   * @brief set the motion model used by all humans that do not have a
   * per-agent override. Must be called before the agents are initialized.
   * @param m motion model
   */
  void setDefaultMotionModel(MotionModel m);
  /**
   * @brief set the motion model for a single human (by name), overriding the
   * default. Patches the agent in place if it is already initialized, and is
   * remembered so it survives a re-initialization (reset).
   * @param name agent name
   * @param m motion model
   */
  void setAgentMotionModel(const std::string& name, MotionModel m);
  /**
   * @brief set the ORCA tuning parameters. Applied to ORCA agents created
   * afterwards; existing ORCA agents are re-created so the change takes effect.
   * @param p ORCA parameters
   */
  void setOrcaParams(const OrcaParams& p);

  /**
   * @brief set the cross-reset motion-model assignment strategy (fixed, random
   * or sweep). See MotionModelStrategy. Must be set before initialization.
   * @param s strategy
   */
  void setMotionModelStrategy(MotionModelStrategy s);
  /**
   * @brief set the list of motion models the `random` strategy draws from. An
   * empty list is ignored (the default {sfm, cv, orca} is kept).
   * @param choices motion models to sample uniformly
   */
  void setRandomMotionModelChoices(const std::vector<MotionModel>& choices);
  /**
   * @brief set the two motion models the `sweep` strategy migrates between:
   * experiment 0 uses `from` for everyone; each reset switches one more agent
   * to `to`.
   * @param from starting motion model
   * @param to target motion model
   */
  void setSweepMotionModels(MotionModel from, MotionModel to);
  /**
   * @brief enable cyclic `sweep`: once every agent has migrated to `to`, the
   * next reset wraps back to all-`from` and the migration repeats. When
   * disabled (default) the sweep saturates at all-`to` and stays there.
   * @param cycle true to loop the sweep, false to saturate
   */
  void setSweepCycle(bool cycle);
  /**
   * @brief seed the random generator used by the `random` strategy (and by the
   * random sweep order) so experiment sequences are reproducible.
   * @param seed seed value
   */
  void setMotionModelSeed(unsigned int seed);
  /**
   * @brief advance to the next experiment: bump the episode index and re-assign
   * every agent's motion model according to the active strategy. Call once per
   * reset (no-op for the `fixed` strategy). The agents must be initialized.
   */
  void resetEpisode();

  int step_count;
  int step_count2;
  bool move;

  inline double normalizeAngle(double a)
  {
    double value = a;
    while (value <= -M_PI)
      value += 2 * M_PI;
    while (value > M_PI)
      value -= 2 * M_PI;
    return value;
  }

  inline utils::Vector2d computeDesiredForce(sfm::Agent& agent) const
  {
    utils::Vector2d desiredDirection;
    if (!agent.goals.empty() && (agent.goals.front().center - agent.position).norm() > agent.goals.front().radius)
    {
      utils::Vector2d diff = agent.goals.front().center - agent.position;
      desiredDirection = diff.normalized();
      agent.forces.desiredForce = agent.params.forceFactorDesired *
                                  (desiredDirection * agent.desiredVelocity - agent.velocity) /
                                  agent.params.relaxationTime;
      agent.antimove = false;
    }
    else
    {
      agent.forces.desiredForce = -agent.velocity / agent.params.relaxationTime;
      agent.antimove = true;
    }
    return desiredDirection;
  }

protected:
  /**
   * @brief build the neighbor set for one agent: every other human plus every
   * robot, as sfm::Agent. Same membership computeForces uses. Caller must hold
   * mutex_ (used from updatePosition, which already locks).
   * @param id agent id to exclude as self
   */
  std::vector<sfm::Agent> getNeighbors(int id);

  /**
   * @brief (re)assign each agent's motion model according to mm_strategy_ and
   * the current episode_index_. No-op for the `fixed` strategy (the per-agent
   * resolution in initializeAgents already handles that case). Caller must hold
   * mutex_ (called from initializeAgents and resetEpisode).
   */
  void assignMotionModels();

  // Default model for humans without a per-agent override, and the set of
  // per-agent overrides (kept so resets re-apply them).
  MotionModel default_motion_model_ = MotionModel::Sfm;
  std::unordered_map<std::string, MotionModel> motion_model_overrides_;
  OrcaParams orca_params_;

  // Cross-reset motion-model assignment strategy (see MotionModelStrategy).
  MotionModelStrategy mm_strategy_ = MotionModelStrategy::Fixed;
  // Pool the `random` strategy samples from.
  std::vector<MotionModel> mm_random_choices_{ MotionModel::Sfm, MotionModel::Cv, MotionModel::Orca };
  // Endpoints of the `sweep` strategy migration.
  MotionModel mm_sweep_from_ = MotionModel::Sfm;
  MotionModel mm_sweep_to_ = MotionModel::Cv;
  // When true, the `sweep` loops back to all-`from` after reaching all-`to`
  // instead of saturating there.
  bool mm_sweep_cycle_ = false;
  // Experiment index: 0 for the first run, +1 on every reset. Drives the
  // deterministic sweep progression and re-draws for the random strategy.
  int episode_index_ = 0;
  // Random generator for the random strategy (seeded non-deterministically by
  // default; setMotionModelSeed makes runs reproducible).
  std::mt19937 mm_rng_{ std::random_device{}() };

  // std::vector<bool> agent_status_;
  // std::unordered_map<int, bool> agents_computed_;
  // int status_;
  bool agents_received_;
  bool robot_received_;
  bool agents_initialized_;
  bool robot_initialized_;
  std::mutex mutex_;
  // std::vector<hunav_msgs::msg::Agent> agents_;
  // std::vector<sfm::Agent> sfm_agents_;
  std::unordered_map<int, agent> agents_;
  // hunav_msgs::msg::Agent robot_;
  // One entry per robot (keyed by robot id). Replaces the old single robot_.
  std::unordered_map<int, agent> robots_;
  std_msgs::msg::Header header_;
  // sfm::Agent sfm_robot_;
  float max_dist_view_;
  float max_dist_view_squared_;
  double time_step_secs_;
  rclcpp::Time prev_time_;
  // rclcpp::Clock::SharedPtr clock_;

  // std::string pkg_shared_tree_dir_;
  // std::vector<BT::Tree> trees_;

  // // Topic subscriptions
  // rclcpp::Subscription<hunav_msgs::msg::Agents>::SharedPtr agents_sub_;

  // // Services provided
  // rclcpp::Service<hunav_msgs::srv::ComputeAgents>::SharedPtr
  // agents_srv_;
};

}  // namespace hunav
#endif  // HUNAV_BEHAVIORS__AGENT_MANAGER_HPP_
