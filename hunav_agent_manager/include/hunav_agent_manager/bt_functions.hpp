#include "hunav_agent_manager/agent_manager.hpp"

#include "rclcpp/rclcpp.hpp"
#include <ament_index_cpp/get_package_prefix.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "hunav_msgs/msg/agent.hpp"
#include "hunav_msgs/msg/agents.hpp"

// Behavior Trees
#include "behaviortree_cpp/behavior_tree.h"
#include "behaviortree_cpp/bt_factory.h"
#include "behaviortree_cpp/loggers/bt_cout_logger.h"
#include "behaviortree_cpp/loggers/bt_file_logger_v2.h"
#include "behaviortree_cpp/loggers/bt_minitrace_logger.h"
//#ifdef ZMQ_FOUND
//#include "behaviortree_cpp_v3/loggers/bt_zmq_publisher.h"
//#endif

#include <iostream>
//#include <memory>
#include <chrono>
#include <math.h> /* fabs */
#include <mutex>
#include <string>

// Social Force Model
#include <lightsfm/sfm.hpp>

namespace hunav {

class BTfunctions {
public:
  /**
   * @brief Construct a new BTfunctions object
   *
   */
  BTfunctions();
  /**
   * @brief Destroy the BTfunctions object
   *
   */
  ~BTfunctions();

  void init();

  void updateAllAgents(const hunav_msgs::msg::Agents::SharedPtr robots,
                       const hunav_msgs::msg::Agents::SharedPtr msg) {
    agent_manager_.updateAllAgents(robots, msg);
  }

  void resetAllAgents(const hunav_msgs::msg::Agents::SharedPtr robots,
                      const hunav_msgs::msg::Agents::SharedPtr msg) {
    agent_manager_.resetAllAgents(robots, msg);
  }

  void updateAgentsAndRobot(const hunav_msgs::msg::Agents::SharedPtr msg) {
    agent_manager_.updateAgentsAndRobot(msg);
  }

  hunav_msgs::msg::Agents getUpdatedAgents() {
    return agent_manager_.getUpdatedAgentsMsg();
  }
  hunav_msgs::msg::Agent getUpdatedAgent(int id) {
    return agent_manager_.getUpdatedAgentMsg(id);
  }

  // bool running();
  // bool ok() { return agent_manager_.canCompute(); }

  sfm::Forces getAgentForces(int id) {
    return agent_manager_.getAgentForces(id);
  };

  // Motion model selection (passthrough to the agent manager).
  void setDefaultMotionModel(MotionModel m) {
    agent_manager_.setDefaultMotionModel(m);
  }
  void setAgentMotionModel(const std::string &name, MotionModel m) {
    agent_manager_.setAgentMotionModel(name, m);
  }
  void setOrcaParams(const OrcaParams &p) {
    agent_manager_.setOrcaParams(p);
  }

  // Cross-reset motion-model strategy (passthrough to the agent manager).
  void setMotionModelStrategy(MotionModelStrategy s) {
    agent_manager_.setMotionModelStrategy(s);
  }
  void setRandomMotionModelChoices(const std::vector<MotionModel> &choices) {
    agent_manager_.setRandomMotionModelChoices(choices);
  }
  void setSweepMotionModels(MotionModel from, MotionModel to) {
    agent_manager_.setSweepMotionModels(from, to);
  }
  void setSweepCycle(bool cycle) {
    agent_manager_.setSweepCycle(cycle);
  }
  void setMotionModelSeed(unsigned int seed) {
    agent_manager_.setMotionModelSeed(seed);
  }
  void resetEpisode() {
    agent_manager_.resetEpisode();
  }

  // BT Conditions
  BT::NodeStatus robotVisible(BT::TreeNode &self);
  BT::NodeStatus goalReached(BT::TreeNode &self);

  // BT Actions to be registered with SimpleActionNode
  BT::NodeStatus updateGoal(BT::TreeNode &self);
  BT::NodeStatus regularNav(BT::TreeNode &self);
  BT::NodeStatus surprisedNav(BT::TreeNode &self);
  BT::NodeStatus scaredNav(BT::TreeNode &self);
  BT::NodeStatus curiousNav(BT::TreeNode &self);
  BT::NodeStatus threateningNav(BT::TreeNode &self);
  // the impassive behavior is taken into account
  // in the ComputeForces method, by adding the robot
  // to the agent's obstacles. So we do not need to
  // implement the method.
  // BT::NodeStatus impassiveNav(BT::TreeNode &self);

private:
  AgentManager agent_manager_;
};

} // namespace hunav
