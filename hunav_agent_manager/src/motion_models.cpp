#include "hunav_agent_manager/motion_models.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>

// RVO2 (ORCA). Provided by the FetchContent `RVO` target (see CMakeLists.txt).
#include "RVO.h"

namespace hunav
{

namespace
{
inline RVO::Vector2 toRVO(const utils::Vector2d& v)
{
  return RVO::Vector2(static_cast<float>(v.getX()), static_cast<float>(v.getY()));
}
}  // namespace

MotionModel motionModelFromString(const std::string& s)
{
  std::string l = s;
  std::transform(l.begin(), l.end(), l.begin(), [](unsigned char c) { return std::tolower(c); });
  if (l == "cv")
    return MotionModel::Cv;
  if (l == "orca")
    return MotionModel::Orca;
  // "sfm", empty or anything unknown -> SFM
  return MotionModel::Sfm;
}

const char* motionModelToString(MotionModel m)
{
  switch (m)
  {
    case MotionModel::Cv:
      return "cv";
    case MotionModel::Orca:
      return "orca";
    case MotionModel::Sfm:
    default:
      return "sfm";
  }
}

MotionModelStrategy motionModelStrategyFromString(const std::string& s)
{
  std::string l = s;
  std::transform(l.begin(), l.end(), l.begin(), [](unsigned char c) { return std::tolower(c); });
  if (l == "random")
    return MotionModelStrategy::Random;
  if (l == "sweep")
    return MotionModelStrategy::Sweep;
  // "fixed", empty or anything unknown -> Fixed
  return MotionModelStrategy::Fixed;
}

const char* motionModelStrategyToString(MotionModelStrategy s)
{
  switch (s)
  {
    case MotionModelStrategy::Random:
      return "random";
    case MotionModelStrategy::Sweep:
      return "sweep";
    case MotionModelStrategy::Fixed:
    default:
      return "fixed";
  }
}

void SFMModel::update(sfm::Agent& self, const std::vector<sfm::Agent>& neighbors, double dt)
{
  // Forces were already computed by AgentManager::computeForces; integrate them.
  (void)neighbors;
  sfm::SFM.updatePosition(self, dt);
}

void CVModel::update(sfm::Agent& self, const std::vector<sfm::Agent>& neighbors, double dt)
{
  // Non-reactive: head straight to the current goal at desiredVelocity.
  (void)neighbors;

  utils::Vector2d initPos = self.position;
  utils::Angle initYaw = self.yaw;

  if (!self.goals.empty() && (self.goals.front().center - self.position).norm() > self.goals.front().radius)
  {
    utils::Vector2d dir = (self.goals.front().center - self.position).normalized();
    self.velocity = dir * self.desiredVelocity;
    self.yaw = self.velocity.angle();
    self.antimove = false;
  }
  else
  {
    // No goal (or already inside it): stop, keep current heading.
    self.velocity.set(0.0, 0.0);
    self.antimove = true;
  }

  self.position += self.velocity * dt;

  // Mirror sfm::SFM.updatePosition bookkeeping so downstream consumers match.
  self.linearVelocity = self.velocity.norm();
  self.angularVelocity = (self.yaw - initYaw).toRadian() / dt;
  self.movement = self.position - initPos;

  // Advance / cycle goals exactly like the SFM integrator.
  if (!self.goals.empty() && (self.goals.front().center - self.position).norm() <= self.goals.front().radius)
  {
    sfm::Goal g = self.goals.front();
    self.goals.pop_front();
    if (self.cyclicGoals)
      self.goals.push_back(g);
  }
}

void ORCAModel::update(sfm::Agent& self, const std::vector<sfm::Agent>& neighbors, double dt)
{
  utils::Vector2d initPos = self.position;
  utils::Angle initYaw = self.yaw;

  // ORCA tuning (from ROS params; see OrcaParams / hunav_loader).
  const float neighborDist = static_cast<float>(params_.neighbor_dist);
  const std::size_t maxNeighbors = static_cast<std::size_t>(std::max(0, params_.max_neighbors));
  const float timeHorizon = static_cast<float>(params_.time_horizon);
  const float timeHorizonObst = static_cast<float>(params_.time_horizon_obst);
  const double obstHalfLen = 0.5 * params_.obstacle_segment;

  // A fresh simulator per step: self is agent 0, neighbors follow.
  RVO::RVOSimulator sim;
  sim.setTimeStep(static_cast<float>(dt));

  sim.addAgent(toRVO(self.position), neighborDist, maxNeighbors, timeHorizon, timeHorizonObst,
               static_cast<float>(self.radius), static_cast<float>(self.desiredVelocity), toRVO(self.velocity));

  // Neighbors keep their current velocity as preferred velocity, so ORCA
  // reacts reciprocally to where they are actually heading this step.
  for (const auto& n : neighbors)
  {
    const float nspeed = std::max(0.1f, static_cast<float>(n.desiredVelocity));
    std::size_t nid = sim.addAgent(toRVO(n.position), neighborDist, maxNeighbors, timeHorizon, timeHorizonObst,
                                   static_cast<float>(n.radius), nspeed, toRVO(n.velocity));
    sim.setAgentPrefVelocity(nid, toRVO(n.velocity));
  }

  // Static obstacles: approximate each closest-obstacle point as a short wall
  // segment perpendicular to the agent->obstacle direction.
  for (const auto& o : self.obstacles1)
  {
    utils::Vector2d d = o - self.position;
    if (d.norm() < 1e-6)
      continue;
    utils::Vector2d perp(-d.getY(), d.getX());
    perp = perp.normalized() * obstHalfLen;
    std::vector<RVO::Vector2> seg = { toRVO(o + perp), toRVO(o - perp) };
    sim.addObstacle(seg);
  }
  sim.processObstacles();

  // Preferred velocity: toward the current goal at desiredVelocity (else stop).
  RVO::Vector2 pref(0.0f, 0.0f);
  if (!self.goals.empty() && (self.goals.front().center - self.position).norm() > self.goals.front().radius)
  {
    utils::Vector2d dir = (self.goals.front().center - self.position).normalized();
    pref = toRVO(dir * self.desiredVelocity);
  }
  sim.setAgentPrefVelocity(0, pref);

  sim.doStep();

  // Read back the collision-free velocity and integrate.
  RVO::Vector2 v = sim.getAgentVelocity(0);
  self.velocity.set(v.x(), v.y());
  if (self.velocity.norm() > 1e-3)
  {
    self.yaw = self.velocity.angle();
    self.antimove = false;
  }
  else
  {
    self.antimove = true;
  }

  self.position += self.velocity * dt;

  // Mirror sfm::SFM.updatePosition bookkeeping.
  self.linearVelocity = self.velocity.norm();
  self.angularVelocity = (self.yaw - initYaw).toRadian() / dt;
  self.movement = self.position - initPos;

  if (!self.goals.empty() && (self.goals.front().center - self.position).norm() <= self.goals.front().radius)
  {
    sfm::Goal g = self.goals.front();
    self.goals.pop_front();
    if (self.cyclicGoals)
      self.goals.push_back(g);
  }
}

std::shared_ptr<AgentMotionModel> makeMotionModel(MotionModel m, const OrcaParams& orca)
{
  switch (m)
  {
    case MotionModel::Cv:
      return std::make_shared<CVModel>();
    case MotionModel::Orca:
      return std::make_shared<ORCAModel>(orca);
    case MotionModel::Sfm:
    default:
      return std::make_shared<SFMModel>();
  }
}

}  // namespace hunav
