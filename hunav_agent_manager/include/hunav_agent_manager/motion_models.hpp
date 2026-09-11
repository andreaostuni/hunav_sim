#ifndef HUNAV_AGENT_MANAGER__MOTION_MODELS_HPP_
#define HUNAV_AGENT_MANAGER__MOTION_MODELS_HPP_

#include <memory>
#include <string>
#include <vector>

// Social Force Model (provides sfm::Agent, sfm::SFM, utils::Vector2d, ...)
#include <lightsfm/sfm.hpp>

namespace hunav
{

// Which motion model drives a human agent on the *regular* navigation path.
// Non-regular HuNav behaviors (surprised/scared/curious/threatening/impassive)
// always fall back to SFM, since they are expressed as SFM force/goal tweaks.
// NOTE: lightsfm defines `SFM` as a macro (the SocialForceModel singleton), so
// the enumerators must NOT be named SFM/CV/ORCA in all-caps to avoid macro
// expansion. We use Sfm/Cv/Orca.
enum class MotionModel
{
  Sfm,   // Social Force Model (default; current behavior)
  Cv,    // Constant Velocity (non-reactive baseline)
  Orca   // Optimal Reciprocal Collision Avoidance (RVO2) - Phase 3, not yet wired
};

// Parse a motion model name ("sfm" | "cv" | "orca", case-insensitive).
// Unknown or empty strings fall back to SFM.
MotionModel motionModelFromString(const std::string& s);
const char* motionModelToString(MotionModel m);

// Strategy that decides, per experiment/reset, WHICH MotionModel each agent
// uses. Layered on top of the per-agent selection:
//   Fixed  - classic behavior: default_motion_model + per-agent overrides,
//            resolved once and kept across resets.
//   Random - on every reset each agent re-draws its model uniformly from a
//            configurable list of choices.
//   Sweep  - systematic migration: experiment 0 moves every agent with the
//            `from` model; after each reset one more agent (by ascending id)
//            switches to the `to` model, until all agents use `to`.
enum class MotionModelStrategy
{
  Fixed,
  Random,
  Sweep
};

// Parse a strategy name ("fixed" | "random" | "sweep", case-insensitive).
// Unknown or empty strings fall back to Fixed.
MotionModelStrategy motionModelStrategyFromString(const std::string& s);
const char* motionModelStrategyToString(MotionModelStrategy s);

// Tunables for the ORCA / RVO2 model. Defaults match the previously
// hard-coded values. Exposed as ROS params (see hunav_loader / bt_node).
struct OrcaParams
{
  double neighbor_dist = 5.0;       // max center-to-center distance considered
  int max_neighbors = 10;           // max neighbors taken into account
  double time_horizon = 5.0;        // safety horizon w.r.t. other agents [s]
  double time_horizon_obst = 5.0;   // safety horizon w.r.t. obstacles [s]
  double obstacle_segment = 0.2;    // length of approximated obstacle wall [m]
};

// Strategy interface: (decide velocity, integrate pose) for one agent over dt.
// An implementation must write self.velocity, self.position, self.yaw,
// self.linearVelocity, self.angularVelocity and advance self.goals, mirroring
// the bookkeeping done by sfm::SFM.updatePosition so downstream consumers
// (getUpdatedAgentMsg, goal cycling) keep working unchanged.
//
// `neighbors` is the same set computeForces uses (all other humans + all
// robots), so reactive models compose with the multi-robot setup for free.
class AgentMotionModel
{
public:
  virtual ~AgentMotionModel() = default;
  virtual void update(sfm::Agent& self, const std::vector<sfm::Agent>& neighbors, double dt) = 0;
};

// SFM: integrate using the forces already computed by
// AgentManager::computeForces. Ignores `neighbors` (they are already baked into
// the forces). This is a pure pass-through to sfm::SFM.updatePosition, so SFM
// trajectories are byte-for-byte identical to the pre-refactor code.
class SFMModel : public AgentMotionModel
{
public:
  void update(sfm::Agent& self, const std::vector<sfm::Agent>& neighbors, double dt) override;
};

// CV: move straight toward the current goal at desiredVelocity, no neighbor
// reaction. The non-reactive baseline.
class CVModel : public AgentMotionModel
{
public:
  void update(sfm::Agent& self, const std::vector<sfm::Agent>& neighbors, double dt) override;
};

// ORCA (RVO2): per step, build a fresh RVO::RVOSimulator from self + neighbors
// (+ static obstacles), set the preferred velocity toward the goal, doStep, and
// read the collision-free velocity back. Reactive, velocity-based avoidance.
// The RVO2 headers are only needed in the .cpp, so they are not pulled in here.
class ORCAModel : public AgentMotionModel
{
public:
  explicit ORCAModel(const OrcaParams& params = OrcaParams{}) : params_(params)
  {
  }
  void update(sfm::Agent& self, const std::vector<sfm::Agent>& neighbors, double dt) override;

private:
  OrcaParams params_;
};

// Factory. `orca` is only used when m == MotionModel::Orca.
std::shared_ptr<AgentMotionModel> makeMotionModel(MotionModel m, const OrcaParams& orca = OrcaParams{});

}  // namespace hunav
#endif  // HUNAV_AGENT_MANAGER__MOTION_MODELS_HPP_
