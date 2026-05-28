#ifndef EXOSKELETRON_SAFETY_MANAGER_SENSOR_DEGRADED_MODE_HPP
#define EXOSKELETRON_SAFETY_MANAGER_SENSOR_DEGRADED_MODE_HPP

#include "exoskeleton_safety_manager/SafetyTools.hpp"
#include "exoskeleton_safety_manager/BridgeModeClient.hpp"

namespace functional_safety
{

/**
 * SensorDegradedMode — safety plugin for a "degraded sensing" condition.
 *
 * Used when one or more sensor channels are reporting unreliable values
 * (e.g. drift detected by the FDI Level 3 autoencoder, or a sustained
 * residual flagged by the FDI Level 2 detectors). The system is NOT in
 * mechanical danger, but the controller's input data is degraded and
 * the dynamic envelope must be reduced.
 *
 * Bridge mode for now: 'compliant'. This is a deliberate choice to avoid
 * extending the bridge with a new mode while the FDI is still being
 * developed. When the FDI is mature, a dedicated 'sensor_degraded' bridge
 * mode could be added that, for example, ignores the τ_ext sensor and
 * falls back to position-only control. Until then, sharing the compliant
 * bridge mode keeps the bridge code untouched.
 *
 * Severity ranking (see safety_states.hpp::severity_rank):
 *   FAULT_MONITOR < SENSOR_DEGRADED < COMPLIANT < TORQUE_LIMIT < SAFE_STOP
 *
 * SENSOR_DEGRADED is therefore LESS severe than COMPLIANT_MODE: a request
 * from the FDI to enter SENSOR_DEGRADED will not override a concurrent
 * COMPLIANT_MODE request from the FaultMonitorMode (which detects an
 * actual mechanical issue). The "max ordinal" arbitration policy in the
 * FDI supervisor honors this ordering.
 *
 * Bridge mode lifecycle (identical to CompliantMode):
 * - initialize(): requests bridge → 'compliant'
 * - stop() / resume(): re-asserts 'compliant' in case the bridge drifted
 * - shutdown(): does NOT send any mode request. The incoming plugin sets
 *               the correct bridge mode in its initialize().
 */
class SensorDegradedMode : public SafetyTools, protected BridgeModeClient
{
public:
  SensorDegradedMode() = default;
  ~SensorDegradedMode() override = default;

  void initialize(const rclcpp::Node::SharedPtr & node) override
  {
    init_bridge_client(node);
    RCLCPP_INFO(node_->get_logger(), "SensorDegradedMode initialized");
    request_mode("compliant");
  }
  void stop()     override { request_mode("compliant"); }
  void pause()    override {}
  void resume()   override { request_mode("compliant"); }

  void shutdown() override
  {
    if (node_) {
      RCLCPP_INFO(node_->get_logger(),
        "SensorDegradedMode shutdown (bridge mode delegated to next plugin)");
    }
  }

  void set_safety_params(double) override {}
};
}  // namespace functional_safety
#endif