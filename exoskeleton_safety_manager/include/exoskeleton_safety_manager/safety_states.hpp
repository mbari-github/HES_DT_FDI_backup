#ifndef EXOSKELETRON_SAFETY_MANAGER_SAFETY_STATES_HPP
#define EXOSKELETRON_SAFETY_MANAGER_SAFETY_STATES_HPP

namespace functional_safety {

/**
 * SafetyState — enumeration of all possible states of the safety state machine.
 *
 * Allowed transitions (see StateMachine::can_transition_to):
 *
 *   FAULT_MONITOR        ──►  SENSOR_DEGRADED_MODE
 *                        ──►  COMPLIANT_MODE
 *                        ──►  TORQUE_LIMIT_MODE
 *                        ──►  SAFE_STOP
 *
 *   SENSOR_DEGRADED_MODE ──►  COMPLIANT_MODE
 *                        ──►  TORQUE_LIMIT_MODE
 *                        ──►  FAULT_MONITOR  (automatic downgrade or release)
 *                        ──►  SAFE_STOP
 *
 *   COMPLIANT_MODE       ──►  TORQUE_LIMIT_MODE
 *                        ──►  SENSOR_DEGRADED_MODE  (downgrade if applicable)
 *                        ──►  FAULT_MONITOR  (automatic downgrade)
 *                        ──►  SAFE_STOP
 *
 *   TORQUE_LIMIT         ──►  COMPLIANT_MODE        (automatic downgrade)
 *                        ──►  SENSOR_DEGRADED_MODE  (automatic downgrade)
 *                        ──►  FAULT_MONITOR        (automatic downgrade)
 *                        ──►  SAFE_STOP
 *
 *   SAFE_STOP            ──►  FAULT_MONITOR  (only via /reset_safety_request)
 *
 * Severity rank ordering (used by the FDI arbiter, NOT by state_to_id):
 *   FAULT_MONITOR (0) < SENSOR_DEGRADED (1) < COMPLIANT (2) < TORQUE_LIMIT (3) < SAFE_STOP (4)
 *
 * Note: severity_rank is distinct from state_to_id. The state_to_id mapping
 * preserves the historical IDs (0..3) used by the SafetyStatus.msg contract;
 * SENSOR_DEGRADED_MODE was added as ID=4 so existing consumers keep working.
 */
enum class SafetyState {
    FAULT_MONITOR,        ///< Normal monitoring — no active fault, FaultMonitorMode plugin loaded
    SAFE_STOP,            ///< Latched emergency stop — bridge set to 'stop', requires manual reset
    COMPLIANT_MODE,       ///< Reduced torque/velocity limits — bridge set to 'compliant'
    TORQUE_LIMIT_MODE,    ///< Hard torque cap — bridge set to 'torque_limit'
    SENSOR_DEGRADED_MODE  ///< Sensor degraded — bridge set to 'compliant' (least severe degraded mode)
};

/**
 * Severity ranking used by the FDI arbiter to compare requests from
 * multiple detectors. Higher rank = more severe state.
 *
 * This is intentionally separate from state_to_id() because:
 *   - state_to_id() is part of the wire contract (SafetyStatus.msg)
 *   - severity_rank is an internal ordering used for "max ordinal" arbitration
 */
inline int severity_rank(SafetyState s)
{
    switch (s) {
        case SafetyState::FAULT_MONITOR:        return 0;
        case SafetyState::SENSOR_DEGRADED_MODE: return 1;
        case SafetyState::COMPLIANT_MODE:       return 2;
        case SafetyState::TORQUE_LIMIT_MODE:    return 3;
        case SafetyState::SAFE_STOP:            return 4;
        default:                                return -1;
    }
}

}  // namespace functional_safety

#endif