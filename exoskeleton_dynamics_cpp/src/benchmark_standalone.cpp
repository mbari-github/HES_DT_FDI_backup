// Standalone C++ benchmark for exoskeleton reduced dynamics.
// No ROS — pure Pinocchio + Eigen + chrono.
// Mirrors the logic of dynamics_node_cpp.cpp with per-section nanosecond timing.
//
// Usage:
//   ./benchmark_standalone [urdf_path] [steps] [warmup] [output_csv]
//
// Defaults:
//   urdf_path  = <repo>/paper_benchmarks/urdf/assembly_with_hand.urdf
//   steps      = 50000
//   warmup     = 2000
//   output_csv = /tmp/benchmark_standalone_cpp.csv

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/force.hpp>
#pragma GCC diagnostic pop

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace pin = pinocchio;
using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

static inline int64_t now_ns()
{
  return std::chrono::duration_cast<ns>(Clock::now().time_since_epoch()).count();
}

// ============================================================
// Benchmark record
// ============================================================
static constexpr int N_PUB_JOINTS = 9;
static const char * PUB_JOINT_NAMES[N_PUB_JOINTS] = {
  "rev_crank", "rev_body2linkAC", "rev_crank2shaft", "slider",
  "rev_slider2linkBC", "rev_linkAC2linkCE",
  "rev_palmo2prossimale", "rev_prossimale2mediale", "slider2"
};

struct StepRecord {
  int64_t step;
  int64_t total_ns, closure_ns, compute_B_ns, crba_ns, ext_passive_ns, rnea_ns;
  double  closure_norm;
  int     nfev;
  double  at_limit;
  double  theta, theta_dot;
  double  q_joints[N_PUB_JOINTS];  // position of each published joint
};

// ============================================================
// Dynamics class (no ROS)
// ============================================================
class ExoBenchmarkDynamics
{
public:
  // Parameters (defaults mirror Python benchmark)
  double dt_            = 0.001;
  double motor_inertia_ = 0.1;
  double fric_visc_     = 2.0;
  double fric_coul_     = 2.0;
  double fric_eps_      = 0.005;
  double damping_theta_ = 1.0;
  double max_theta_dot_ = 10.0;
  double max_theta_ddot_= 400.0;
  double closure_tol_   = 1e-5;
  int    max_nfev_      = 150;
  double denom_min_     = 1e-7;
  double theta_min_     = -0.75;
  double theta_max_     =  0.09;
  bool   limit_hold_    = true;
  bool   limit_use_theta_bounds_ = true;
  double limit_release_tau_  = 0.02;
  double limit_backoff_step_ = 0.003;
  int    limit_backoff_tries_= 6;
  bool   wrench_enable_ = false;
  bool   passive_enable_= true;
  double K0_MCF_ = 0.03, alpha_MCF_ = 4.0, B_MCF_ = 0.01, rest_MCF_ = -0.3;
  double K0_IFP_ = 0.03, alpha_IFP_ = 4.0, B_IFP_ = 0.01, rest_IFP_ = -0.5;
  double passive_K_MAX_  = 5.0;
  double passive_exp_clip_= 12.0;
  bool   medial_soft_enable_ = true;
  double medial_soft_lower_  = -2.2;
  double medial_soft_upper_  = 0.0;
  double medial_soft_weight_ = 1.0;

  // Pinocchio
  pin::Model model_;
  pin::Data  data_;
  int nq_, nv_;
  int jid_crank_, idx_theta_;
  int jid_MCF_, jid_IFP_;

  std::vector<std::pair<int,int>> closure_frame_pairs_;
  std::vector<int>   idx_opt_;
  Eigen::VectorXd    lower_, upper_;
  int                fid_ext_ = -1;

  // State
  double          theta_, theta_dot_, theta_ddot_;
  Eigen::VectorXd q_, dq_, ddq_;
  Eigen::VectorXd last_x_, B_, B_prev_;
  Eigen::VectorXd wrench_world_;
  Eigen::VectorXd tau_ext_, tau_pass_, tau_full_;
  double          tau_ext_theta_ = 0.0, tau_pass_theta_ = 0.0;

  // Diagnostics
  bool   last_solver_success_ = false;
  int    last_solver_nfev_    = 0;
  double last_closure_norm_   = std::numeric_limits<double>::quiet_NaN();
  bool   last_hit_bounds_     = false;
  double denom_last_ = 0.0, proj_last_ = 0.0, gproj_last_ = 0.0;

  // Limit state
  bool            at_limit_   = false;
  double          limit_dir_  = 0.0;
  bool            have_valid_ = false;
  double          theta_valid_= 0.0;
  Eigen::VectorXd q_valid_, dq_valid_, B_valid_, last_x_valid_;

  // Cached publish joint indices (same as dynamics_node_cpp)
  int pub_joint_idxs_[N_PUB_JOINTS];

  // ── Constructor ──
  explicit ExoBenchmarkDynamics(const std::string & urdf_path, double theta_init = 0.0)
  {
    pin::urdf::buildModel(urdf_path, model_);
    data_ = pin::Data(model_);
    nq_ = model_.nq;
    nv_ = model_.nv;

    auto check_joint = [&](const std::string & name) -> int {
      size_t jid = model_.getJointId(name);
      if (jid >= static_cast<size_t>(model_.njoints))
        throw std::runtime_error("Joint '" + name + "' not in URDF");
      return static_cast<int>(jid);
    };

    jid_crank_ = check_joint("rev_crank");
    idx_theta_ = jid_crank_ - 1;
    jid_MCF_   = check_joint("rev_palmo2prossimale");
    jid_IFP_   = check_joint("rev_prossimale2mediale");

    // Closure frame pairs
    std::vector<std::pair<std::string,std::string>> pair_names = {
      {"frame_AC_end",   "frame_rod_end"},
      {"frame_AC_end_2", "frame_BC_end"},
      {"frame_CE_end",   "frame_BC_end_2"},
      {"frame_CE_end_2", "slider_t"},
    };
    for (auto & [a, b] : pair_names) {
      int ida = static_cast<int>(model_.getFrameId(a));
      int idb = static_cast<int>(model_.getFrameId(b));
      if (ida >= 0 && idb >= 0)
        closure_frame_pairs_.push_back({ida, idb});
    }

    // Dependent DOFs
    for (const auto & n : std::vector<std::string>{
        "rev_body2linkAC","rev_crank2shaft","slider","rev_slider2linkBC",
        "rev_linkAC2linkCE","rev_palmo2prossimale","rev_prossimale2mediale","slider2"})
      idx_opt_.push_back(static_cast<int>(model_.getJointId(n)) - 1);

    lower_ = (Eigen::VectorXd(8) << -2.5,-2.5,-0.015,-2.5,-2.5,-1.5,-2.5,-0.008).finished();
    upper_ = (Eigen::VectorXd(8) <<  2.5, 2.5, 0.004, 2.5, 2.5, 1.2, 2.5, 0.008).finished();

    // Cache published joint indices
    for (int k = 0; k < N_PUB_JOINTS; k++) {
      size_t jid = model_.getJointId(PUB_JOINT_NAMES[k]);
      if (jid >= static_cast<size_t>(model_.njoints))
        throw std::runtime_error(std::string("Pub joint not found: ") + PUB_JOINT_NAMES[k]);
      pub_joint_idxs_[k] = static_cast<int>(jid) - 1;
    }

    // Init state
    theta_ = theta_init; theta_dot_ = 0.0; theta_ddot_ = 0.0;
    q_   = pin::neutral(model_);
    dq_  = Eigen::VectorXd::Zero(nv_);
    ddq_ = Eigen::VectorXd::Zero(nv_);
    last_x_ = Eigen::VectorXd::Zero(static_cast<int>(idx_opt_.size()));
    B_      = Eigen::VectorXd::Zero(nv_);
    B_prev_ = Eigen::VectorXd::Zero(nv_);
    wrench_world_ = Eigen::VectorXd::Zero(6);
    tau_ext_  = Eigen::VectorXd::Zero(nv_);
    tau_pass_ = Eigen::VectorXd::Zero(nv_);
    tau_full_ = Eigen::VectorXd::Zero(nv_);
    q_valid_ = q_; dq_valid_ = dq_; B_valid_ = B_; last_x_valid_ = last_x_;

    // Initial closure
    auto res0 = solve_closure(theta_, true);
    if (res0.success) {
      q_[idx_theta_] = theta_;
      for (int k = 0; k < static_cast<int>(idx_opt_.size()); k++)
        q_[idx_opt_[k]] = res0.x[k];
      B_ = compute_B(q_);
      store_valid_state();
    } else {
      at_limit_ = true;
    }
  }

  // ── Helpers ──
  void store_valid_state() {
    have_valid_   = true;
    theta_valid_  = theta_;
    q_valid_      = q_;
    dq_valid_     = dq_;
    B_valid_      = B_;
    last_x_valid_ = last_x_;
  }

  double clamp_theta(double t) const {
    if (!limit_use_theta_bounds_) return t;
    return std::clamp(t, theta_min_, theta_max_);
  }

  // ── Closure residual ──
  Eigen::VectorXd closure_residual(const Eigen::VectorXd & q_loc) {
    pin::forwardKinematics(model_, data_, q_loc);
    pin::updateFramePlacements(model_, data_);
    int n = static_cast<int>(closure_frame_pairs_.size());
    Eigen::VectorXd r(3 * n);
    for (int i = 0; i < n; i++) {
      auto [ida, idb] = closure_frame_pairs_[i];
      r.segment(3*i, 3) = data_.oMf[ida].translation() - data_.oMf[idb].translation();
    }
    return r;
  }

  // ── Active-set projected Gauss-Newton closure solver ──
  struct SolverResult { bool success; int nfev; double closure_norm; bool hit_bounds; Eigen::VectorXd x; };

  SolverResult solve_closure(double theta, bool update_warmstart) {
    SolverResult info{false, 0, std::numeric_limits<double>::infinity(), false, {}};
    Eigen::VectorXd q = q_;
    q[idx_theta_] = theta;

    if (closure_frame_pairs_.empty()) {
      info.success = true;
      info.nfev = 0;
      info.closure_norm = closure_residual(q).norm();
      info.x = last_x_;
      return info;
    }

    int n_dep = static_cast<int>(idx_opt_.size());
    int n_con = static_cast<int>(closure_frame_pairs_.size()) * 3;
    Eigen::VectorXd x = last_x_;
    constexpr double eps_b = 1e-9;

    for (int iter = 0; iter < max_nfev_; iter++) {
      Eigen::VectorXd q_loc = q;
      for (int k = 0; k < n_dep; k++) q_loc[idx_opt_[k]] = x[k];

      Eigen::VectorXd r = closure_residual(q_loc);
      int n_r = n_con;

      if (medial_soft_enable_) {
        double medial = x[6];
        double penalty = 0.0;
        if (medial < medial_soft_lower_) { double e = medial - medial_soft_lower_; penalty += e*e; }
        if (medial > medial_soft_upper_) { double e = medial - medial_soft_upper_; penalty += e*e; }
        r.conservativeResize(n_con + 1);
        r[n_con] = std::sqrt(medial_soft_weight_ * penalty);
        n_r = n_con + 1;
      }

      double rnorm = r.norm();
      if (rnorm < closure_tol_) {
        info.success = true; info.nfev = iter; info.closure_norm = rnorm; info.x = x;
        double epsb = 1e-6;
        for (int k = 0; k < n_dep && !info.hit_bounds; k++)
          if (std::abs(x[k]-lower_[k]) < epsb || std::abs(x[k]-upper_[k]) < epsb)
            info.hit_bounds = true;
        if (update_warmstart) last_x_ = x;
        return info;
      }

      pin::computeJointJacobians(model_, data_, q_loc);
      Eigen::MatrixXd J(n_r, n_dep);
      J.setZero();
      for (int i = 0; i < static_cast<int>(closure_frame_pairs_.size()); i++) {
        auto [ida, idb] = closure_frame_pairs_[i];
        Eigen::MatrixXd JA = pin::getFrameJacobian(model_, data_, ida, pin::LOCAL_WORLD_ALIGNED).topRows(3);
        Eigen::MatrixXd JB = pin::getFrameJacobian(model_, data_, idb, pin::LOCAL_WORLD_ALIGNED).topRows(3);
        Eigen::MatrixXd dJi = JA - JB;
        for (int k = 0; k < n_dep; k++) J.block(3*i, k, 3, 1) = dJi.col(idx_opt_[k]);
      }

      std::vector<int> free;
      free.reserve(n_dep);
      for (int k = 0; k < n_dep; k++) {
        bool at_lo = x[k] <= lower_[k] + eps_b;
        bool at_hi = x[k] >= upper_[k] - eps_b;
        if (!at_lo && !at_hi) { free.push_back(k); }
        else {
          double grad_k = J.col(k).head(n_con).dot(r.head(n_con));
          if (at_lo && grad_k < 0.0) free.push_back(k);
          else if (at_hi && grad_k > 0.0) free.push_back(k);
        }
      }

      Eigen::VectorXd dx_full = Eigen::VectorXd::Zero(n_dep);
      if (!free.empty()) {
        int n_free = static_cast<int>(free.size());
        Eigen::MatrixXd J_free(n_r, n_free);
        for (int fi = 0; fi < n_free; fi++) J_free.col(fi) = J.col(free[fi]);
        Eigen::VectorXd dx_free = J_free.colPivHouseholderQr().solve(r);
        for (int fi = 0; fi < n_free; fi++) dx_full[free[fi]] = dx_free[fi];
      }

      x = (x - dx_full).cwiseMax(lower_).cwiseMin(upper_);
      info.nfev = iter + 1;
    }

    Eigen::VectorXd q_loc = q;
    for (int k = 0; k < n_dep; k++) q_loc[idx_opt_[k]] = x[k];
    info.closure_norm = closure_residual(q_loc).norm();
    return info;
  }

  // ── compute_B ──
  Eigen::VectorXd compute_B(const Eigen::VectorXd & q) {
    if (closure_frame_pairs_.empty()) {
      Eigen::VectorXd B = Eigen::VectorXd::Zero(nv_); B[idx_theta_] = 1.0; return B;
    }
    int n_pairs = static_cast<int>(closure_frame_pairs_.size());
    Eigen::MatrixXd A(3*n_pairs, nv_);
    pin::computeJointJacobians(model_, data_, q);
    for (int i = 0; i < n_pairs; i++) {
      auto [ida, idb] = closure_frame_pairs_[i];
      Eigen::MatrixXd JA = pin::getFrameJacobian(model_, data_, ida, pin::LOCAL_WORLD_ALIGNED).topRows(3);
      Eigen::MatrixXd JB = pin::getFrameJacobian(model_, data_, idb, pin::LOCAL_WORLD_ALIGNED).topRows(3);
      A.middleRows(3*i, 3) = JA - JB;
    }
    Eigen::VectorXd e_theta = Eigen::VectorXd::Zero(nv_); e_theta[idx_theta_] = 1.0;
    Eigen::VectorXd rhs = -A * e_theta;
    Eigen::MatrixXd A_red(3*n_pairs, nv_-1);
    int col = 0;
    for (int j = 0; j < nv_; j++) if (j != idx_theta_) A_red.col(col++) = A.col(j);
    Eigen::VectorXd x_red = A_red.colPivHouseholderQr().solve(rhs);
    Eigen::VectorXd B = e_theta;
    col = 0;
    for (int j = 0; j < nv_; j++) if (j != idx_theta_) B[j] += x_red[col++];
    return B;
  }

  // ── compute_tau_ext ──
  void compute_tau_ext(const Eigen::VectorXd & q) {
    tau_ext_.setZero(); tau_ext_theta_ = 0.0;
    if (!wrench_enable_ || fid_ext_ < 0) return;
    pin::forwardKinematics(model_, data_, q);
    pin::updateFramePlacements(model_, data_);
    pin::computeJointJacobians(model_, data_, q);
    Eigen::MatrixXd J6 = pin::getFrameJacobian(model_, data_, fid_ext_, pin::LOCAL_WORLD_ALIGNED);
    pin::Force W(wrench_world_.head<3>(), wrench_world_.tail<3>());
    tau_ext_ = J6.transpose() * W.toVector();
    tau_ext_theta_ = B_.dot(tau_ext_);
  }

  // ── compute_passive_torques ──
  void compute_passive_torques(const Eigen::VectorXd & q, const Eigen::VectorXd & dq) {
    tau_pass_.setZero(); tau_pass_theta_ = 0.0;
    if (!passive_enable_) return;
    auto safe_K = [&](double K0, double alpha, double err) -> double {
      double x = alpha * std::abs(err);
      return x > passive_exp_clip_ ? passive_K_MAX_ : std::min(K0 * std::exp(x), passive_K_MAX_);
    };
    int idx_MCF = jid_MCF_ - 1, idx_IFP = jid_IFP_ - 1;
    double err_MCF = q[idx_MCF] - rest_MCF_;
    tau_pass_[idx_MCF] = -safe_K(K0_MCF_, alpha_MCF_, err_MCF) * err_MCF - B_MCF_ * dq[idx_MCF];
    double err_IFP = q[idx_IFP] - rest_IFP_;
    tau_pass_[idx_IFP] = -safe_K(K0_IFP_, alpha_IFP_, err_IFP) * err_IFP - B_IFP_ * dq[idx_IFP];
    tau_pass_theta_ = B_.dot(tau_pass_);
  }

  // ── stuck_try_release ──
  bool stuck_try_release(double tau_m) {
    if (!at_limit_ || !have_valid_) return false;
    double tau_fric = fric_visc_ * theta_dot_ + fric_coul_ * std::tanh(theta_dot_ / std::max(fric_eps_, 1e-9));
    double tau_eff = tau_m + tau_ext_theta_ + tau_pass_theta_ - tau_fric - gproj_last_;
    if (tau_eff * limit_dir_ >= -limit_release_tau_) return false;
    for (int k = 1; k <= limit_backoff_tries_; k++) {
      double th_try = clamp_theta(theta_valid_ - limit_dir_ * (limit_backoff_step_ * k));
      q_ = q_valid_; last_x_ = last_x_valid_;
      auto res = solve_closure(th_try, false);
      if (!res.success) continue;
      theta_ = th_try; q_[idx_theta_] = th_try;
      for (int i = 0; i < static_cast<int>(idx_opt_.size()); i++) q_[idx_opt_[i]] = res.x[i];
      B_ = compute_B(q_);
      theta_dot_ = 0.0; theta_ddot_ = 0.0; dq_.setZero(); ddq_.setZero();
      at_limit_ = false; store_valid_state();
      return true;
    }
    return false;
  }

  // ── step — returns per-section timing ──
  StepRecord step(double tau_m)
  {
    StepRecord rec{};
    rec.at_limit = 0.0;

    int64_t t0 = now_ns();

    if (at_limit_) {
      if (!stuck_try_release(tau_m)) {
        dq_.setZero(); ddq_.setZero();
        rec.total_ns = now_ns() - t0;
        rec.at_limit = 1.0;
        rec.closure_norm = std::isfinite(last_closure_norm_) ? last_closure_norm_ : -1.0;
        return rec;
      }
    }

    theta_ = clamp_theta(theta_);

    // 1) Closure
    int64_t tc0 = now_ns();
    auto res = solve_closure(theta_, true);
    rec.closure_ns = now_ns() - tc0;

    last_solver_success_ = res.success;
    last_solver_nfev_    = res.nfev;
    last_closure_norm_   = res.closure_norm;
    last_hit_bounds_     = res.hit_bounds;
    rec.nfev        = res.nfev;
    rec.closure_norm= res.closure_norm;

    if (!res.success) {
      at_limit_ = true;
      limit_dir_ = (std::abs(theta_dot_) > 1e-9) ? (theta_dot_ > 0.0 ? 1.0 : -1.0) : 1.0;
      theta_ = theta_valid_; q_ = q_valid_; B_ = B_valid_; last_x_ = last_x_valid_;
      theta_dot_ = 0.0; theta_ddot_ = 0.0; dq_.setZero(); ddq_.setZero();
      rec.total_ns = now_ns() - t0; rec.at_limit = 1.0;
      return rec;
    }

    // 2) Update q, compute B
    q_[idx_theta_] = theta_;
    for (int k = 0; k < static_cast<int>(idx_opt_.size()); k++) q_[idx_opt_[k]] = res.x[k];
    B_prev_ = B_;
    int64_t tb0 = now_ns();
    B_ = compute_B(q_);
    rec.compute_B_ns = now_ns() - tb0;
    at_limit_ = false; store_valid_state();

    Eigen::VectorXd Bdot = (B_ - B_prev_) / dt_;
    dq_  = B_ * theta_dot_;
    ddq_ = B_ * theta_ddot_ + Bdot * theta_dot_;

    // 3) CRBA + nonLinearEffects
    int64_t tM0 = now_ns();
    Eigen::MatrixXd M = pin::crba(model_, data_, q_);
    Eigen::VectorXd h = pin::nonLinearEffects(model_, data_, q_, dq_);
    rec.crba_ns = now_ns() - tM0;

    // 4) External wrench + passive
    int64_t tep0 = now_ns();
    compute_tau_ext(q_);
    compute_passive_torques(q_, dq_);
    rec.ext_passive_ns = now_ns() - tep0;

    // 5) Reduced dynamics
    double denom_mech = B_.dot(M * B_);
    double denom = denom_mech + motor_inertia_;
    denom_last_ = denom;
    gproj_last_ = B_.dot(h);
    proj_last_  = B_.dot(M * (Bdot * theta_dot_) + h);

    if (std::abs(denom) < denom_min_) {
      at_limit_ = true;
      limit_dir_ = (std::abs(theta_dot_) > 1e-6) ? (theta_dot_ > 0.0 ? 1.0 : -1.0) : 1.0;
      theta_ = theta_valid_; q_ = q_valid_; B_ = B_valid_; last_x_ = last_x_valid_;
      theta_dot_ = 0.0; theta_ddot_ = 0.0; dq_.setZero(); ddq_.setZero();
      rec.total_ns = now_ns() - t0; rec.at_limit = 1.0;
      return rec;
    }

    double tau_fric = fric_visc_ * theta_dot_ + fric_coul_ * std::tanh(theta_dot_ / std::max(fric_eps_, 1e-9));
    double tau_damp = damping_theta_ * theta_dot_;
    double num = (tau_m + tau_pass_theta_ + tau_ext_theta_) - proj_last_ - tau_fric - tau_damp;
    theta_ddot_ = std::clamp(num / denom, -max_theta_ddot_, max_theta_ddot_);

    theta_dot_ = std::clamp(theta_dot_ + theta_ddot_ * dt_, -max_theta_dot_, max_theta_dot_);
    double theta_next = clamp_theta(theta_ + theta_dot_ * dt_);

    if (limit_hold_) {
      if (std::abs(theta_next - theta_) < 1e-12 && std::abs(theta_dot_) > 1e-8) {
        at_limit_ = true;
        limit_dir_ = (std::abs(theta_dot_) > 1e-9) ? (theta_dot_ > 0.0 ? 1.0 : -1.0) : 1.0;
        theta_dot_ = 0.0; theta_ddot_ = 0.0; theta_ = theta_next; store_valid_state();
        rec.total_ns = now_ns() - t0; rec.at_limit = 1.0;
        rec.theta = theta_; rec.theta_dot = theta_dot_;
        return rec;
      }
    }
    theta_ = theta_next;

    dq_  = B_ * theta_dot_;
    ddq_ = B_ * theta_ddot_ + Bdot * theta_dot_;

    // 6) RNEA
    int64_t tr0 = now_ns();
    Eigen::VectorXd tau_rnea = pin::rnea(model_, data_, q_, dq_, ddq_);
    rec.rnea_ns = now_ns() - tr0;
    tau_full_ = tau_rnea - tau_ext_ - tau_pass_;

    rec.total_ns = now_ns() - t0;
    rec.theta    = theta_;
    rec.theta_dot= theta_dot_;
    for (int k = 0; k < N_PUB_JOINTS; k++) rec.q_joints[k] = q_[pub_joint_idxs_[k]];
    return rec;
  }

  // ── Ramp step: prescribes theta directly (no integration) ──────────
  // meas_step: step index within the measured phase (0 … n_measured-1)
  // n_half:    steps per half-sweep (from theta_min to theta_max or back)
  // Pattern:   theta_min → theta_max (n_half steps) → theta_min (n_half steps) → repeat
  StepRecord step_ramp(int meas_step, int n_half)
  {
    StepRecord rec{};
    int64_t t0 = now_ns();

    double ramp_v = (theta_max_ - theta_min_) / (n_half * dt_);
    int half_idx = meas_step % (2 * n_half);
    if (half_idx < n_half) {
      theta_     = theta_min_ + ramp_v * half_idx * dt_;
      theta_dot_ = ramp_v;
    } else {
      theta_     = theta_max_ - ramp_v * (half_idx - n_half) * dt_;
      theta_dot_ = -ramp_v;
    }
    theta_ddot_ = 0.0;

    // 1) Closure
    int64_t tc0 = now_ns();
    auto res = solve_closure(theta_, true);
    rec.closure_ns   = now_ns() - tc0;
    rec.nfev         = res.nfev;
    rec.closure_norm = res.closure_norm;

    if (!res.success) {
      rec.total_ns = now_ns() - t0; rec.at_limit = 1.0;
      rec.theta = theta_; rec.theta_dot = theta_dot_;
      return rec;
    }

    // 2) Update q, compute B
    q_[idx_theta_] = theta_;
    for (int k = 0; k < static_cast<int>(idx_opt_.size()); k++) q_[idx_opt_[k]] = res.x[k];
    B_prev_ = B_;
    int64_t tb0 = now_ns();
    B_ = compute_B(q_);
    rec.compute_B_ns = now_ns() - tb0;

    Eigen::VectorXd Bdot = (B_ - B_prev_) / dt_;
    dq_  = B_ * theta_dot_;
    ddq_ = Bdot * theta_dot_;  // theta_ddot_=0

    // 3) CRBA + nonLinearEffects
    int64_t tM0 = now_ns();
    Eigen::MatrixXd M = pin::crba(model_, data_, q_);
    Eigen::VectorXd h = pin::nonLinearEffects(model_, data_, q_, dq_);
    rec.crba_ns = now_ns() - tM0;

    // 4) External wrench + passive
    int64_t tep0 = now_ns();
    compute_tau_ext(q_);
    compute_passive_torques(q_, dq_);
    rec.ext_passive_ns = now_ns() - tep0;

    // 5) RNEA
    int64_t tr0 = now_ns();
    Eigen::VectorXd tau_rnea = pin::rnea(model_, data_, q_, dq_, ddq_);
    rec.rnea_ns = now_ns() - tr0;
    tau_full_ = tau_rnea - tau_ext_ - tau_pass_;

    rec.total_ns  = now_ns() - t0;
    rec.theta     = theta_;
    rec.theta_dot = theta_dot_;
    for (int k = 0; k < N_PUB_JOINTS; k++) rec.q_joints[k] = q_[pub_joint_idxs_[k]];
    return rec;
  }
};

// ============================================================
// Statistics helper
// ============================================================
struct Stats { double mean, std_, min, median, p95, p99, max; };
static Stats compute_stats(const std::vector<int64_t> & v)
{
  std::vector<double> d(v.size());
  for (size_t i = 0; i < v.size(); i++) d[i] = static_cast<double>(v[i]);
  double sum = 0; for (auto x : d) sum += x;
  double mean = sum / d.size();
  double var = 0; for (auto x : d) var += (x-mean)*(x-mean);
  Stats s;
  s.mean = mean; s.std_ = std::sqrt(var / d.size());
  std::vector<double> sorted = d; std::sort(sorted.begin(), sorted.end());
  s.min    = sorted.front();
  s.max    = sorted.back();
  s.median = sorted[sorted.size()/2];
  s.p95    = sorted[static_cast<size_t>(sorted.size() * 0.95)];
  s.p99    = sorted[static_cast<size_t>(sorted.size() * 0.99)];
  return s;
}

// ============================================================
// main
// ============================================================
int main(int argc, char * argv[])
{
  // -- Defaults --
  std::string urdf_path = "";
  int n_steps  = 50000;
  int n_warmup = 2000;
  std::string out_csv = "";  // filled below if empty

  // -- Command-line overrides --
  if (argc > 1) urdf_path = argv[1];
  if (argc > 2) n_steps   = std::stoi(argv[2]);
  if (argc > 3) n_warmup  = std::stoi(argv[3]);
  if (argc > 4) out_csv   = argv[4];

  // Resolve default URDF: relative to this binary's location
  if (urdf_path.empty()) {
    // Try next to executable, then fallback
    std::vector<std::string> candidates = {
      "/home/mbari/ros2_ws/src/HES_DT_FDI/paper_benchmarks/urdf/assembly_with_hand.urdf",
      "/home/mbari/ros2_ws/src/HES_DT_FDI/exoskeleton_description/urdf/assembly_with_hand.urdf",
    };
    for (auto & c : candidates) {
      if (std::filesystem::exists(c)) { urdf_path = c; break; }
    }
  }
  if (urdf_path.empty()) {
    std::cerr << "ERROR: URDF not found. Pass path as first argument.\n";
    return 1;
  }
  if (!std::filesystem::exists(urdf_path)) {
    std::cerr << "ERROR: URDF not found: " << urdf_path << "\n";
    return 1;
  }

  // Resolve default output CSV
  if (out_csv.empty()) {
    out_csv = "/tmp/benchmark_standalone_cpp.csv";
  }
  std::filesystem::create_directories(std::filesystem::path(out_csv).parent_path());

  std::cout << "======================================================\n";
  std::cout << "HES Dynamics C++ — Standalone Computational Benchmark\n";
  std::cout << "======================================================\n";
  std::cout << "  URDF:    " << urdf_path << "\n";
  std::cout << "  Steps:   " << n_steps << "  (warmup: " << n_warmup << ")\n";
  std::cout << "  Output:  " << out_csv << "\n";
  std::cout << "------------------------------------------------------\n";

  // -- Load model --
  auto t_load0 = std::chrono::steady_clock::now();
  ExoBenchmarkDynamics bm(urdf_path);
  double t_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
  std::cout << "  Model: " << bm.model_.name << "  nq=" << bm.nq_ << " nv=" << bm.nv_ << "\n";
  std::cout << "  Closure pairs: " << bm.closure_frame_pairs_.size() << "\n";
  std::cout << "  Load time: " << t_load << " s\n\n";

  // Ramp profile: prescribes theta as triangular wave over the full ROM
  // 2 full cycles (theta_min→max→min) in n_steps measured steps
  // n_half = n_steps/4  →  each quarter-sweep has n_steps/4 steps
  int n_half = n_steps / 4;
  std::cout << "Profile: RAMP  theta=[" << bm.theta_min_ << ", " << bm.theta_max_ << "] rad"
            << "  n_half=" << n_half << "  ramp_v="
            << (bm.theta_max_ - bm.theta_min_) / (n_half * bm.dt_) << " rad/s\n";

  int total_steps = n_steps + n_warmup;
  std::cout << "Running " << total_steps << " steps...\n";

  std::vector<StepRecord> records;
  records.reserve(n_steps);

  auto t_run0 = std::chrono::steady_clock::now();
  for (int i = 0; i < total_steps; i++) {
    int meas_step = i - n_warmup;  // negative during warmup, 0-based after
    StepRecord r = bm.step_ramp(meas_step < 0 ? 0 : meas_step, n_half);
    r.step = meas_step;
    if (i >= n_warmup) records.push_back(r);

    if ((i + 1) % 10000 == 0) {
      double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run0).count();
      std::cout << "  step " << (i+1) << "/" << total_steps
                << "  elapsed=" << elapsed << "s"
                << "  theta=" << bm.theta_ << "\n";
    }
  }
  double t_run = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run0).count();

  double sim_time = n_steps * bm.dt_;
  std::cout << "\n  Wall clock:  " << t_run << " s\n";
  std::cout << "  Simulated:   " << sim_time << " s\n";
  std::cout << "  RT factor:   " << t_run / sim_time << "x\n";
  std::cout << "  Step rate:   " << n_steps / t_run << " step/s\n\n";

  // -- Statistics --
  std::vector<int64_t> total_v, closure_v, B_v, crba_v, ep_v, rnea_v;
  for (auto & r : records) {
    total_v.push_back(r.total_ns);
    closure_v.push_back(r.closure_ns);
    B_v.push_back(r.compute_B_ns);
    crba_v.push_back(r.crba_ns);
    ep_v.push_back(r.ext_passive_ns);
    rnea_v.push_back(r.rnea_ns);
  }

  auto print_stats = [](const char * name, const Stats & s) {
    printf("  %-20s  mean=%7.3f  std=%7.3f  median=%7.3f  p95=%7.3f  p99=%7.3f  max=%7.3f ms\n",
      name, s.mean/1e6, s.std_/1e6, s.median/1e6, s.p95/1e6, s.p99/1e6, s.max/1e6);
  };

  std::cout << "--- Section Timing ---\n";
  auto st = compute_stats(total_v);
  print_stats("total", st);
  print_stats("closure",   compute_stats(closure_v));
  print_stats("compute_B", compute_stats(B_v));
  print_stats("crba",      compute_stats(crba_v));
  print_stats("ext_passive", compute_stats(ep_v));
  print_stats("rnea",      compute_stats(rnea_v));

  // Section breakdown
  auto cl_s = compute_stats(closure_v);
  auto b_s  = compute_stats(B_v);
  auto cr_s = compute_stats(crba_v);
  auto ep_s = compute_stats(ep_v);
  auto rn_s = compute_stats(rnea_v);
  double tot_mean = st.mean;
  double other = tot_mean - cl_s.mean - b_s.mean - cr_s.mean - ep_s.mean - rn_s.mean;
  std::cout << "\n--- Section Breakdown ---\n";
  printf("  %-20s  %6.3f ms  (%5.1f%%)\n", "closure",    cl_s.mean/1e6, 100*cl_s.mean/tot_mean);
  printf("  %-20s  %6.3f ms  (%5.1f%%)\n", "compute_B",  b_s.mean/1e6,  100*b_s.mean/tot_mean);
  printf("  %-20s  %6.3f ms  (%5.1f%%)\n", "crba",       cr_s.mean/1e6, 100*cr_s.mean/tot_mean);
  printf("  %-20s  %6.3f ms  (%5.1f%%)\n", "ext_passive",ep_s.mean/1e6, 100*ep_s.mean/tot_mean);
  printf("  %-20s  %6.3f ms  (%5.1f%%)\n", "rnea",       rn_s.mean/1e6, 100*rn_s.mean/tot_mean);
  printf("  %-20s  %6.3f ms  (%5.1f%%)\n", "other",      other/1e6,     100*other/tot_mean);

  // -- Solver statistics --
  double norm_sum = 0; double norm_max = 0; double nfev_sum = 0;
  int at_limit_count = 0;
  for (auto & r : records) {
    double cn = std::isfinite(r.closure_norm) ? r.closure_norm : 0.0;
    norm_sum += cn; norm_max = std::max(norm_max, cn);
    nfev_sum += r.nfev;
    if (r.at_limit > 0.5) at_limit_count++;
  }
  std::cout << "\n--- Solver Statistics ---\n";
  printf("  closure_norm: mean=%.3e  max=%.3e\n", norm_sum/records.size(), norm_max);
  printf("  nfev:         mean=%.1f\n", nfev_sum/records.size());
  printf("  at_limit:     %d / %zu\n", at_limit_count, records.size());

  // -- Write CSV --
  std::ofstream csv(out_csv);
  csv << "step,total_ns,closure_ns,compute_B_ns,crba_ns,ext_passive_ns,rnea_ns,"
         "closure_norm,nfev,at_limit,theta,theta_dot";
  for (int k = 0; k < N_PUB_JOINTS; k++) csv << "," << PUB_JOINT_NAMES[k];
  csv << "\n";
  for (auto & r : records) {
    csv << r.step << "," << r.total_ns << "," << r.closure_ns << ","
        << r.compute_B_ns << "," << r.crba_ns << "," << r.ext_passive_ns << ","
        << r.rnea_ns << "," << r.closure_norm << "," << r.nfev << ","
        << r.at_limit << "," << r.theta << "," << r.theta_dot;
    for (int k = 0; k < N_PUB_JOINTS; k++) csv << "," << r.q_joints[k];
    csv << "\n";
  }
  csv.close();
  std::cout << "\nSaved: " << out_csv << "\n";
  std::cout << "======================================================\n";
  printf("  Mean: %.3f ms  P95: %.3f ms  RT: %.2fx\n",
    st.mean/1e6, st.p95/1e6, t_run/sim_time);
  std::cout << "======================================================\n";
  return 0;
}
