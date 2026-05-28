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

#include <Eigen/Dense>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <exoskeleton_safety_msgs/msg/float64_array_stamped.hpp>
#include <exoskeleton_safety_msgs/msg/float64_stamped.hpp>

#include <sched.h>
#include <sys/mman.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace pin = pinocchio;
using Float64Stamped = exoskeleton_safety_msgs::msg::Float64Stamped;
using Float64ArrayStamped = exoskeleton_safety_msgs::msg::Float64ArrayStamped;

class ExoReducedDynamicsWithHand : public rclcpp::Node
{
public:
  ExoReducedDynamicsWithHand();

private:
  // ── Pinocchio model ──
  pin::Model model_;
  pin::Data data_;
  int nq_, nv_;
  int jid_crank_, idx_theta_;
  int jid_MCF_, jid_IFP_;

  // ── Closure frame pairs ──
  std::vector<std::pair<int, int>> closure_frame_pairs_;
  std::vector<int> idx_opt_;
  Eigen::VectorXd lower_, upper_;

  // ── Cached parameters ──
  double dt_, publish_dt_;
  double motor_inertia_, fric_visc_, fric_coul_, fric_eps_, damping_theta_;
  double max_theta_dot_, max_theta_ddot_;
  double closure_tol_;
  int max_nfev_;
  double denom_min_;
  int log_every_n_steps_;
  double theta_min_, theta_max_;
  bool limit_hold_, limit_use_theta_bounds_;
  double limit_release_tau_, limit_backoff_step_;
  int limit_backoff_tries_;
  bool wrench_enable_;
  std::string wrench_frame_name_;
  double wrench_scale_;
  int fid_ext_;
  bool passive_enable_;
  double K0_MCF_, alpha_MCF_, B_MCF_, rest_MCF_;
  double K0_IFP_, alpha_IFP_, B_IFP_, rest_IFP_;
  double passive_K_MAX_, passive_exp_clip_;
  bool medial_soft_enable_;
  double medial_soft_lower_, medial_soft_upper_, medial_soft_weight_;
  bool ce_force_enable_;
  std::string ce_force_frame_;
  double ce_force_marker_scale_, ce_force_marker_diameter_;
  int fid_CE_;
  bool rt_enable_;
  int rt_priority_;

  // ── State ──
  double theta_, theta_dot_, theta_ddot_;
  Eigen::VectorXd q_, dq_, ddq_;
  Eigen::VectorXd last_x_;
  Eigen::VectorXd B_, B_prev_;
  double tau_m_;
  Eigen::VectorXd wrench_world_;
  Eigen::VectorXd tau_ext_;
  double tau_ext_theta_;
  Eigen::VectorXd tau_pass_;
  double tau_pass_theta_;
  Eigen::VectorXd tau_full_;

  // ── Diagnostics ──
  int step_count_;
  bool last_solver_success_;
  int last_solver_nfev_;
  double last_closure_norm_;
  bool last_hit_bounds_;
  double denom_last_, proj_last_, gproj_last_, reaction_theta_last_;
  double tau_fric_last_, tau_damp_last_, num_last_;
  double dyn_residual_last_, tau_model_last_, tau_model_error_last_;

  // ── Limit state ──
  bool at_limit_;
  double limit_dir_;
  bool have_valid_;
  double theta_valid_;
  Eigen::VectorXd q_valid_, dq_valid_, B_valid_, last_x_valid_;

  // ── Cached publish joint indices ──
  std::vector<int> pub_joint_idxs_;
  std::vector<std::string> pub_joint_names_;

  // ── ROS I/O ──
  rclcpp::Subscription<Float64Stamped>::SharedPtr sub_tau_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr sub_wrench_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_js_;
  rclcpp::Publisher<Float64ArrayStamped>::SharedPtr pub_dbg_, pub_ff_terms_, pub_model_dbg_;
  rclcpp::Publisher<Float64Stamped>::SharedPtr pub_tau_ext_theta_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr pub_ce_force_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_ce_marker_;
  rclcpp::TimerBase::SharedPtr step_timer_, pub_timer_;

  // ── Methods ──
  void torque_cb(const Float64Stamped::SharedPtr msg);
  void wrench_cb(const geometry_msgs::msg::WrenchStamped::SharedPtr msg);
  void store_valid_state();
  bool theta_within_bounds(double theta) const;
  double clamp_theta_bounds(double theta) const;

  struct SolverResult {
    bool success;
    int nfev;
    double closure_norm;
    bool hit_bounds;
    Eigen::VectorXd x;
  };
  Eigen::VectorXd closure_residual(const Eigen::VectorXd & q_loc);
  SolverResult solve_closure(double theta, bool update_warmstart);
  Eigen::VectorXd compute_B(const Eigen::VectorXd & q);
  void compute_tau_ext(const Eigen::VectorXd & q);
  void compute_passive_torques(const Eigen::VectorXd & q, const Eigen::VectorXd & dq);
  bool stuck_try_release();
  void step();
  void publish();
  void publish_ce_force();
};

ExoReducedDynamicsWithHand::ExoReducedDynamicsWithHand()
: Node("exo_reduced_dynamics_with_hand"),
  tau_m_(0.0), tau_ext_theta_(0.0), tau_pass_theta_(0.0),
  step_count_(0), last_solver_success_(false), last_solver_nfev_(0),
  last_closure_norm_(std::numeric_limits<double>::quiet_NaN()),
  last_hit_bounds_(false), denom_last_(0.0), proj_last_(0.0),
  gproj_last_(0.0), reaction_theta_last_(0.0),
  tau_fric_last_(0.0), tau_damp_last_(0.0), num_last_(0.0),
  dyn_residual_last_(0.0), tau_model_last_(0.0), tau_model_error_last_(0.0),
  at_limit_(false), limit_dir_(0.0), have_valid_(false), theta_valid_(0.0)
{
  // ── URDF path ──
  std::string default_urdf;
  try {
    default_urdf = ament_index_cpp::get_package_share_directory("exoskeleton_description")
      + "/urdf/assembly_with_hand.urdf";
  } catch (...) {}

  // ── Declare parameters (same names as Python node) ──
  declare_parameter("urdf_path", default_urdf);
  declare_parameter("dt", 0.001);
  declare_parameter("publish_dt", 0.005);
  declare_parameter("theta_init", 0.0);
  declare_parameter("gravity_zero", false);
  declare_parameter("motor_inertia", 0.1);
  declare_parameter("fric_visc", 0.3);
  declare_parameter("fric_coul", 0.3);
  declare_parameter("fric_eps", 0.01);
  declare_parameter("damping_theta", 0.0);
  declare_parameter("max_theta_dot", 10.0);
  declare_parameter("max_theta_ddot", 400.0);
  declare_parameter("closure_tol", 1e-5);
  declare_parameter("max_nfev", 150);
  declare_parameter("denom_min", 1e-7);
  declare_parameter("log_every_n_steps", 200);
  declare_parameter("theta_min", -2.5);
  declare_parameter("theta_max", 2.5);
  declare_parameter("limit_hold", true);
  declare_parameter("limit_release_tau", 0.02);
  declare_parameter("limit_backoff_step", 0.003);
  declare_parameter("limit_backoff_tries", 6);
  declare_parameter("limit_use_theta_bounds", true);
  declare_parameter("external_wrench_enable", true);
  declare_parameter("external_wrench_frame", std::string("frame_CE_end_2"));
  declare_parameter("external_wrench_topic", std::string("/exo_dynamics/external_wrench"));
  declare_parameter("external_wrench_scale", 1.0);
  declare_parameter("passive_enable", true);
  declare_parameter("K0_MCF", 0.03);
  declare_parameter("alpha_MCF", 4.0);
  declare_parameter("B_MCF", 0.01);
  declare_parameter("rest_MCF", -0.3);
  declare_parameter("K0_IFP", 0.03);
  declare_parameter("alpha_IFP", 4.0);
  declare_parameter("B_IFP", 0.01);
  declare_parameter("rest_IFP", -0.5);
  declare_parameter("passive_K_MAX", 5.0);
  declare_parameter("passive_exp_clip", 12.0);
  declare_parameter("medial_soft_enable", true);
  declare_parameter("medial_soft_lower", -2.2);
  declare_parameter("medial_soft_upper", 0.0);
  declare_parameter("medial_soft_weight", 1.0);
  declare_parameter("ce_force_enable", false);
  declare_parameter("ce_force_frame", std::string("frame_CE_end_2"));
  declare_parameter("ce_force_marker_scale", 0.02);
  declare_parameter("ce_force_marker_diameter", 0.01);
  declare_parameter("rt_enable", false);
  declare_parameter("rt_priority", 80);

  // ── Cache all parameters ──
  auto urdf_path = get_parameter("urdf_path").as_string();
  dt_           = get_parameter("dt").as_double();
  publish_dt_   = get_parameter("publish_dt").as_double();
  motor_inertia_ = get_parameter("motor_inertia").as_double();
  fric_visc_    = get_parameter("fric_visc").as_double();
  fric_coul_    = get_parameter("fric_coul").as_double();
  fric_eps_     = get_parameter("fric_eps").as_double();
  damping_theta_ = get_parameter("damping_theta").as_double();
  max_theta_dot_ = get_parameter("max_theta_dot").as_double();
  max_theta_ddot_ = get_parameter("max_theta_ddot").as_double();
  closure_tol_  = get_parameter("closure_tol").as_double();
  max_nfev_     = get_parameter("max_nfev").as_int();
  denom_min_    = get_parameter("denom_min").as_double();
  log_every_n_steps_ = get_parameter("log_every_n_steps").as_int();
  theta_min_    = get_parameter("theta_min").as_double();
  theta_max_    = get_parameter("theta_max").as_double();
  limit_hold_   = get_parameter("limit_hold").as_bool();
  limit_use_theta_bounds_ = get_parameter("limit_use_theta_bounds").as_bool();
  limit_release_tau_ = get_parameter("limit_release_tau").as_double();
  limit_backoff_step_ = get_parameter("limit_backoff_step").as_double();
  limit_backoff_tries_ = get_parameter("limit_backoff_tries").as_int();
  wrench_enable_ = get_parameter("external_wrench_enable").as_bool();
  wrench_frame_name_ = get_parameter("external_wrench_frame").as_string();
  wrench_scale_  = get_parameter("external_wrench_scale").as_double();
  passive_enable_ = get_parameter("passive_enable").as_bool();
  K0_MCF_   = get_parameter("K0_MCF").as_double();
  alpha_MCF_ = get_parameter("alpha_MCF").as_double();
  B_MCF_    = get_parameter("B_MCF").as_double();
  rest_MCF_ = get_parameter("rest_MCF").as_double();
  K0_IFP_   = get_parameter("K0_IFP").as_double();
  alpha_IFP_ = get_parameter("alpha_IFP").as_double();
  B_IFP_    = get_parameter("B_IFP").as_double();
  rest_IFP_ = get_parameter("rest_IFP").as_double();
  passive_K_MAX_  = get_parameter("passive_K_MAX").as_double();
  passive_exp_clip_ = get_parameter("passive_exp_clip").as_double();
  medial_soft_enable_ = get_parameter("medial_soft_enable").as_bool();
  medial_soft_lower_  = get_parameter("medial_soft_lower").as_double();
  medial_soft_upper_  = get_parameter("medial_soft_upper").as_double();
  medial_soft_weight_ = get_parameter("medial_soft_weight").as_double();
  ce_force_enable_ = get_parameter("ce_force_enable").as_bool();
  ce_force_frame_  = get_parameter("ce_force_frame").as_string();
  ce_force_marker_scale_ = get_parameter("ce_force_marker_scale").as_double();
  ce_force_marker_diameter_ = get_parameter("ce_force_marker_diameter").as_double();
  rt_enable_   = get_parameter("rt_enable").as_bool();
  rt_priority_ = get_parameter("rt_priority").as_int();

  // ── Pinocchio model ──
  try {
    pin::urdf::buildModel(urdf_path, model_);
  } catch (...) {
    pin::urdf::buildModel(urdf_path, pin::JointModelFreeFlyer(), model_);
  }
  data_ = pin::Data(model_);
  nq_ = model_.nq;
  nv_ = model_.nv;

  if (get_parameter("gravity_zero").as_bool()) {
    model_.gravity.linear(Eigen::Vector3d::Zero());
    RCLCPP_WARN(get_logger(), "gravity_zero=true: gravity disabled.");
  }

  // ── Joint indices ──
  auto check_joint = [&](const std::string & name) -> int {
    size_t jid = model_.getJointId(name);
    if (jid >= static_cast<size_t>(model_.njoints))
      throw std::runtime_error("Joint '" + name + "' not found in URDF.");
    return static_cast<int>(jid);
  };

  jid_crank_ = check_joint("rev_crank");
  idx_theta_ = jid_crank_ - 1;

  jid_MCF_ = check_joint("rev_palmo2prossimale");
  jid_IFP_ = check_joint("rev_prossimale2mediale");

  // ── CE frame ──
  fid_CE_ = static_cast<int>(model_.getFrameId(ce_force_frame_));
  if (ce_force_enable_ && fid_CE_ < 0) {
    RCLCPP_WARN(get_logger(), "ce_force_enable=true but frame not found. Disabling.");
    ce_force_enable_ = false;
  }

  // ── Closure frame pairs ──
  std::vector<std::pair<std::string, std::string>> pair_names = {
    {"frame_AC_end",   "frame_rod_end"},
    {"frame_AC_end_2", "frame_BC_end"},
    {"frame_CE_end",   "frame_BC_end_2"},
    {"frame_CE_end_2", "slider_t"},
  };
  for (auto & [a, b] : pair_names) {
    int ida = static_cast<int>(model_.getFrameId(a));
    int idb = static_cast<int>(model_.getFrameId(b));
    if (ida < 0 || idb < 0) {
      RCLCPP_WARN(get_logger(), "Closure frame pair not found: %s, %s", a.c_str(), b.c_str());
    } else {
      closure_frame_pairs_.push_back({ida, idb});
    }
  }

  // ── Dependent DOFs ──
  std::vector<std::string> dep_names = {
    "rev_body2linkAC", "rev_crank2shaft", "slider", "rev_slider2linkBC",
    "rev_linkAC2linkCE", "rev_palmo2prossimale", "rev_prossimale2mediale", "slider2"
  };
  for (auto & n : dep_names)
    idx_opt_.push_back(static_cast<int>(model_.getJointId(n)) - 1);

  lower_ = (Eigen::VectorXd(8) << -2.5,-2.5,-0.015,-2.5,-2.5,-1.5,-2.5,-0.008).finished();
  upper_ = (Eigen::VectorXd(8) <<  2.5, 2.5, 0.004, 2.5, 2.5, 1.2, 2.5, 0.008).finished();

  // ── Cache joint indices for publish() ──
  pub_joint_names_ = {
    "rev_crank", "rev_body2linkAC", "rev_crank2shaft", "slider",
    "rev_slider2linkBC", "rev_linkAC2linkCE",
    "rev_palmo2prossimale", "rev_prossimale2mediale", "slider2"
  };
  for (const auto & n : pub_joint_names_) {
    size_t jid = model_.getJointId(n);
    if (jid >= static_cast<size_t>(model_.njoints))
      throw std::runtime_error("Publish joint '" + n + "' not found in URDF.");
    pub_joint_idxs_.push_back(static_cast<int>(jid) - 1);
  }

  // ── State init ──
  theta_    = get_parameter("theta_init").as_double();
  theta_dot_ = 0.0;
  theta_ddot_ = 0.0;
  q_  = pin::neutral(model_);
  dq_ = Eigen::VectorXd::Zero(nv_);
  ddq_ = Eigen::VectorXd::Zero(nv_);
  last_x_ = Eigen::VectorXd::Zero(static_cast<int>(idx_opt_.size()));
  B_      = Eigen::VectorXd::Zero(nv_);
  B_prev_ = Eigen::VectorXd::Zero(nv_);
  wrench_world_ = Eigen::VectorXd::Zero(6);
  tau_ext_  = Eigen::VectorXd::Zero(nv_);
  tau_pass_ = Eigen::VectorXd::Zero(nv_);
  tau_full_ = Eigen::VectorXd::Zero(nv_);
  q_valid_ = q_; dq_valid_ = dq_; B_valid_ = B_; last_x_valid_ = last_x_;

  // ── External wrench frame ──
  fid_ext_ = -1;
  if (wrench_enable_) {
    fid_ext_ = static_cast<int>(model_.getFrameId(wrench_frame_name_));
    if (fid_ext_ < 0) {
      RCLCPP_WARN(get_logger(), "external_wrench_frame not found. Disabling.");
      wrench_enable_ = false;
    }
  }

  // ── Initial closure ──
  auto res0 = solve_closure(theta_, true);
  if (res0.success && static_cast<int>(res0.x.size()) == static_cast<int>(idx_opt_.size())) {
    q_[idx_theta_] = theta_;
    for (int k = 0; k < static_cast<int>(idx_opt_.size()); k++) q_[idx_opt_[k]] = res0.x[k];
    B_ = compute_B(q_);
    store_valid_state();
  } else if (res0.success) {
    B_ = compute_B(q_);
    store_valid_state();
  } else {
    RCLCPP_WARN(get_logger(), "Initial solve_closure failed: starting AT_LIMIT.");
    at_limit_ = true;
  }
  // ── ROS I/O ──
  sub_tau_ = create_subscription<Float64Stamped>(
    "/torque", 10,
    [this](const Float64Stamped::SharedPtr m) { torque_cb(m); });

  if (wrench_enable_) {
    auto wrench_topic = get_parameter("external_wrench_topic").as_string();
    sub_wrench_ = create_subscription<geometry_msgs::msg::WrenchStamped>(
      wrench_topic, 10,
      [this](const geometry_msgs::msg::WrenchStamped::SharedPtr m) { wrench_cb(m); });
  }

  pub_js_          = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
  pub_dbg_         = create_publisher<Float64ArrayStamped>("/exo_dynamics/debug", 10);
  pub_ff_terms_    = create_publisher<Float64ArrayStamped>("/exo_dynamics/ff_terms", 10);
  pub_tau_ext_theta_ = create_publisher<Float64Stamped>("/exo_dynamics/tau_ext_theta", 10);
  pub_model_dbg_   = create_publisher<Float64ArrayStamped>("/exo_dynamics/model_debug", 10);

  if (ce_force_enable_) {
    pub_ce_force_  = create_publisher<geometry_msgs::msg::WrenchStamped>("/ce_force", 10);
    pub_ce_marker_ = create_publisher<visualization_msgs::msg::Marker>("/ce_force_marker", 10);
  }

  // ── RT scheduling ──
  if (rt_enable_) {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
      RCLCPP_WARN(get_logger(), "mlockall failed (need root or CAP_IPC_LOCK).");
    sched_param sp;
    sp.sched_priority = rt_priority_;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
      RCLCPP_WARN(get_logger(), "sched_setscheduler failed (need root or CAP_SYS_NICE).");
    else
      RCLCPP_INFO(get_logger(), "RT scheduling enabled: SCHED_FIFO priority=%d", rt_priority_);
  }

  // ── Timers ──
  step_timer_ = create_wall_timer(
    std::chrono::duration<double>(dt_),
    [this]() { step(); });
  pub_timer_ = create_wall_timer(
    std::chrono::duration<double>(publish_dt_),
    [this]() { publish(); });

  RCLCPP_INFO(get_logger(),
    "ExoReducedDynamicsWithHand started | URDF=%s | passive=%s | wrench=%s | rt=%s",
    urdf_path.c_str(),
    passive_enable_ ? "true" : "false",
    wrench_enable_  ? "true" : "false",
    rt_enable_      ? "true" : "false");
}

void ExoReducedDynamicsWithHand::torque_cb(const Float64Stamped::SharedPtr msg)
{ tau_m_ = msg->data; }

void ExoReducedDynamicsWithHand::wrench_cb(
  const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
{
  wrench_world_ << wrench_scale_ * msg->wrench.force.x,
                   wrench_scale_ * msg->wrench.force.y,
                   wrench_scale_ * msg->wrench.force.z,
                   wrench_scale_ * msg->wrench.torque.x,
                   wrench_scale_ * msg->wrench.torque.y,
                   wrench_scale_ * msg->wrench.torque.z;
}

void ExoReducedDynamicsWithHand::store_valid_state()
{
  have_valid_   = true;
  theta_valid_  = theta_;
  q_valid_      = q_;
  dq_valid_     = dq_;
  B_valid_      = B_;
  last_x_valid_ = last_x_;
}

bool ExoReducedDynamicsWithHand::theta_within_bounds(double theta) const
{
  if (!limit_use_theta_bounds_) return true;
  return theta >= theta_min_ && theta <= theta_max_;
}

double ExoReducedDynamicsWithHand::clamp_theta_bounds(double theta) const
{
  if (!limit_use_theta_bounds_) return theta;
  return std::clamp(theta, theta_min_, theta_max_);
}

// ── Task 3: NR projected closure solver ──
Eigen::VectorXd ExoReducedDynamicsWithHand::closure_residual(const Eigen::VectorXd & q_loc)
{
  pin::forwardKinematics(model_, data_, q_loc);
  pin::updateFramePlacements(model_, data_);

  int n = static_cast<int>(closure_frame_pairs_.size());
  Eigen::VectorXd r(3 * n);
  for (int i = 0; i < n; i++) {
    auto [ida, idb] = closure_frame_pairs_[i];
    r.segment(3 * i, 3) = data_.oMf[ida].translation() - data_.oMf[idb].translation();
  }
  return r;
}

ExoReducedDynamicsWithHand::SolverResult
ExoReducedDynamicsWithHand::solve_closure(double theta, bool update_warmstart)
{
  SolverResult info{false, 0, std::numeric_limits<double>::infinity(), false, {}};

  Eigen::VectorXd q = q_;
  q[idx_theta_] = theta;

  if (closure_frame_pairs_.empty()) {
    Eigen::VectorXd q_loc = q;
    double cnorm = closure_residual(q_loc).norm();
    info.success = true;
    info.nfev = 0;
    info.closure_norm = cnorm;
    info.x = last_x_;
    return info;
  }

  int n_dep = static_cast<int>(idx_opt_.size());
  int n_con = static_cast<int>(closure_frame_pairs_.size()) * 3;
  Eigen::VectorXd x = last_x_;

  constexpr double eps_b = 1e-9;  // active-bound tolerance

  for (int iter = 0; iter < max_nfev_; iter++) {
    // Build q_loc with current dependent DOF values
    Eigen::VectorXd q_loc = q;
    for (int k = 0; k < n_dep; k++) q_loc[idx_opt_[k]] = x[k];

    // Geometric residual
    Eigen::VectorXd r = closure_residual(q_loc);

    // Medial soft penalty (matches Python: idx_medial_in_x = 6)
    int n_r = n_con;
    if (medial_soft_enable_) {
      double medial = x[6];
      double penalty = 0.0;
      if (medial < medial_soft_lower_) {
        double e = medial - medial_soft_lower_;
        penalty += e * e;
      }
      if (medial > medial_soft_upper_) {
        double e = medial - medial_soft_upper_;
        penalty += e * e;
      }
      r.conservativeResize(n_con + 1);
      r[n_con] = std::sqrt(medial_soft_weight_ * penalty);
      n_r = n_con + 1;
    }

    double rnorm = r.norm();

    if (rnorm < closure_tol_) {
      info.success = true;
      info.nfev = iter;
      info.closure_norm = rnorm;
      info.x = x;
      double epsb = 1e-6;
      for (int k = 0; k < n_dep && !info.hit_bounds; k++) {
        if (std::abs(x[k] - lower_[k]) < epsb || std::abs(x[k] - upper_[k]) < epsb)
          info.hit_bounds = true;
      }
      if (update_warmstart) last_x_ = x;
      return info;
    }

    // Build full Jacobian: J is (n_r × n_dep)
    // J[3i:3i+3, k] = (J_A - J_B)[:3, idx_opt_[k]]
    pin::computeJointJacobians(model_, data_, q_loc);
    Eigen::MatrixXd J(n_r, n_dep);
    J.setZero();

    for (int i = 0; i < static_cast<int>(closure_frame_pairs_.size()); i++) {
      auto [ida, idb] = closure_frame_pairs_[i];
      Eigen::MatrixXd JA = pin::getFrameJacobian(
        model_, data_, ida, pin::LOCAL_WORLD_ALIGNED).topRows(3);
      Eigen::MatrixXd JB = pin::getFrameJacobian(
        model_, data_, idb, pin::LOCAL_WORLD_ALIGNED).topRows(3);
      Eigen::MatrixXd dJi = JA - JB;
      for (int k = 0; k < n_dep; k++)
        J.block(3 * i, k, 3, 1) = dJi.col(idx_opt_[k]);
    }
    // Last row (soft penalty Jacobian) stays zero — penalty is small near solution

    // Active-set projected Gauss-Newton step:
    // Variables at a bound whose gradient pushes further into the bound are frozen.
    // This correctly handles rank-deficient systems with active bound constraints.
    std::vector<int> free;
    free.reserve(n_dep);
    for (int k = 0; k < n_dep; k++) {
      bool at_lo = x[k] <= lower_[k] + eps_b;
      bool at_hi = x[k] >= upper_[k] - eps_b;
      if (!at_lo && !at_hi) {
        free.push_back(k);
      } else {
        // grad_k = J[:,k]^T * r  (direction of descent for variable k)
        double grad_k = J.col(k).head(n_con).dot(r.head(n_con));
        if (at_lo && grad_k < 0.0) free.push_back(k);   // gradient pointing away from lower
        else if (at_hi && grad_k > 0.0) free.push_back(k); // gradient pointing away from upper
        // otherwise: frozen at this bound
      }
    }

    Eigen::VectorXd dx_full = Eigen::VectorXd::Zero(n_dep);
    if (!free.empty()) {
      int n_free = static_cast<int>(free.size());
      Eigen::MatrixXd J_free(n_r, n_free);
      for (int fi = 0; fi < n_free; fi++)
        J_free.col(fi) = J.col(free[fi]);
      Eigen::VectorXd dx_free = J_free.colPivHouseholderQr().solve(r);
      for (int fi = 0; fi < n_free; fi++)
        dx_full[free[fi]] = dx_free[fi];
    }

    x = (x - dx_full).cwiseMax(lower_).cwiseMin(upper_);
    info.nfev = iter + 1;
  }

  // Did not converge — compute final closure norm
  Eigen::VectorXd q_loc = q;
  for (int k = 0; k < n_dep; k++) q_loc[idx_opt_[k]] = x[k];
  info.closure_norm = closure_residual(q_loc).norm();
  return info;
}

Eigen::VectorXd ExoReducedDynamicsWithHand::compute_B(const Eigen::VectorXd & q)
{
  if (closure_frame_pairs_.empty()) {
    Eigen::VectorXd B = Eigen::VectorXd::Zero(nv_);
    B[idx_theta_] = 1.0;
    return B;
  }

  int n_pairs = static_cast<int>(closure_frame_pairs_.size());
  Eigen::MatrixXd A(3 * n_pairs, nv_);

  pin::computeJointJacobians(model_, data_, q);
  for (int i = 0; i < n_pairs; i++) {
    auto [ida, idb] = closure_frame_pairs_[i];
    Eigen::MatrixXd JA = pin::getFrameJacobian(
      model_, data_, ida, pin::LOCAL_WORLD_ALIGNED).topRows(3);
    Eigen::MatrixXd JB = pin::getFrameJacobian(
      model_, data_, idb, pin::LOCAL_WORLD_ALIGNED).topRows(3);
    A.middleRows(3 * i, 3) = JA - JB;
  }

  Eigen::VectorXd e_theta = Eigen::VectorXd::Zero(nv_);
  e_theta[idx_theta_] = 1.0;
  Eigen::VectorXd rhs = -A * e_theta;

  // Build A_red: drop column idx_theta_
  Eigen::MatrixXd A_red(3 * n_pairs, nv_ - 1);
  int col = 0;
  for (int j = 0; j < nv_; j++) {
    if (j != idx_theta_) A_red.col(col++) = A.col(j);
  }

  Eigen::VectorXd x_red = A_red.colPivHouseholderQr().solve(rhs);

  Eigen::VectorXd B = e_theta;
  col = 0;
  for (int j = 0; j < nv_; j++) {
    if (j != idx_theta_) B[j] += x_red[col++];
  }
  return B;
}

void ExoReducedDynamicsWithHand::compute_tau_ext(const Eigen::VectorXd & q)
{
  if (!wrench_enable_ || fid_ext_ < 0) {
    tau_ext_.setZero();
    tau_ext_theta_ = 0.0;
    return;
  }

  pin::forwardKinematics(model_, data_, q);
  pin::updateFramePlacements(model_, data_);
  pin::computeJointJacobians(model_, data_, q);

  Eigen::MatrixXd J6 = pin::getFrameJacobian(
    model_, data_, fid_ext_, pin::LOCAL_WORLD_ALIGNED);
  pin::Force W(wrench_world_.head<3>(), wrench_world_.tail<3>());
  tau_ext_ = J6.transpose() * W.toVector();
  tau_ext_theta_ = B_.dot(tau_ext_);
}
void ExoReducedDynamicsWithHand::compute_passive_torques(
  const Eigen::VectorXd & q, const Eigen::VectorXd & dq)
{
  tau_pass_.setZero();
  tau_pass_theta_ = 0.0;
  if (!passive_enable_) return;

  auto safe_exp_stiffness = [&](double K0, double alpha, double error) -> double {
    double x = alpha * std::abs(error);
    if (x > passive_exp_clip_) return passive_K_MAX_;
    return std::min(K0 * std::exp(x), passive_K_MAX_);
  };

  int idx_MCF = jid_MCF_ - 1;
  int idx_IFP = jid_IFP_ - 1;

  double q_MCF  = q[idx_MCF];
  double dq_MCF = dq[idx_MCF];
  double err_MCF = q_MCF - rest_MCF_;
  double K_MCF = safe_exp_stiffness(K0_MCF_, alpha_MCF_, err_MCF);
  double tau_MCF = -K_MCF * err_MCF - B_MCF_ * dq_MCF;

  double q_IFP  = q[idx_IFP];
  double dq_IFP = dq[idx_IFP];
  double err_IFP = q_IFP - rest_IFP_;
  double K_IFP = safe_exp_stiffness(K0_IFP_, alpha_IFP_, err_IFP);
  double tau_IFP = -K_IFP * err_IFP - B_IFP_ * dq_IFP;

  tau_pass_[idx_MCF] = tau_MCF;
  tau_pass_[idx_IFP] = tau_IFP;
  tau_pass_theta_ = B_.dot(tau_pass_);
}

bool ExoReducedDynamicsWithHand::stuck_try_release()
{
  if (!at_limit_ || !have_valid_) return false;

  double tau_fric = fric_visc_ * theta_dot_
    + fric_coul_ * std::tanh(theta_dot_ / std::max(fric_eps_, 1e-9));

  double tau_eff = tau_m_ + tau_ext_theta_ + tau_pass_theta_ - tau_fric - gproj_last_;
  if (tau_eff * limit_dir_ >= -limit_release_tau_) return false;

  for (int k = 1; k <= limit_backoff_tries_; k++) {
    double th_try = clamp_theta_bounds(
      theta_valid_ - limit_dir_ * (limit_backoff_step_ * k));

    q_    = q_valid_;
    last_x_ = last_x_valid_;

    auto res = solve_closure(th_try, false);
    if (!res.success || static_cast<int>(res.x.size()) != static_cast<int>(idx_opt_.size()))
      continue;

    theta_ = th_try;
    q_[idx_theta_] = th_try;
    for (int i = 0; i < static_cast<int>(idx_opt_.size()); i++)
      q_[idx_opt_[i]] = res.x[i];
    B_ = compute_B(q_);
    theta_dot_ = 0.0; theta_ddot_ = 0.0;
    dq_.setZero(); ddq_.setZero();
    at_limit_ = false;
    store_valid_state();
    return true;
  }
  return false;
}
void ExoReducedDynamicsWithHand::step()
{
  step_count_++;
  int logN = std::max(1, log_every_n_steps_);

  // 0) If stuck at limit, attempt release
  if (at_limit_) {
    if (!stuck_try_release()) {
      dq_.setZero();
      ddq_.setZero();
      return;
    }
  }

  // 1) Clamp theta within configured bounds
  theta_ = clamp_theta_bounds(theta_);

  // 2) Kinematic closure
  auto res = solve_closure(theta_, true);
  last_solver_success_ = res.success;
  last_solver_nfev_    = res.nfev;
  last_closure_norm_   = res.closure_norm;
  last_hit_bounds_     = res.hit_bounds;

  if (!res.success) {
    at_limit_ = true;
    limit_dir_ = (std::abs(theta_dot_) > 1e-9) ? (theta_dot_ > 0.0 ? 1.0 : -1.0) : 1.0;
    theta_ = theta_valid_;
    q_ = q_valid_;
    B_ = B_valid_;
    last_x_ = last_x_valid_;
    theta_dot_ = 0.0;
    theta_ddot_ = 0.0;
    dq_.setZero();
    ddq_.setZero();
    if (step_count_ % logN == 0)
      RCLCPP_WARN(get_logger(),
        "[step %d] solve_closure failed -> AT_LIMIT | closure_norm=%.3e nfev=%d",
        step_count_, last_closure_norm_, last_solver_nfev_);
    return;
  }

  // 3) Update q and B
  q_[idx_theta_] = theta_;
  for (int k = 0; k < static_cast<int>(idx_opt_.size()); k++)
    q_[idx_opt_[k]] = res.x[k];
  B_prev_ = B_;
  B_ = compute_B(q_);
  at_limit_ = false;
  store_valid_state();

  Eigen::VectorXd Bdot = (B_ - B_prev_) / dt_;

  // 4) Preliminary dq, ddq
  dq_  = B_ * theta_dot_;
  ddq_ = B_ * theta_ddot_ + Bdot * theta_dot_;

  // 5) Mass matrix and nonlinear effects
  Eigen::MatrixXd M = pin::crba(model_, data_, q_);
  Eigen::VectorXd h = pin::nonLinearEffects(model_, data_, q_, dq_);

  // 6) External wrench and passive torques
  compute_tau_ext(q_);
  compute_passive_torques(q_, dq_);

  // 7) Reduced scalar quantities
  double denom_mech = B_.dot(M * B_);
  double denom = denom_mech + motor_inertia_;
  denom_last_  = denom;
  gproj_last_  = B_.dot(h);
  proj_last_   = B_.dot(M * (Bdot * theta_dot_) + h);

  if (std::abs(denom) < denom_min_) {
    at_limit_ = true;
    limit_dir_ = (std::abs(theta_dot_) > 1e-6) ? (theta_dot_ > 0.0 ? 1.0 : -1.0) : 1.0;
    theta_ = theta_valid_;
    q_ = q_valid_;
    B_ = B_valid_;
    last_x_ = last_x_valid_;
    theta_dot_ = 0.0;
    theta_ddot_ = 0.0;
    dq_.setZero();
    ddq_.setZero();
    if (step_count_ % logN == 0)
      RCLCPP_WARN(get_logger(),
        "[step %d] effective inertia too small -> AT_LIMIT | denom=%.3e", step_count_, denom);
    return;
  }

  // 8) Friction and damping
  double tau_fric = fric_visc_ * theta_dot_
    + fric_coul_ * std::tanh(theta_dot_ / std::max(fric_eps_, 1e-9));
  double tau_damp = damping_theta_ * theta_dot_;

  // 9) Scalar equation of motion
  double num = (tau_m_ + tau_pass_theta_ + tau_ext_theta_)
               - proj_last_ - tau_fric - tau_damp;
  theta_ddot_ = std::clamp(num / denom, -max_theta_ddot_, max_theta_ddot_);

  // 10) Extended diagnostics
  tau_fric_last_       = tau_fric;
  tau_damp_last_       = tau_damp;
  num_last_            = num;
  dyn_residual_last_   = denom_last_ * theta_ddot_ - num_last_;
  tau_model_last_      = denom_last_ * theta_ddot_
                         - tau_ext_theta_ - tau_pass_theta_
                         + proj_last_ + tau_fric + tau_damp;
  tau_model_error_last_ = tau_m_ - tau_model_last_;

  // 11) Explicit Euler integration
  theta_dot_ = std::clamp(
    theta_dot_ + theta_ddot_ * dt_, -max_theta_dot_, max_theta_dot_);
  double theta_next = clamp_theta_bounds(theta_ + theta_dot_ * dt_);

  if (limit_hold_) {
    if (std::abs(theta_next - theta_) < 1e-12 && std::abs(theta_dot_) > 1e-8) {
      at_limit_ = true;
      limit_dir_ = (std::abs(theta_dot_) > 1e-9) ? (theta_dot_ > 0.0 ? 1.0 : -1.0) : 1.0;
      theta_dot_  = 0.0;
      theta_ddot_ = 0.0;
      theta_ = theta_next;
      store_valid_state();
      return;
    }
  }
  theta_ = theta_next;

  // 12) Final dq, ddq after integration
  dq_  = B_ * theta_dot_;
  ddq_ = B_ * theta_ddot_ + Bdot * theta_dot_;

  // 13) Full joint torques via RNEA
  Eigen::VectorXd tau_rnea = pin::rnea(model_, data_, q_, dq_, ddq_);
  tau_full_ = tau_rnea - tau_ext_ - tau_pass_;

  // 14) Reaction torque diagnostic
  reaction_theta_last_ = proj_last_ + tau_fric + tau_damp
    - (tau_m_ + tau_pass_theta_ + tau_ext_theta_);
}
void ExoReducedDynamicsWithHand::publish()
{
  auto now = get_clock()->now();

  // --- /joint_states ---
  sensor_msgs::msg::JointState js;
  js.header.stamp = now;
  js.name = pub_joint_names_;
  for (int idx : pub_joint_idxs_) {
    js.position.push_back(q_[idx]);
    js.velocity.push_back(dq_[idx]);
    js.effort.push_back(tau_full_[idx]);
  }
  pub_js_->publish(js);

  // --- /exo_dynamics/debug ---
  Float64ArrayStamped dbg;
  dbg.header.stamp = now;
  double cn = std::isfinite(last_closure_norm_) ? last_closure_norm_ : -1.0;
  dbg.data = std::vector<double>{
    theta_, theta_dot_, theta_ddot_, tau_m_,
    denom_last_, proj_last_, gproj_last_, cn,
    last_solver_success_ ? 1.0 : 0.0,
    static_cast<double>(last_solver_nfev_),
    last_hit_bounds_ ? 1.0 : 0.0,
    at_limit_ ? 1.0 : 0.0,
    tau_ext_theta_, tau_pass_theta_, reaction_theta_last_
  };
  pub_dbg_->publish(dbg);

  // --- /exo_dynamics/ff_terms ---
  Float64ArrayStamped ff;
  ff.header.stamp = now;
  ff.data = std::vector<double>{denom_last_, proj_last_, gproj_last_, tau_pass_theta_};
  pub_ff_terms_->publish(ff);

  // --- /exo_dynamics/model_debug ---
  Float64ArrayStamped mdbg;
  mdbg.header.stamp = now;
  mdbg.data = std::vector<double>{
    theta_, theta_dot_, theta_ddot_, tau_m_,
    tau_ext_theta_, tau_pass_theta_, denom_last_, proj_last_, gproj_last_,
    tau_fric_last_, tau_damp_last_, num_last_,
    dyn_residual_last_, tau_model_last_, tau_model_error_last_
  };
  pub_model_dbg_->publish(mdbg);

  // --- /exo_dynamics/tau_ext_theta ---
  Float64Stamped tauext;
  tauext.header.stamp = now;
  tauext.data = tau_ext_theta_;
  pub_tau_ext_theta_->publish(tauext);

  publish_ce_force();
}

void ExoReducedDynamicsWithHand::publish_ce_force()
{
  if (!ce_force_enable_) return;

  pin::computeJointJacobians(model_, data_, q_);
  pin::updateFramePlacements(model_, data_);

  Eigen::MatrixXd J6 = pin::getFrameJacobian(
    model_, data_, fid_CE_, pin::LOCAL_WORLD_ALIGNED);
  Eigen::MatrixXd Jv = J6.topRows(3);

  Eigen::Vector3d F = Eigen::Vector3d::Zero();
  try {
    F = Jv.transpose().colPivHouseholderQr().solve(tau_full_).head(3);
  } catch (...) {}

  auto now = get_clock()->now();

  geometry_msgs::msg::WrenchStamped w;
  w.header.stamp = now;
  w.header.frame_id = ce_force_frame_;
  w.wrench.force.x = F[0];
  w.wrench.force.y = F[1];
  w.wrench.force.z = F[2];
  pub_ce_force_->publish(w);

  visualization_msgs::msg::Marker m;
  m.header.frame_id = ce_force_frame_;
  m.header.stamp = now;
  m.ns = "ce_force";
  m.id = 0;
  m.type = visualization_msgs::msg::Marker::ARROW;
  m.action = visualization_msgs::msg::Marker::ADD;
  geometry_msgs::msg::Point p0, p1;
  p1.x = ce_force_marker_scale_ * F[0];
  p1.y = ce_force_marker_scale_ * F[1];
  p1.z = ce_force_marker_scale_ * F[2];
  m.points = {p0, p1};
  double d = ce_force_marker_diameter_;
  m.scale.x = d;
  m.scale.y = 2.0 * d;
  m.scale.z = 2.0 * d;
  m.color.r = 1.0f;
  m.color.a = 1.0f;
  pub_ce_marker_->publish(m);
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ExoReducedDynamicsWithHand>());
  rclcpp::shutdown();
  return 0;
}
