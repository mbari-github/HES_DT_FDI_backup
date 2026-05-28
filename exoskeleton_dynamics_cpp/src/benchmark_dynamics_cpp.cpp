// ROS2 C++ dynamics benchmark node.
// Same logic as dynamics_node_cpp.cpp but with per-section timing,
// auto-stop after N steps, and CSV output.
//
// Parameters (all optional):
//   urdf_path             default: exoskeleton_description share dir
//   profiling_output_file default: /tmp/benchmark_ros_cpp.csv
//   auto_stop_steps       default: 52000  (2000 warmup + 50000 measured)
//   profiling_warmup      default: 2000
//   dt                    default: 0.001

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
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <exoskeleton_safety_msgs/msg/float64_stamped.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace pin = pinocchio;
using Float64Stamped = exoskeleton_safety_msgs::msg::Float64Stamped;
using Clock = std::chrono::steady_clock;

static inline int64_t now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    Clock::now().time_since_epoch()).count();
}

static constexpr int N_PUB_JOINTS = 9;
static const char * PUB_JOINT_NAMES_ROS[N_PUB_JOINTS] = {
  "rev_crank", "rev_body2linkAC", "rev_crank2shaft", "slider",
  "rev_slider2linkBC", "rev_linkAC2linkCE",
  "rev_palmo2prossimale", "rev_prossimale2mediale", "slider2"
};

struct StepTiming {
  int64_t step;
  int64_t total_ns, closure_ns, compute_B_ns, crba_ns, ext_passive_ns, rnea_ns;
  double  closure_norm;
  int     nfev;
  double  at_limit, theta, theta_dot;
  double  q_joints[N_PUB_JOINTS];
};

class BenchmarkDynamicsCpp : public rclcpp::Node
{
public:
  BenchmarkDynamicsCpp();

private:
  // ── Pinocchio ──
  pin::Model model_;
  pin::Data  data_;
  int nq_, nv_, idx_theta_, jid_MCF_, jid_IFP_;
  std::vector<std::pair<int,int>> closure_frame_pairs_;
  std::vector<int>   idx_opt_;
  Eigen::VectorXd    lower_, upper_;

  // ── Params ──
  double dt_, motor_inertia_, fric_visc_, fric_coul_, fric_eps_, damping_theta_;
  double max_theta_dot_, max_theta_ddot_, closure_tol_, denom_min_;
  int    max_nfev_;
  double theta_min_, theta_max_;
  bool   limit_hold_, limit_use_theta_bounds_;
  double limit_release_tau_, limit_backoff_step_;
  int    limit_backoff_tries_;
  bool   passive_enable_;
  double K0_MCF_, alpha_MCF_, B_MCF_, rest_MCF_;
  double K0_IFP_, alpha_IFP_, B_IFP_, rest_IFP_;
  double passive_K_MAX_, passive_exp_clip_;
  bool   medial_soft_enable_;
  double medial_soft_lower_, medial_soft_upper_, medial_soft_weight_;
  // Profiling
  std::string profiling_output_file_;
  int         auto_stop_steps_;
  int         profiling_warmup_;

  // ── State ──
  double          theta_, theta_dot_, theta_ddot_;
  Eigen::VectorXd q_, dq_, ddq_, last_x_, B_, B_prev_;
  Eigen::VectorXd tau_ext_, tau_pass_, tau_full_;
  double          tau_ext_theta_ = 0.0, tau_pass_theta_ = 0.0;
  double          denom_last_ = 0.0, proj_last_ = 0.0, gproj_last_ = 0.0;
  bool            last_solver_success_ = false;
  int             last_solver_nfev_ = 0;
  double          last_closure_norm_ = std::numeric_limits<double>::quiet_NaN();
  bool            last_hit_bounds_ = false;
  bool            at_limit_ = false, have_valid_ = false;
  double          limit_dir_ = 0.0, theta_valid_ = 0.0;
  Eigen::VectorXd q_valid_, dq_valid_, B_valid_, last_x_valid_;
  double          tau_m_ = 0.0;

  // ── Cached publish joint indices ──
  int pub_joint_idxs_[N_PUB_JOINTS];

  // ── Step counter ──
  int step_count_ = 0;

  // ── Profiling data ──
  std::vector<StepTiming> prof_data_;

  // ── ROS I/O ──
  rclcpp::Subscription<Float64Stamped>::SharedPtr sub_tau_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_js_;
  rclcpp::TimerBase::SharedPtr step_timer_;

  // ── Methods ──
  void tau_cb(const Float64Stamped::SharedPtr msg);
  void store_valid_state();
  double clamp_theta(double t) const;
  Eigen::VectorXd closure_residual(const Eigen::VectorXd & q_loc);

  struct SolverResult { bool success; int nfev; double closure_norm; bool hit_bounds; Eigen::VectorXd x; };
  SolverResult solve_closure(double theta, bool update_warmstart);
  Eigen::VectorXd compute_B(const Eigen::VectorXd & q);
  void compute_passive_torques(const Eigen::VectorXd & q, const Eigen::VectorXd & dq);
  bool stuck_try_release();
  void step();
  void save_csv();
};

BenchmarkDynamicsCpp::BenchmarkDynamicsCpp()
: Node("benchmark_dynamics_cpp")
{
  std::string default_urdf;
  try { default_urdf = ament_index_cpp::get_package_share_directory("exoskeleton_description")
                       + "/urdf/assembly_with_hand.urdf"; }
  catch (...) {}

  declare_parameter("urdf_path", default_urdf);
  declare_parameter("dt", 0.001);
  declare_parameter("theta_init", 0.0);
  declare_parameter("motor_inertia", 0.1);
  declare_parameter("fric_visc", 2.0);
  declare_parameter("fric_coul", 2.0);
  declare_parameter("fric_eps", 0.005);
  declare_parameter("damping_theta", 1.0);
  declare_parameter("max_theta_dot", 10.0);
  declare_parameter("max_theta_ddot", 400.0);
  declare_parameter("closure_tol", 1e-5);
  declare_parameter("max_nfev", 150);
  declare_parameter("denom_min", 1e-7);
  declare_parameter("theta_min", -0.75);
  declare_parameter("theta_max",  0.09);
  declare_parameter("limit_hold", true);
  declare_parameter("limit_use_theta_bounds", true);
  declare_parameter("limit_release_tau", 0.02);
  declare_parameter("limit_backoff_step", 0.003);
  declare_parameter("limit_backoff_tries", 6);
  declare_parameter("passive_enable", true);
  declare_parameter("K0_MCF", 0.03); declare_parameter("alpha_MCF", 4.0);
  declare_parameter("B_MCF", 0.01);  declare_parameter("rest_MCF", -0.3);
  declare_parameter("K0_IFP", 0.03); declare_parameter("alpha_IFP", 4.0);
  declare_parameter("B_IFP", 0.01);  declare_parameter("rest_IFP", -0.5);
  declare_parameter("passive_K_MAX", 5.0);
  declare_parameter("passive_exp_clip", 12.0);
  declare_parameter("medial_soft_enable", true);
  declare_parameter("medial_soft_lower", -2.2);
  declare_parameter("medial_soft_upper", 0.0);
  declare_parameter("medial_soft_weight", 1.0);
  declare_parameter("profiling_output_file", std::string("/tmp/benchmark_ros_cpp.csv"));
  declare_parameter("auto_stop_steps", 52000);
  declare_parameter("profiling_warmup", 2000);

  auto urdf_path          = get_parameter("urdf_path").as_string();
  dt_                     = get_parameter("dt").as_double();
  motor_inertia_          = get_parameter("motor_inertia").as_double();
  fric_visc_              = get_parameter("fric_visc").as_double();
  fric_coul_              = get_parameter("fric_coul").as_double();
  fric_eps_               = get_parameter("fric_eps").as_double();
  damping_theta_          = get_parameter("damping_theta").as_double();
  max_theta_dot_          = get_parameter("max_theta_dot").as_double();
  max_theta_ddot_         = get_parameter("max_theta_ddot").as_double();
  closure_tol_            = get_parameter("closure_tol").as_double();
  max_nfev_               = get_parameter("max_nfev").as_int();
  denom_min_              = get_parameter("denom_min").as_double();
  theta_min_              = get_parameter("theta_min").as_double();
  theta_max_              = get_parameter("theta_max").as_double();
  limit_hold_             = get_parameter("limit_hold").as_bool();
  limit_use_theta_bounds_ = get_parameter("limit_use_theta_bounds").as_bool();
  limit_release_tau_      = get_parameter("limit_release_tau").as_double();
  limit_backoff_step_     = get_parameter("limit_backoff_step").as_double();
  limit_backoff_tries_    = get_parameter("limit_backoff_tries").as_int();
  passive_enable_         = get_parameter("passive_enable").as_bool();
  K0_MCF_                 = get_parameter("K0_MCF").as_double();
  alpha_MCF_              = get_parameter("alpha_MCF").as_double();
  B_MCF_                  = get_parameter("B_MCF").as_double();
  rest_MCF_               = get_parameter("rest_MCF").as_double();
  K0_IFP_                 = get_parameter("K0_IFP").as_double();
  alpha_IFP_              = get_parameter("alpha_IFP").as_double();
  B_IFP_                  = get_parameter("B_IFP").as_double();
  rest_IFP_               = get_parameter("rest_IFP").as_double();
  passive_K_MAX_          = get_parameter("passive_K_MAX").as_double();
  passive_exp_clip_       = get_parameter("passive_exp_clip").as_double();
  medial_soft_enable_     = get_parameter("medial_soft_enable").as_bool();
  medial_soft_lower_      = get_parameter("medial_soft_lower").as_double();
  medial_soft_upper_      = get_parameter("medial_soft_upper").as_double();
  medial_soft_weight_     = get_parameter("medial_soft_weight").as_double();
  profiling_output_file_  = get_parameter("profiling_output_file").as_string();
  auto_stop_steps_        = get_parameter("auto_stop_steps").as_int();
  profiling_warmup_       = get_parameter("profiling_warmup").as_int();

  // ── Model ──
  pin::urdf::buildModel(urdf_path, model_);
  data_ = pin::Data(model_);
  nq_ = model_.nq; nv_ = model_.nv;

  auto check_joint = [&](const std::string & name) -> int {
    size_t jid = model_.getJointId(name);
    if (jid >= static_cast<size_t>(model_.njoints))
      throw std::runtime_error("Joint not found: " + name);
    return static_cast<int>(jid);
  };

  idx_theta_ = check_joint("rev_crank") - 1;
  jid_MCF_   = check_joint("rev_palmo2prossimale");
  jid_IFP_   = check_joint("rev_prossimale2mediale");

  for (auto & [a, b] : std::vector<std::pair<std::string,std::string>>{
      {"frame_AC_end","frame_rod_end"},{"frame_AC_end_2","frame_BC_end"},
      {"frame_CE_end","frame_BC_end_2"},{"frame_CE_end_2","slider_t"}}) {
    int ia = static_cast<int>(model_.getFrameId(a));
    int ib = static_cast<int>(model_.getFrameId(b));
    if (ia >= 0 && ib >= 0) closure_frame_pairs_.push_back({ia, ib});
  }

  for (const auto & n : std::vector<std::string>{
      "rev_body2linkAC","rev_crank2shaft","slider","rev_slider2linkBC",
      "rev_linkAC2linkCE","rev_palmo2prossimale","rev_prossimale2mediale","slider2"})
    idx_opt_.push_back(static_cast<int>(model_.getJointId(n)) - 1);

  lower_ = (Eigen::VectorXd(8) << -2.5,-2.5,-0.015,-2.5,-2.5,-1.5,-2.5,-0.008).finished();
  upper_ = (Eigen::VectorXd(8) <<  2.5, 2.5, 0.004, 2.5, 2.5, 1.2, 2.5, 0.008).finished();

  for (int k = 0; k < N_PUB_JOINTS; k++) {
    size_t jid = model_.getJointId(PUB_JOINT_NAMES_ROS[k]);
    if (jid >= static_cast<size_t>(model_.njoints))
      throw std::runtime_error(std::string("Pub joint not found: ") + PUB_JOINT_NAMES_ROS[k]);
    pub_joint_idxs_[k] = static_cast<int>(jid) - 1;
  }

  // ── State init ──
  theta_    = get_parameter("theta_init").as_double();
  theta_dot_ = 0.0; theta_ddot_ = 0.0;
  q_   = pin::neutral(model_);
  dq_  = Eigen::VectorXd::Zero(nv_);
  ddq_ = Eigen::VectorXd::Zero(nv_);
  last_x_ = Eigen::VectorXd::Zero(static_cast<int>(idx_opt_.size()));
  B_      = Eigen::VectorXd::Zero(nv_);
  B_prev_ = Eigen::VectorXd::Zero(nv_);
  tau_ext_  = Eigen::VectorXd::Zero(nv_);
  tau_pass_ = Eigen::VectorXd::Zero(nv_);
  tau_full_ = Eigen::VectorXd::Zero(nv_);
  q_valid_ = q_; dq_valid_ = dq_; B_valid_ = B_; last_x_valid_ = last_x_;

  // Initial closure
  auto res0 = solve_closure(theta_, true);
  if (res0.success) {
    q_[idx_theta_] = theta_;
    for (int k = 0; k < static_cast<int>(idx_opt_.size()); k++) q_[idx_opt_[k]] = res0.x[k];
    B_ = compute_B(q_);
    store_valid_state();
  } else {
    RCLCPP_WARN(get_logger(), "Initial closure failed.");
    at_limit_ = true;
  }

  prof_data_.reserve(auto_stop_steps_);

  sub_tau_ = create_subscription<Float64Stamped>("/torque", 10,
    [this](const Float64Stamped::SharedPtr m) { tau_cb(m); });
  pub_js_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);

  step_timer_ = create_wall_timer(
    std::chrono::duration<double>(dt_), [this]() { step(); });

  RCLCPP_INFO(get_logger(), "BenchmarkDynamicsCpp ready | auto_stop=%d warmup=%d output=%s",
    auto_stop_steps_, profiling_warmup_, profiling_output_file_.c_str());
}

void BenchmarkDynamicsCpp::tau_cb(const Float64Stamped::SharedPtr msg) { tau_m_ = msg->data; }

void BenchmarkDynamicsCpp::store_valid_state() {
  have_valid_ = true; theta_valid_ = theta_;
  q_valid_ = q_; dq_valid_ = dq_; B_valid_ = B_; last_x_valid_ = last_x_;
}

double BenchmarkDynamicsCpp::clamp_theta(double t) const {
  if (!limit_use_theta_bounds_) return t;
  return std::clamp(t, theta_min_, theta_max_);
}

Eigen::VectorXd BenchmarkDynamicsCpp::closure_residual(const Eigen::VectorXd & q_loc) {
  pin::forwardKinematics(model_, data_, q_loc);
  pin::updateFramePlacements(model_, data_);
  int n = static_cast<int>(closure_frame_pairs_.size());
  Eigen::VectorXd r(3*n);
  for (int i = 0; i < n; i++) {
    auto [ia, ib] = closure_frame_pairs_[i];
    r.segment(3*i,3) = data_.oMf[ia].translation() - data_.oMf[ib].translation();
  }
  return r;
}

BenchmarkDynamicsCpp::SolverResult
BenchmarkDynamicsCpp::solve_closure(double theta, bool update_warmstart)
{
  SolverResult info{false,0,std::numeric_limits<double>::infinity(),false,{}};
  Eigen::VectorXd q = q_;
  q[idx_theta_] = theta;

  if (closure_frame_pairs_.empty()) {
    info.success=true; info.nfev=0; info.closure_norm=closure_residual(q).norm();
    info.x=last_x_; return info;
  }

  int n_dep = static_cast<int>(idx_opt_.size());
  int n_con = static_cast<int>(closure_frame_pairs_.size())*3;
  Eigen::VectorXd x = last_x_;
  constexpr double eps_b = 1e-9;

  for (int iter = 0; iter < max_nfev_; iter++) {
    Eigen::VectorXd q_loc = q;
    for (int k = 0; k < n_dep; k++) q_loc[idx_opt_[k]] = x[k];
    Eigen::VectorXd r = closure_residual(q_loc);
    int n_r = n_con;
    if (medial_soft_enable_) {
      double med = x[6]; double pen=0.0;
      if (med < medial_soft_lower_) { double e=med-medial_soft_lower_; pen+=e*e; }
      if (med > medial_soft_upper_) { double e=med-medial_soft_upper_; pen+=e*e; }
      r.conservativeResize(n_con+1);
      r[n_con]=std::sqrt(medial_soft_weight_*pen); n_r=n_con+1;
    }
    double rnorm=r.norm();
    if (rnorm < closure_tol_) {
      info.success=true; info.nfev=iter; info.closure_norm=rnorm; info.x=x;
      double epsb=1e-6;
      for (int k=0;k<n_dep&&!info.hit_bounds;k++)
        if (std::abs(x[k]-lower_[k])<epsb||std::abs(x[k]-upper_[k])<epsb) info.hit_bounds=true;
      if (update_warmstart) last_x_=x;
      return info;
    }
    pin::computeJointJacobians(model_,data_,q_loc);
    Eigen::MatrixXd J(n_r,n_dep); J.setZero();
    for (int i=0;i<static_cast<int>(closure_frame_pairs_.size());i++) {
      auto [ia,ib]=closure_frame_pairs_[i];
      Eigen::MatrixXd dJ=pin::getFrameJacobian(model_,data_,ia,pin::LOCAL_WORLD_ALIGNED).topRows(3)
                        -pin::getFrameJacobian(model_,data_,ib,pin::LOCAL_WORLD_ALIGNED).topRows(3);
      for (int k=0;k<n_dep;k++) J.block(3*i,k,3,1)=dJ.col(idx_opt_[k]);
    }
    std::vector<int> free; free.reserve(n_dep);
    for (int k=0;k<n_dep;k++) {
      bool atlo=x[k]<=lower_[k]+eps_b, athi=x[k]>=upper_[k]-eps_b;
      if (!atlo&&!athi) { free.push_back(k); continue; }
      double gk=J.col(k).head(n_con).dot(r.head(n_con));
      if ((atlo&&gk<0.0)||(athi&&gk>0.0)) free.push_back(k);
    }
    Eigen::VectorXd dx=Eigen::VectorXd::Zero(n_dep);
    if (!free.empty()) {
      int nf=static_cast<int>(free.size());
      Eigen::MatrixXd Jf(n_r,nf);
      for (int fi=0;fi<nf;fi++) Jf.col(fi)=J.col(free[fi]);
      Eigen::VectorXd dxf=Jf.colPivHouseholderQr().solve(r);
      for (int fi=0;fi<nf;fi++) dx[free[fi]]=dxf[fi];
    }
    x=(x-dx).cwiseMax(lower_).cwiseMin(upper_);
    info.nfev=iter+1;
  }
  Eigen::VectorXd q_loc=q;
  for (int k=0;k<n_dep;k++) q_loc[idx_opt_[k]]=x[k];
  info.closure_norm=closure_residual(q_loc).norm();
  return info;
}

Eigen::VectorXd BenchmarkDynamicsCpp::compute_B(const Eigen::VectorXd & q) {
  if (closure_frame_pairs_.empty()) {
    Eigen::VectorXd B=Eigen::VectorXd::Zero(nv_); B[idx_theta_]=1.0; return B;
  }
  int np=static_cast<int>(closure_frame_pairs_.size());
  Eigen::MatrixXd A(3*np,nv_);
  pin::computeJointJacobians(model_,data_,q);
  for (int i=0;i<np;i++) {
    auto [ia,ib]=closure_frame_pairs_[i];
    A.middleRows(3*i,3)=pin::getFrameJacobian(model_,data_,ia,pin::LOCAL_WORLD_ALIGNED).topRows(3)
                       -pin::getFrameJacobian(model_,data_,ib,pin::LOCAL_WORLD_ALIGNED).topRows(3);
  }
  Eigen::VectorXd eth=Eigen::VectorXd::Zero(nv_); eth[idx_theta_]=1.0;
  Eigen::VectorXd rhs=-A*eth;
  Eigen::MatrixXd Ar(3*np,nv_-1); int col=0;
  for (int j=0;j<nv_;j++) if (j!=idx_theta_) Ar.col(col++)=A.col(j);
  Eigen::VectorXd xr=Ar.colPivHouseholderQr().solve(rhs);
  Eigen::VectorXd B=eth; col=0;
  for (int j=0;j<nv_;j++) if (j!=idx_theta_) B[j]+=xr[col++];
  return B;
}

void BenchmarkDynamicsCpp::compute_passive_torques(
  const Eigen::VectorXd & q, const Eigen::VectorXd & dq)
{
  tau_pass_.setZero(); tau_pass_theta_=0.0;
  if (!passive_enable_) return;
  auto Kexp=[&](double K0,double a,double e)->double{
    double x=a*std::abs(e);
    return x>passive_exp_clip_?passive_K_MAX_:std::min(K0*std::exp(x),passive_K_MAX_);
  };
  int im=jid_MCF_-1, ii=jid_IFP_-1;
  double em=q[im]-rest_MCF_; tau_pass_[im]=-Kexp(K0_MCF_,alpha_MCF_,em)*em-B_MCF_*dq[im];
  double ei=q[ii]-rest_IFP_; tau_pass_[ii]=-Kexp(K0_IFP_,alpha_IFP_,ei)*ei-B_IFP_*dq[ii];
  tau_pass_theta_=B_.dot(tau_pass_);
}

bool BenchmarkDynamicsCpp::stuck_try_release() {
  if (!at_limit_||!have_valid_) return false;
  double tf=fric_visc_*theta_dot_+fric_coul_*std::tanh(theta_dot_/std::max(fric_eps_,1e-9));
  double te=tau_m_+tau_pass_theta_-tf-gproj_last_;
  if (te*limit_dir_>=-limit_release_tau_) return false;
  for (int k=1;k<=limit_backoff_tries_;k++) {
    double th=clamp_theta(theta_valid_-limit_dir_*(limit_backoff_step_*k));
    q_=q_valid_; last_x_=last_x_valid_;
    auto res=solve_closure(th,false);
    if (!res.success) continue;
    theta_=th; q_[idx_theta_]=th;
    for (int i=0;i<static_cast<int>(idx_opt_.size());i++) q_[idx_opt_[i]]=res.x[i];
    B_=compute_B(q_); theta_dot_=0.0; theta_ddot_=0.0;
    dq_.setZero(); ddq_.setZero(); at_limit_=false; store_valid_state();
    return true;
  }
  return false;
}

void BenchmarkDynamicsCpp::step()
{
  if (step_count_ >= auto_stop_steps_) return;

  // Ramp profile: prescribe theta as triangular wave over full ROM.
  // n_half = (measured steps) / 4 → 2 full cycles in the measured phase.
  int n_measured = auto_stop_steps_ - profiling_warmup_;
  int n_half     = n_measured / 4;
  double ramp_v  = (theta_max_ - theta_min_) / (n_half * dt_);
  int meas_step  = std::max(0, step_count_ - profiling_warmup_);
  int half_idx   = meas_step % (2 * n_half);
  if (half_idx < n_half) {
    theta_     = theta_min_ + ramp_v * half_idx * dt_;
    theta_dot_ = ramp_v;
  } else {
    theta_     = theta_max_ - ramp_v * (half_idx - n_half) * dt_;
    theta_dot_ = -ramp_v;
  }
  theta_ddot_ = 0.0;
  (void)tau_m_;

  int64_t t0 = now_ns();
  bool measuring = (step_count_ >= profiling_warmup_);

  StepTiming rec{};
  rec.step     = step_count_ - profiling_warmup_;
  rec.at_limit = 0.0;

  // 1) Closure
  int64_t tc0 = now_ns();
  auto res = solve_closure(theta_, true);
  rec.closure_ns   = now_ns() - tc0;
  rec.nfev         = res.nfev;
  rec.closure_norm = res.closure_norm;

  if (!res.success) {
    rec.total_ns = now_ns() - t0; rec.at_limit = 1.0;
    rec.theta = theta_; rec.theta_dot = theta_dot_;
    if (measuring) prof_data_.push_back(rec);
    step_count_++;
    if (step_count_ >= auto_stop_steps_) { save_csv(); rclcpp::shutdown(); }
    return;
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
  tau_ext_.setZero(); tau_pass_.setZero(); tau_pass_theta_ = 0.0;
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

  if (measuring) prof_data_.push_back(rec);
  step_count_++;

  if (step_count_ % 10000 == 0) {
    RCLCPP_INFO(get_logger(), "step %d/%d  theta=%.4f  mean_total=%.3f ms",
      step_count_, auto_stop_steps_, theta_,
      prof_data_.empty() ? 0.0 :
        [this]{ double s=0; for(auto&r:prof_data_) s+=r.total_ns; return s/prof_data_.size()/1e6; }());
  }

  if (step_count_ >= auto_stop_steps_) { save_csv(); rclcpp::shutdown(); }
}

void BenchmarkDynamicsCpp::save_csv()
{
  std::filesystem::create_directories(
    std::filesystem::path(profiling_output_file_).parent_path());
  std::ofstream f(profiling_output_file_);
  f << "step,total_ns,closure_ns,compute_B_ns,crba_ns,ext_passive_ns,rnea_ns,"
       "closure_norm,nfev,at_limit,theta,theta_dot";
  for (int k = 0; k < N_PUB_JOINTS; k++) f << "," << PUB_JOINT_NAMES_ROS[k];
  f << "\n";
  for (auto & r : prof_data_) {
    f << r.step << "," << r.total_ns << "," << r.closure_ns << ","
      << r.compute_B_ns << "," << r.crba_ns << "," << r.ext_passive_ns << ","
      << r.rnea_ns << "," << r.closure_norm << "," << r.nfev << ","
      << r.at_limit << "," << r.theta << "," << r.theta_dot;
    for (int k = 0; k < N_PUB_JOINTS; k++) f << "," << r.q_joints[k];
    f << "\n";
  }
  f.close();

  int n = static_cast<int>(prof_data_.size());
  double total_mean = 0;
  for (auto & r : prof_data_) total_mean += r.total_ns;
  total_mean /= n;

  RCLCPP_INFO(get_logger(),
    "Benchmark done: %d steps  mean=%.3f ms  saved=%s",
    n, total_mean/1e6, profiling_output_file_.c_str());
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BenchmarkDynamicsCpp>());
  rclcpp::shutdown();
  return 0;
}
