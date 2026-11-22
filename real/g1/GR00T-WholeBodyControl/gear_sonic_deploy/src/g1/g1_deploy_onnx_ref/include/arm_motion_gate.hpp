#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

// Hardware-order arm indices include shoulders, elbows and wrists, not fingers.
class ArmMotionGate {
 public:
  static constexpr int kFirstArm = 15;
  static constexpr int kEndArm = 29;
  std::array<float,29> apply(const std::array<float,29>& proposed,
                            const std::array<float,29>& actual,
                            bool enabled, double dt, bool control_active = true) {
    if (!control_active) {
      // INIT/WAIT uses the normal SONIC default-pose ramp for the arms too.
      // Do not capture a hold pose until ABXY starts policy control.
      for (int i=kFirstArm; i<kEndArm; ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(proposed[i]))
          throw std::runtime_error("Invalid arm initialization position");
      }
      initialized_ = false;
      enabled_ = false;
      return proposed;
    }
    auto result = proposed;
    if (!initialized_ || enabled != enabled_) {
      for (int i=kFirstArm; i<kEndArm; ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("Invalid actual arm position");
        target_[i] = actual[i];
      }
      initialized_ = true;
      enabled_ = enabled;
      // First enabled output is exactly the actual pose, not a policy jump.
      if (enabled) dt = 0;
    }
    const float limit = static_cast<float>(std::clamp(dt,0.,.05)); // 1 rad/s
    for (int i=kFirstArm; i<kEndArm; ++i) {
      if (enabled) {
        if (!std::isfinite(proposed[i])) throw std::runtime_error("Invalid arm policy target");
        target_[i] += std::clamp(proposed[i]-target_[i],-limit,limit);
      }
      result[i] = target_[i];
    }
    return result;
  }
 private:
  bool initialized_ = false;
  bool enabled_ = false;
  std::array<float,29> target_{};
};

// A newly started receiver must see disabled consent before accepting enabled.
class MotionConsentLatch {
 public:
  void receive(bool enabled, double now) {
    stamp_ = now;
    if (!enabled) { ready_ = true; enabled_ = false; }
    else if (ready_) enabled_ = true;
  }
  void invalidate() { ready_ = false; enabled_ = false; stamp_ = -1.; }
  bool enabled(double now) {
    if (stamp_<0 || now-stamp_>.5 || now<stamp_) invalidate();
    return enabled_;
  }
 private:
  bool ready_ = false;
  bool enabled_ = false;
  double stamp_ = -1.;
};
