#pragma once

#include <cstdint>

// Opt-in lease for the local V/M/F client. Legacy PICO packets never arm it.
// A timeout is latched: only a new explicit mode epoch can restart streaming.
class VmfWatchdog {
 public:
  static constexpr double kTimeout = 0.15;

  bool Command(int64_t epoch, double now) {
    if (epoch < epoch_ || (tripped_ && epoch == epoch_)) return false;
    if (!enabled_ || epoch != epoch_) {
      epoch_ = epoch;
      tripped_ = false;
    }
    enabled_ = true;
    command_time_ = now;
    return true;
  }

  void EnterStream(double now, int64_t token_sequence) {
    stream_time_ = now;
    baseline_sequence_ = token_sequence;
  }

  bool Expired(double now, bool streaming, double token_time, int64_t token_sequence) {
    if (!enabled_) return false;
    if (now - command_time_ > kTimeout ||
        (streaming && now - stream_time_ > kTimeout &&
         (token_sequence <= baseline_sequence_ || now - token_time > kTimeout))) {
      tripped_ = true;
    }
    return tripped_;
  }

  bool enabled() const { return enabled_; }
  bool tripped() const { return tripped_; }
  int64_t epoch() const { return epoch_; }

 private:
  bool enabled_ = false;
  bool tripped_ = false;
  int64_t epoch_ = -1;
  int64_t baseline_sequence_ = 0;
  double command_time_ = 0.;
  double stream_time_ = 0.;
};
