#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <thread>
#include "../include/input_interface/zmq_manager.hpp"

namespace {
using Json = nlohmann::json;

template<class T> void Append(std::string& data, T value) {
  data.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

void Send(zmq::socket_t& publisher, const std::string& topic, Json fields,
          const std::string& data, int version) {
  std::string header = Json{{"v", version}, {"endian", "le"}, {"count", 1},
                            {"fields", fields}}.dump();
  header.resize(1280, '\0');
  publisher.send(zmq::buffer(topic + header + data), zmq::send_flags::none);
}

void Command(zmq::socket_t& publisher, int64_t epoch, bool planner) {
  Json fields = Json::array();
  for (auto name : {"start", "stop", "planner"})
    fields.push_back({{"name", name}, {"dtype", "i32"}, {"shape", {1}}});
  fields.push_back({{"name", "vmf_epoch"}, {"dtype", "i64"}, {"shape", {1}}});
  std::string data;
  Append<int32_t>(data, 1); Append<int32_t>(data, 0); Append<int32_t>(data, planner);
  Append<int64_t>(data, epoch);
  Send(publisher, "command", fields, data, 1);
}

void Pose(zmq::socket_t& publisher, int64_t epoch, int64_t frame, float token) {
  Json fields = Json::array({
    {{"name", "token_state"}, {"dtype", "f32"}, {"shape", {1, 64}}},
    {{"name", "frame_index"}, {"dtype", "i64"}, {"shape", {1}}},
    {{"name", "vmf_epoch"}, {"dtype", "i64"}, {"shape", {1}}}});
  std::string data;
  for (int i = 0; i < 64; ++i) Append<float>(data, token);
  Append<int64_t>(data, frame); Append<int64_t>(data, epoch);
  Send(publisher, "pose", fields, data, 4);
}

void Walking(zmq::socket_t& publisher, float forward) {
  Json fields = Json::array({
    {{"name", "mode"}, {"dtype", "i32"}, {"shape", {1}}},
    {{"name", "movement"}, {"dtype", "f32"}, {"shape", {3}}},
    {{"name", "facing"}, {"dtype", "f32"}, {"shape", {3}}},
    {{"name", "speed"}, {"dtype", "f32"}, {"shape", {1}}},
    {{"name", "height"}, {"dtype", "f32"}, {"shape", {1}}}});
  std::string data;
  Append<int32_t>(data, 1);
  for (float value : {forward, 0.f, 0.f, 1.f, 0.f, 0.f, .2f, -1.f}) Append<float>(data,value);
  Send(publisher,"planner",fields,data,1);
}
}  // namespace

TEST(VmfNative, RealPrivateZmqRejectsStaleEpochAndLatchesTokenTimeout) {
  // Only input subscribers on an ephemeral localhost port. No DDS, policy,
  // motors or production ports are opened by this test.
  zmq::context_t context(1);
  zmq::socket_t publisher(context, zmq::socket_type::pub);
  publisher.set(zmq::sockopt::linger, 0);
  publisher.bind("tcp://127.0.0.1:*");
  auto endpoint = publisher.get(zmq::sockopt::last_endpoint);
  int port = std::stoi(endpoint.substr(endpoint.rfind(':') + 1));
  const int original_stdin_flags = fcntl(STDIN_FILENO, F_GETFL);
  {
    ZMQManager manager("127.0.0.1", port);
    MotionDataReader reader;
    auto motion = std::make_shared<MotionSequence>();
    motion->name = "planner_motion";
    motion->SetEncodeMode(0);
    reader.motions.push_back(motion);
    std::shared_ptr<const MotionSequence> current = motion;
    OperatorState op;
    op.start = false;  // Same state as a new SONIC at WAIT_FOR_CONTROL.
    PlannerState planner{true, true};
    DataBuffer<MovementState> movement;
    movement.SetData(MovementState(0, {0., 0., 0.}, {1., 0., 0.}, 0., -1.));
    DataBuffer<HeadingState> heading;
    std::mutex motion_mutex;
    int frame = 0;
    bool reinitialize = false, temperature = false;
    auto tick = [&] {
      manager.update();
      // Simulate the independent planner's initialized state, without running
      // TensorRT or creating any robot transport.
      planner.enabled = planner.initialized = true;
      if (!manager.GetExternalTokenState().first) current = motion;
      manager.handle_input(reader, current, frame, op, reinitialize, heading,
                           true, planner, movement, motion_mutex, temperature);
    };
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    auto pump = [&](int64_t epoch, bool use_planner, int repeats) {
      for (int i = 0; i < repeats; ++i) {
        Command(publisher, epoch, use_planner);
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        tick();
      }
    };
    pump(10, true, 4);
    EXPECT_TRUE(op.start);
    EXPECT_FALSE(op.stop);
    EXPECT_EQ(manager.GetControlDiagnostics().at("control_active")[0], 1);
    EXPECT_EQ(manager.GetControlDiagnostics().at("vmf_epoch")[0], 10);
    Walking(publisher,1.f);
    pump(10,true,2);
    EXPECT_EQ(movement.GetDataWithTime().data->locomotion_mode,1);
    EXPECT_NEAR(movement.GetDataWithTime().data->movement_speed,.2,1e-6);
    EXPECT_DOUBLE_EQ(movement.GetDataWithTime().data->movement_direction[0],1.);
    Walking(publisher,-1.f);
    pump(10,true,2);
    EXPECT_EQ(movement.GetDataWithTime().data->locomotion_mode,1);
    EXPECT_DOUBLE_EQ(movement.GetDataWithTime().data->movement_direction[0],-1.);
    pump(11, false, 2);
    Pose(publisher, 11, 0, .125f);
    pump(11, false, 2);
    auto [has_token, token] = manager.GetExternalTokenState();
    ASSERT_TRUE(has_token);
    ASSERT_EQ(token.size(), 64u);
    EXPECT_DOUBLE_EQ(token[0], .125);
    Pose(publisher, 10, 1000, .5f);  // Late reference from an old mode epoch.
    pump(11, false, 2);
    EXPECT_DOUBLE_EQ(manager.GetExternalTokenState().second[0], .125);
    // Heartbeats alone cannot keep the last model token alive indefinitely.
    pump(11, false, 14);
    auto status = manager.GetControlDiagnostics();
    EXPECT_EQ(status.at("input_control_mode")[0], 0);
    EXPECT_EQ(status.at("vmf_watchdog_tripped")[0], 1);
    EXPECT_FALSE(manager.GetExternalTokenState().first);
    EXPECT_EQ(movement.GetDataWithTime().data->locomotion_mode, 0);
    pump(11, false, 2);
    EXPECT_EQ(manager.GetControlDiagnostics().at("input_control_mode")[0], 0);
    pump(12, true, 2);  // An explicit new epoch clears the latched timeout.
    EXPECT_EQ(manager.GetControlDiagnostics().at("vmf_watchdog_tripped")[0], 0);
    planner.initialized = false;
    auto before_pending_planner = std::chrono::steady_clock::now();
    manager.handle_input(reader, current, frame, op, reinitialize, heading,
                         true, planner, movement, motion_mutex, temperature);
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() -
                  before_pending_planner).count(), .05);
    EXPECT_EQ(manager.GetControlDiagnostics().at("planner_ready")[0], 0);
  }
  fcntl(STDIN_FILENO, F_SETFL, original_stdin_flags);
}
