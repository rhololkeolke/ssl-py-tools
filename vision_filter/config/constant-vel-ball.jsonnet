local detection_rate = 0.016;
local detection_time = 2;
local num_frames = std.ceil(detection_time / detection_rate);

local ball_x = 0;
local ball_y = 0;
local ball_vx = 8000.0;
local ball_vy = 8000.0;
local capture_latency = 0;
local ball_friction = 686;

local BallDetection(frame_number) = [
  {
    confidence: 100,
    area: 45.0,
    x: ball_x + frame_number * detection_rate * ball_vx - frame_number * detection_rate * detection_rate * ball_friction / 2,
    y: ball_y + frame_number * detection_rate * ball_vy - frame_number * detection_rate * detection_rate * ball_friction / 2,
    z: 0,
  },
];

local Frame(frame_number) = {
  frame_number: frame_number,
  t_capture: detection_rate * frame_number,
  t_sent: self.t_capture + capture_latency,
  camera_id: 0,
  balls: BallDetection(frame_number),
};

{
  detections:[Frame(i) for i in std.range(0, num_frames)]
}
