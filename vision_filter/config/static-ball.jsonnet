local detection_rate = 0.016;
local detection_time = 10;
local num_frames = std.ceil(detection_time / detection_rate);

local ball_x = 0;
local ball_y = 0;
local capture_latency = 0;

local BallDetection = [
  {
    confidence: 100,
    area: 45.0,
    x: ball_x,
    y: ball_y,
    z: 0,
  },
];

local Frame(frame_number) = {
  frame_number: frame_number,
  t_capture: detection_rate * frame_number,
  t_sent: self.t_capture + capture_latency,
  camera_id: 0,
  balls: BallDetection,
};

{
  detections:[Frame(i) for i in std.range(0, num_frames)]
}
