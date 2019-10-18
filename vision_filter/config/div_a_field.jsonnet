local PI = 3.14159;
local line_thickness = 10;
local penalty_area_depth = 1200;
local penalty_area_width = 2400;
local center_circle_radius = 500;

local point(x, y) = {
  x: x,
  y: y,
};

local line(name, p1, p2) = {
  name: name,
  p1: p1,
  p2: p2,
  thickness: line_thickness,
};

local arc(name, center, radius, a1, a2) = {
  name: name,
  center: center,
  radius: radius,
  a1: a1,
  a2: a2,
  thickness: line_thickness,
};

{
  local geom = self,
  field_length: 12000,
  field_width: 9000,
  goal_width: 1200,
  goal_depth: 180,
  boundary_width: 250,

  local field_half_length = self.field_length / 2,
  local field_half_width = self.field_width / 2,
  local pen_area_x = field_half_length - penalty_area_depth,
  local pen_area_y = penalty_area_width / 2,
  field_lines: [
    line('TopTouchLine', point(-field_half_length, field_half_width), point(field_half_length, field_half_width)),
    line('BottomTouchLine', point(-field_half_length, -field_half_width), point(field_half_length, -field_half_width)),
    line('LeftGoalLine', point(-field_half_length, -field_half_width), point(-field_half_length, field_half_width)),
    line('RightGoalLine', point(field_half_length, -field_half_width), point(field_half_length, field_half_width)),
    line('HalfwayLine', point(0, -field_half_width), point(0, field_half_width)),
    line('CenterLine', point(-field_half_length, 0), point(field_half_length, 0)),
    line('LeftPenaltyStretch', point(-pen_area_x, -pen_area_y), point(-pen_area_x, pen_area_y)),
    line('RightPenaltyStretch', point(pen_area_x, -pen_area_y), point(pen_area_x, pen_area_y)),
    line('LeftFieldLeftPenaltyStretch', point(-field_half_length, -pen_area_y), point(-pen_area_x, -pen_area_y)),
    line('LeftFieldRightPenaltyStretch', point(-field_half_length, pen_area_y), point(-pen_area_x, pen_area_y)),
    line('RightFieldRightPenaltyStretch', point(field_half_length, -pen_area_y), point(pen_area_x, -pen_area_y)),
    line('RightFieldLeftPenaltyStretch', point(field_half_length, pen_area_y), point(pen_area_x, pen_area_y)),
  ],
  field_arcs: [
    arc('CenterCircle', point(0, 0), center_circle_radius, 0, 2 * PI),
  ],
}
