from typing import List

import imgui
import structlog

from vision_filter.proto.ssl.field.geometry_pb2 import GeometryFieldSize


class FieldGeometryEditor:
    def __init__(self, visible: bool):
        self._log = structlog.get_logger()
        self.visible = visible
        self._field_line_expanded: List[int] = []
        self._field_arc_expanded: List[int] = []

    def __call__(self, field_geometry: GeometryFieldSize):
        if not self.visible:
            return

        _, self.visible = imgui.begin("Field Geometry Editor", True)

        self._edit_field_size(field_geometry)
        self._edit_field_lines(field_geometry)
        self._edit_field_arcs(field_geometry)

        imgui.end()

    def _edit_field_size(self, field_geometry: GeometryFieldSize):
        imgui.text("Field Size")
        imgui.separator()
        # Field Length
        changed, value = imgui.input_int("Field Length", field_geometry.field_length)
        if changed:
            if value >= 0:
                field_geometry.field_length = value
            else:
                self._log.error(f"Field length cannot be negative. Got {value}.")
        # Field Width
        changed, value = imgui.input_int("Field Width", field_geometry.field_width)
        if changed:
            if value >= 0:
                field_geometry.field_width = value
            else:
                self._log.error(f"Field width cannot be negative. Got {value}.")
        # Goal Width
        changed, value = imgui.input_int("Goal Width", field_geometry.goal_width)
        if changed:
            if value >= 0:
                field_geometry.goal_width = value
            else:
                self._log.error(f"Goal width cannot be negative. Got {value}.")
        # Goal Depth
        changed, value = imgui.input_int("Goal Depth", field_geometry.goal_depth)
        if changed:
            if value >= 0:
                field_geometry.goal_depth = value
            else:
                self._log.error(f"Goal depth cannot be negative. Got {value}.")

        # Boundary Width
        changed, value = imgui.input_int(
            "Boundary Width", field_geometry.boundary_width
        )
        if changed:
            if value >= 0:
                field_geometry.boundary_width = value
            else:
                self._log.error(f"Boundary width cannot be negative. Got {value}.")

    def _edit_field_lines(self, field_geometry: GeometryFieldSize):
        imgui.text("Field Lines")
        imgui.separator()
        lines_to_delete = []
        for i, line in enumerate(field_geometry.field_lines):
            expanded, visible = imgui.collapsing_header(
                f"{line.name}##{i}",
                True,
                imgui.TREE_NODE_DEFAULT_OPEN if self._field_line_expanded[i] else 0,
            )
            if not visible:
                lines_to_delete.append(i)
            if expanded:
                self._field_line_expanded[i] = True
                changed, value = imgui.input_text(f"Name##{i}", line.name, 255)
                if changed:
                    line.name = value
                changed, values = imgui.input_float2(f"P1##{i}", line.p1.x, line.p1.y)
                if changed:
                    line.p1.x, line.p1.y = values[0], values[1]
                changed, values = imgui.input_float2(f"P2##{i}", line.p2.x, line.p2.y)
                if changed:
                    line.p2.x, line.p2.y = values[0], values[1]
                changed, value = imgui.input_float(f"Thickness##{i}", line.thickness)
                if changed:
                    if value > 0:
                        line.thickness = value
                    else:
                        self._log.error(
                            "Field line must have thickness > 0. "
                            f"Got {line.thickness}"
                        )
            else:
                self._field_line_expanded[i] = False
        for line in lines_to_delete[::-1]:
            del field_geometry.field_lines[line]
            del self._field_line_expanded[line]
        if imgui.button("Add field line"):
            new_line = field_geometry.field_lines.add()
            new_line.thickness = 10
            self._field_line_expanded.append(False)

    def _edit_field_arcs(self, field_geometry: GeometryFieldSize):
        imgui.text("Field Arcs")
        imgui.separator()
        arcs_to_delete = []
        for i, arc in enumerate(field_geometry.field_arcs):
            expanded, visible = imgui.collapsing_header(
                arc.name,
                True,
                imgui.TREE_NODE_DEFAULT_OPEN if self._field_arc_expanded[i] else 0,
            )
            if not visible:
                arcs_to_delete.append(i)
            if expanded:
                self._field_arc_expanded[i] = True
                changed, value = imgui.input_text(f"Name##{i}", arc.name, 255)
                if changed:
                    arc.name = value
                changed, values = imgui.input_float2(
                    "Center", arc.center.x, arc.center.y
                )
                if changed:
                    arc.center.x, arc.center.y = values
                changed, values = imgui.input_float(f"Radius##{i}", arc.radius)
                if changed:
                    if value > 0:
                        arc.radius = value
                    else:
                        self._log.error(f"Arc radius must be > 0. Got {arc.radius}")
                changed, values = imgui.input_float2(
                    f"Start Angle, End Angle##{i}", arc.a1, arc.a2
                )
                if changed:
                    arc.a1, arc.a2 = sorted(values)
                changed, value = imgui.input_float(f"Thickness##{i}", arc.thickness)
                if changed:
                    if value > 0:
                        arc.thickness = value
                    else:
                        self._log.error(
                            "Field arc must have thickness > 0. " f"Got {arc.thickness}"
                        )
        for arc in arcs_to_delete[::-1]:
            del field_geometry.field_arcs[arc]
            del self._field_arc_expanded[arc]
        if imgui.button("Add field arc"):
            new_arc = field_geometry.field_arcs.add()
            new_arc.radius = 1000
            new_arc.thickness = 10
            self._field_arc_expanded.append(False)
