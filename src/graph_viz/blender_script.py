import math
import bpy

with open("data/fuel_nodes.tsv", "r") as f:
    for l in f:
        x, y, z, radius = l.strip().split("\t")
        x = int(x)
        y = int(y)
        z = int(z)
        radius = float(radius)
        location = (x, y, z)
        bpy.ops.surface.primitive_nurbs_surface_sphere_add(
            radius=radius, location=location
        )


def cylinder_between(x1, y1, z1, x2, y2, z2, r):

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=r, depth=dist, location=(dx / 2 + x1, dy / 2 + y1, dz / 2 + z1)
    )

    phi = math.atan2(dy, dx)
    theta = math.acos(dz / dist)

    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi


with open("data/fuel_edges.tsv", "r") as f:
    for l in f:
        l = l.strip()
        x1, y1, z1, x2, y2, z2, t = l.split("\t")
        x1 = int(x1)
        y1 = int(y1)
        z1 = int(z1)
        x2 = int(x2)
        y2 = int(y2)
        z2 = int(z2)

        t = float(t)

        cylinder_between(x1, y1, z1, x2, y2, z2, t)
        # bpy.ops.curve.primitive_bezier_curve_add()
        # obj = bpy.context.object
        # obj.data.dimensions = "3D"
        # obj.data.fill_mode = "FULL"
        # obj.data.bevel_depth = 1
        # obj.data.bevel_resolution = 1
        # # set first point to centre of sphere1
        # obj.data.splines[0].bezier_points[0].co = location1
        # obj.data.splines[0].bezier_points[0].handle_left_type = "VECTOR"
        # # set second point to centre of sphere2
        # obj.data.splines[0].bezier_points[1].co = location2
        # obj.data.splines[0].bezier_points[1].handle_left_type = "VECTOR"
