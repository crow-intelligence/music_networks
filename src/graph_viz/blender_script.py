import bpy

with open("data/node_coords.tsv", "r") as f:
    for l in f:
        x, y, z, radius = l.strip().split("\t")
        x = int(float(x))
        y = int(float(y))
        z = int(float(z))
        radius = float(radius)
        location = (x, y, z)
        bpy.ops.surface.primitive_nurbs_surface_sphere_add(radius=radius, location=location)

with open("data/edge_coords.tsv", "r") as f:
    for l in f:
        l = l.strip()
        x1, y1, z1, x2, y2, z2 = l.split("\t")
        x1 = int(float(x1))
        y1 = int(float(y1))
        z1 = int(float(z1))
        x2 = int(float(x2))
        y2 = int(float(y2))
        z2 = int(float(z2))
        bpy.ops.curve.primitive_bezier_curve_add()
        obj = bpy.context.object
        obj.data.dimensions = '3D'
        obj.data.fill_mode = 'FULL'
        location1 = (x1, y1, z1)
        location2 = (x2, y2, z2)
        obj.data.bevel_depth = 2
        obj.data.bevel_resolution = 4
        # set first point to centre of sphere1
        obj.data.splines[0].bezier_points[0].co = location1
        obj.data.splines[0].bezier_points[0].handle_left_type = 'VECTOR'
        # set second point to centre of sphere2
        obj.data.splines[0].bezier_points[1].co = location2
        obj.data.splines[0].bezier_points[1].handle_left_type = 'VECTOR'

