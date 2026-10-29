import slangpy as spy
import device
import buffer
import window
import math
from . import hw2_scene
import time
import statistics
import numpy as np
import image
from PIL import Image
import os

# Pixel bounding box of a screen-space triangle, clamped to the image.
# Only pixels inside this box get a GPU thread, so we don't test the whole screen.
# If the triangle is fully off-screen, min ends up > max (callers check for that).
def screen_aabb(sp0, sp1, sp2, width, height):
    xs = (sp0[0], sp1[0], sp2[0])
    ys = (sp0[1], sp1[1], sp2[1])
    aabb_min = (max(math.floor(min(xs)), 0),           # floor the min, never below 0
                max(math.floor(min(ys)), 0))
    aabb_max = (min(math.ceil(max(xs)), width - 1),    # ceil the max, never past the last pixel
                min(math.ceil(max(ys)), height - 1))
    return aabb_min, aabb_max

# Shared helper (written in 2.4, reused by 2.5, 2.6, 2.7)
# Returns an object's (or the camera's) 4x4 transform at time t:
#  - no keyframes -> the fixed matrix parsed from the scene's "transform"
#  - keyframes    -> blend the two surrounding keyframes' parameters
#                    (interpolate_transformation), then build the matrix
#                    (compose_transformations)
def matrix_at(static_matrix, keyframes, t):
    if len(keyframes) == 0:
        return static_matrix          # not animated: use the transform parsed from the scene
    return hw2_scene.compose_transformations(interpolate_transformation(keyframes, t))

def hw2_1(args):
    slang_device = spy.create_device(type=spy.DeviceType.metal)
    compute_shader = device.load_compute_shader(slang_device, 'hw2/hw2_1.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    a = img.width / img.height                                    # aspect ratio
    # camera-space point -> pixel coordinates.
    # The visible image plane spans x in [-s*a, s*a] and y in [-s, s]; we remap that to [0, width] x [0, height].
    def to_screen(p):
        x, y, z = p
        xp, yp = -x / z, -y / z                                   # Eq. 1: project onto the plane z = -1
        sx = img.width * (xp + args.s * a) / (2 * args.s * a)     # Eq. 3
        sy = img.height * (args.s - yp) / (2 * args.s)            # y version: flipped (screen y points down)
        return (sx, sy)

    sp0 = to_screen(args.p0)
    sp1 = to_screen(args.p1)
    sp2 = to_screen(args.p2)

    # only launch threads for pixels the triangle could cover
    aabb_min, aabb_max = screen_aabb(sp0, sp1, sp2, img.width, img.height)

    vars = {'sp0': sp0,
            'sp1': sp1,
            'sp2': sp2,
            'aabb_min': aabb_min,
            'color': args.color}

    def render(command_encoder, vars, img):
        # Clear the image to background color
        command_encoder.clear_texture_float(img, clear_value=spy.float4(0.5, 0.5, 0.5, 1.0))

        if aabb_min[0] > aabb_max[0] or aabb_min[1] > aabb_max[1]:
            # the entire triangle is outside the screen
            return

        # Launch the kernel
        compute_shader.kernel.dispatch(
            thread_count=[aabb_max[0] - aabb_min[0] + 1,      # one thread per pixel in the AABB
                          aabb_max[1] - aabb_min[1] + 1,
                          1],
            command_encoder=command_encoder,
            image=img,
            **vars,
        )

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(vars, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, vars, img)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, img.to_numpy())

def hw2_2(args):
    slang_device = spy.create_device(type=spy.DeviceType.metal)
    compute_shader = device.load_compute_shader(slang_device, 'hw2/hw2_2.slang')
    color_image = buffer.create_texture_2d(slang_device, args.width, args.height, num_channels=4)
    depth_image = buffer.create_texture_2d(slang_device, args.width, args.height, num_channels=1)

    a = args.width / args.height

    mesh = hw2_scene.parse_triangle_mesh(args.mesh)

    def to_screen(p):
        # HW2.2: implement this (basically the same as HW2.1)
        x, y, z = (float(v) for v in p)                               # numpy row -> plain floats
        xp, yp = -x / z, -y / z                                       # Eq. 1
        sx = args.width * (xp + args.s * a) / (2 * args.s * a)        # Eq. 3
        sy = args.height * (args.s - yp) / (2 * args.s)               # flipped y
        return (sx, sy)

    # splat the triangles onto the screen
    screen_vertices = [to_screen(mesh.vertices[vid, :]) for vid in range(mesh.vertices.shape[0])]

    def render(command_encoder, vars, img):
        # Clear color buffer
        command_encoder.clear_texture_float(
            color_image,
            clear_value=spy.float4(0.5, 0.5, 0.5, 1.0)
        )
        # Clear depth buffer
        command_encoder.clear_texture_float(
            depth_image,
            clear_value=spy.float4(float("inf"), 0.0, 0.0, 0.0)
        )
        for tri_id in range(mesh.faces.shape[0]):
            # HW2.2: implement this
            # you need to retrieve/compute:
            # sp0, sp1, sp2 from screen_vertices
            # inv_depth0, inv_depth1, inv_depth2 from the Z coordinates
            # (recall that camera-space depth is -z, which we assume is positive)
            # aabb_min & aabb_max like in 2.1
            # c0, c1, c2 from mesh.vertex_colors
            # use mesh.faces[tri_id] to obtain the vertex indices
            i0, i1, i2 = mesh.faces[tri_id]                           # vertex indices of this triangle
          
            # screen positions were already computed for every vertex above
            sp0 = screen_vertices[i0]
            sp1 = screen_vertices[i1]
            sp2 = screen_vertices[i2]
            
            # the shader needs 1/depth per vertex for perspective-correct interpolation
            inv_depth0 = -1.0 / float(mesh.vertices[i0, 2])           # depth = -z, so 1/depth = -1/z (positive)
            inv_depth1 = -1.0 / float(mesh.vertices[i1, 2])
            inv_depth2 = -1.0 / float(mesh.vertices[i2, 2])
            aabb_min, aabb_max = screen_aabb(sp0, sp1, sp2, args.width, args.height)
            # per-vertex RGB; tuple() so slangpy can pass it as a float3
            c0 = tuple(mesh.vertex_colors[i0].tolist())
            c1 = tuple(mesh.vertex_colors[i1].tolist())
            c2 = tuple(mesh.vertex_colors[i2].tolist())

            # triangle is entirely off-screen: nothing to draw
            if aabb_min[0] > aabb_max[0] or aabb_min[1] > aabb_max[1]:
                continue

            shade_vars = {'sp0': sp0,
                          'sp1': sp1,
                          'sp2': sp2,
                          'inv_depth0': inv_depth0,
                          'inv_depth1': inv_depth1,
                          'inv_depth2': inv_depth2,
                          'aabb_min': aabb_min,
                          'c0': c0,
                          'c1': c1,
                          'c2': c2,
                          's': args.s,
                          'a': a}
            compute_shader.kernel.dispatch(
                thread_count=[aabb_max[0] - aabb_min[0] + 1,
                              aabb_max[1] - aabb_min[1] + 1,
                              1],
                command_encoder=command_encoder,
                color_image=color_image,
                depth_image=depth_image,
                **shade_vars
            )
            # Synchronize to avoid race conditions on the depth buffer
            command_encoder.global_barrier()

    if args.output == None:
        app = window.App(slang_device, color_image)
        app.run({}, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, {}, color_image)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, color_image.to_numpy())

def hw2_3(args):
    slang_device = spy.create_device(type=spy.DeviceType.metal)
    compute_shader = device.load_compute_shader(slang_device, 'hw2/hw2_2.slang')
    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    color_image = buffer.create_texture_2d(slang_device, width, height, num_channels=4)
    depth_image = buffer.create_texture_2d(slang_device, width, height, num_channels=1)

    a = width / height

    # HW 2.3: implement this
    # change np.zeros into the right matrices
    #projection = np.zeros((4, 4), dtype=np.float32)
    #camera_to_screen = np.zeros((4, 4), dtype=np.float32)
    # use camera.cam_to_world
    #view_matrix = np.zeros((4, 4), dtype=np.float32)
    #world_to_screen = np.zeros((4, 4), dtype=np.float32)

    # Perspective projection: (x, y, z, 1) -> (x, y, 1, -z).
    # After dividing by w = -z this gives (x/-z, y/-z, 1/-z): Eq. 1, plus 1/depth stored in z.
    projection = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, -1, 0]], dtype=np.float32)               # Eq. 29: puts -z into w
    # Image plane -> pixels: same as Eq. 3 in to_screen, written as scale + offset.
    # The z row is identity so the 1/depth value passes through untouched.
    camera_to_screen = np.array([[width / (2 * camera.s * a), 0, 0, width / 2],
                                 [0, -height / (2 * camera.s), 0, height / 2],   # minus = y flip
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], dtype=np.float32)
    view_matrix = np.linalg.inv(camera.cam_to_world)                        # world -> camera
    # Read right to left: world -> camera -> projected -> pixels
    world_to_screen = camera_to_screen @ projection @ view_matrix
    
    screen_vertices_list = []
    inv_depth_list = []
    for mesh in scene.meshes:
        # HW 2.3: implement this
        # change np.zeros into the right matrix
        # use mesh.model_matrix
        MVP = world_to_screen @ mesh.model_matrix                           # object -> screen in one matrix
        ones = np.ones((mesh.vertices.shape[0], 1), dtype=mesh.vertices.dtype)
        vertices = np.concatenate(
            [mesh.vertices, ones],
            axis=1
        )
        # transform vertices using MVP & homogeneous coordinates division
        # The resulting vertices should contain
        # (screen_x, screen_y, inverse_depth).
        # inverse_depth should be positive
        # ---
        # HW 2.3: your code here
        # ---
        # vertices is N x 4 (one row per vertex), but MVP multiplies column vectors,
        # so transpose to 4 x N, multiply, and transpose back
        vertices = (MVP @ vertices.T).T
        vertices = vertices[:, :3] / vertices[:, 3:4]                      # homogeneous divide by w = -z
        # now each row is (screen_x, screen_y, 1/depth)
        inv_depth_list.append(np.ascontiguousarray(vertices[:, 2]))
        screen_vertices_list.append(np.ascontiguousarray(vertices[:, :2]))

    def render(command_encoder, vars, img):
        # Clear color buffer
        command_encoder.clear_texture_float(
            color_image,
            clear_value=spy.float4(scene.background[0],
                                   scene.background[1], 
                                   scene.background[2],
                                   1.0)
        )
        # Clear depth buffer
        command_encoder.clear_texture_float(
            depth_image,
            clear_value=spy.float4(float("inf"), 0.0, 0.0, 0.0)
        )
        for mesh_id in range(len(scene.meshes)):
            mesh = scene.meshes[mesh_id]
            screen_vertices = screen_vertices_list[mesh_id]
            inv_depth = inv_depth_list[mesh_id]
            for tri_id in range(mesh.faces.shape[0]):
                # HW 2.3: implement this
                # (basically the same as HW 2.2)
                # same as 2.2, except screen position and 1/depth come from the MVP pass above
                i0, i1, i2 = mesh.faces[tri_id]
                sp0 = tuple(screen_vertices[i0].tolist())                # numpy row -> (x, y) tuple
                sp1 = tuple(screen_vertices[i1].tolist())
                sp2 = tuple(screen_vertices[i2].tolist())
                inv_depth0 = float(inv_depth[i0])        # already computed by the divide above
                inv_depth1 = float(inv_depth[i1])
                inv_depth2 = float(inv_depth[i2])
                aabb_min, aabb_max = screen_aabb(sp0, sp1, sp2, width, height)
                c0 = tuple(mesh.vertex_colors[i0].tolist())
                c1 = tuple(mesh.vertex_colors[i1].tolist())
                c2 = tuple(mesh.vertex_colors[i2].tolist())
                
                if aabb_min[0] > aabb_max[0] or aabb_min[1] > aabb_max[1]:
                    continue

                shade_vars = {'sp0': sp0,
                              'sp1': sp1,
                              'sp2': sp2,
                              'inv_depth0': inv_depth0,
                              'inv_depth1': inv_depth1,
                              'inv_depth2': inv_depth2,
                              'aabb_min': aabb_min,
                              'c0': c0,
                              'c1': c1,
                              'c2': c2,
                              's': camera.s,
                              'a': a}
                compute_shader.kernel.dispatch(
                    thread_count=[aabb_max[0] - aabb_min[0] + 1,
                                  aabb_max[1] - aabb_min[1] + 1,
                                  1],
                    command_encoder=command_encoder,
                    color_image=color_image,
                    depth_image=depth_image,
                    **shade_vars
                )
                # Synchronize to avoid race conditions on the depth buffer
                command_encoder.global_barrier()

    if args.output == None:
        app = window.App(slang_device, color_image)
        app.run({}, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, {}, color_image)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, color_image.to_numpy())

# Linear interpolation, element by element: alpha = 0 gives v0, alpha = 1 gives v1
def lerp(v0, v1, alpha):
    return [(1 - alpha) * x + alpha * y for x, y in zip(v0, v1)]

def interpolate_transformation(transform_keyframes, t):
    # HW 2.4: implement this
    # Find the two keyframes surrounding t and linearly
    # interpolate the corresponding transformation parameters.
    #
    # Return a list of transformations in the same format
    # as the "transform" field in the scene JSON.
    # find the pair of keyframes around t (t is already wrapped by render_frame)
    k0, k1 = transform_keyframes[0], transform_keyframes[-1]    # fallback if t is outside every pair
    for i in range(len(transform_keyframes) - 1):
        if transform_keyframes[i]['time'] <= t <= transform_keyframes[i + 1]['time']:
            k0, k1 = transform_keyframes[i], transform_keyframes[i + 1]
            break

    # how far t is between the two keyframes, as a fraction
    span = k1['time'] - k0['time']
    alpha = 0.0 if span == 0 else (t - k0['time']) / span      # 0 at k0, 1 at k1
    alpha = min(max(alpha, 0.0), 1.0)                          # clamp in case t fell outside the pair

    # interpolate each transform's numbers, keeping the same list structure as the JSON
    interp_transforms = []
    for tr0, tr1 in zip(k0['transform'], k1['transform']):     # same count, types and order (PDF guarantees it)
        (kind, v0), = tr0.items()                              # unpack the one-key dict, e.g. {"translate": [x, y, z]}
        (_, v1), = tr1.items()
        if kind == 'lookat':
            value = {key: lerp(v0[key], v1[key], alpha)         # lerp position, target and up separately
                     for key in ('position', 'target', 'up')}
        else:
            value = lerp(v0, v1, alpha)                         # scale, translate, rotate angle
        interp_transforms.append({kind: value})

    return interp_transforms
    
def hw2_4(args):
    slang_device = spy.create_device(type=spy.DeviceType.metal)
    compute_shader = device.load_compute_shader(slang_device, 'hw2/hw2_2.slang')
    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    color_image = buffer.create_texture_2d(slang_device, width, height, num_channels=4)
    depth_image = buffer.create_texture_2d(slang_device, width, height, num_channels=1)

    a = width / height

    # HW 2.4 TODO: implement this (basically the same as 2.3)
    # change np.zeros into the right matrices
    # these two don't depend on time, so build them once outside render_frame
    projection = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, -1, 0]], dtype=np.float32)
    
    camera_to_screen = np.array([[width / (2 * camera.s * a), 0, 0, width / 2],
                                 [0, -height / (2 * camera.s), 0, height / 2],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], dtype=np.float32)

    def render_frame(command_encoder, t):
        t = t % scene.duration
        # HW 2.4 TODO: insert your code here and fill in world_to_screen
        # by using interpolate_transformation to compute the view matrix at time t
        # world_to_screen = np.zeros((4, 4), dtype=np.float32)
        cam_to_world = matrix_at(camera.cam_to_world, camera.transform_keyframes, t)   # camera at time t
        world_to_screen = camera_to_screen @ projection @ np.linalg.inv(cam_to_world)  # inv = view matrix

        # same vertex transform as 2.3, but redone every frame since things move
        screen_vertices_list = []
        inv_depth_list = []
        for mesh in scene.meshes:
            # HW 2.4 TODO: insert your code here and fill in the MVP matrix
            # by using interpolate_transformation to compute the model matrix at time t
            MVP = world_to_screen @ matrix_at(mesh.model_matrix, mesh.transform_keyframes, t)   # mesh at time t
            ones = np.ones((mesh.vertices.shape[0], 1), dtype=mesh.vertices.dtype)
            vertices = np.concatenate(
                [mesh.vertices, ones],
                axis=1
            )
            # HW 2.4 TODO: insert your code here (basically the same as 2.3)
            vertices = (MVP @ vertices.T).T                                # transpose: rows -> columns and back
            vertices = vertices[:, :3] / vertices[:, 3:4]                  # homogeneous divide by w = -z
            # each row is now (screen_x, screen_y, 1/depth)
            inv_depth_list.append(np.ascontiguousarray(vertices[:, 2]))
            screen_vertices_list.append(np.ascontiguousarray(vertices[:, :2]))

        
        # Clear color buffer
        command_encoder.clear_texture_float(
            color_image,
            clear_value=spy.float4(scene.background[0],
                                   scene.background[1], 
                                   scene.background[2],
                                   1.0)
        )
        # Clear depth buffer
        command_encoder.clear_texture_float(
            depth_image,
            clear_value=spy.float4(float("inf"), 0.0, 0.0, 0.0)
        )
        for mesh_id in range(len(scene.meshes)):
            mesh = scene.meshes[mesh_id]
            screen_vertices = screen_vertices_list[mesh_id]
            inv_depth = inv_depth_list[mesh_id]
            for tri_id in range(mesh.faces.shape[0]):
                # HW 2.4 TODO: implement this
                # (basically the same as HW 2.2 & 2.3)
                # identical to 2.3: gather this triangle's 3 vertices and send them to the shader
                i0, i1, i2 = mesh.faces[tri_id]
                sp0 = tuple(screen_vertices[i0].tolist())
                sp1 = tuple(screen_vertices[i1].tolist())
                sp2 = tuple(screen_vertices[i2].tolist())
                inv_depth0 = float(inv_depth[i0])
                inv_depth1 = float(inv_depth[i1])
                inv_depth2 = float(inv_depth[i2])
                aabb_min, aabb_max = screen_aabb(sp0, sp1, sp2, width, height)
                c0 = tuple(mesh.vertex_colors[i0].tolist())
                c1 = tuple(mesh.vertex_colors[i1].tolist())
                c2 = tuple(mesh.vertex_colors[i2].tolist())

                if aabb_min[0] > aabb_max[0] or aabb_min[1] > aabb_max[1]:
                    continue

                shade_vars = {'sp0': sp0,
                        'sp1': sp1,
                        'sp2': sp2,
                        'inv_depth0': inv_depth0,
                        'inv_depth1': inv_depth1,
                        'inv_depth2': inv_depth2,
                        'aabb_min': aabb_min,
                        'c0': c0,
                        'c1': c1,
                        'c2': c2,
                        's': camera.s,
                        'a': a}
                compute_shader.kernel.dispatch(
                    thread_count=[aabb_max[0] - aabb_min[0] + 1,
                                  aabb_max[1] - aabb_min[1] + 1,
                                  1],
                    command_encoder=command_encoder,
                    color_image=color_image,
                    depth_image=depth_image,
                    **shade_vars
                )
                # Synchronize to avoid race conditions on the depth buffer
                command_encoder.global_barrier()

    if args.output == None:
        app = window.App(slang_device, color_image)
        vars = {'time': 0.0}
        def update(t, vars):
            vars['time'] = t
            return vars

        def render(command_encoder, vars, img):
            render_frame(command_encoder, vars['time'])

        app.run(vars, render, update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        command_encoder = slang_device.create_command_encoder()
        render_frame(command_encoder, t)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, color_image.to_numpy())

def hw2_5(args):
    # HW 2.5 overview: the CPU (this Python) only computes one MVP matrix per mesh
    # and launches 5 GPU kernels per frame (see hw2_5.slang for each kernel):
    #   transform_vertices -> clear_buffers -> rasterize_depth -> rasterize_id -> shade
    # Each kernel is launched ONCE with many threads, instead of once per triangle.
    slang_device = spy.create_device(type=spy.DeviceType.metal)
    transform_vertices_shader = \
        device.load_compute_shader(slang_device,
            'hw2/hw2_5.slang', 'transform_vertices')
    clear_buffers_shader = \
        device.load_compute_shader(slang_device,
            'hw2/hw2_5.slang', 'clear_buffers')
    rasterize_depth_shader = \
        device.load_compute_shader(slang_device,
            'hw2/hw2_5.slang', 'rasterize_depth')
    rasterize_id_shader = \
        device.load_compute_shader(slang_device,
            'hw2/hw2_5.slang', 'rasterize_id')
    shade_shader = \
        device.load_compute_shader(slang_device,
            'hw2/hw2_5.slang', 'shade')

    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    a = width / height

    # Preprocessing: merge all meshes into one list
    # so that we can launch them on GPU
    # (One big vertex array + one big face array means ONE rasterize launch covers
    #  every triangle in the scene. Face indices get shifted by vertex_offset so
    #  they still point at the right vertices in the merged array.)
    vertices_list = []
    faces_list = []
    colors_list = []
    # need to retain the offset list for animation
    vertex_offsets = []
    vertex_offset = 0
    for mesh in scene.meshes:
        vertices_list.append(mesh.vertices)
        # the indexing needs to be updated to the new array
        faces_list.append(mesh.faces + vertex_offset)
        colors_list.append(mesh.vertex_colors)
        vertex_offsets.append(vertex_offset)
        vertex_offset += len(mesh.vertices)
    vertices_host = np.concatenate(vertices_list)
    faces_host = np.concatenate(faces_list).astype(np.uint32)
    vertex_colors_host = np.concatenate(colors_list)
    num_vertices = vertices_host.shape[0]
    num_faces = faces_host.shape[0]

    # Allocate/upload buffer
    vertices = buffer.structured_typed_buffer_from_numpy(\
        slang_device, vertices_host, 'float3')
    projected_vertices = spy.Tensor.empty(
        slang_device, shape=(num_vertices,), dtype='float3')
    faces = buffer.structured_typed_buffer_from_numpy(\
        slang_device, faces_host, 'uint3')
    vertex_colors = buffer.structured_typed_buffer_from_numpy(\
        slang_device, vertex_colors_host, 'float3')
    depth_buffer = spy.Tensor.empty(
        slang_device, shape=(width * height,), dtype='uint')
    triangle_id_buffer = spy.Tensor.empty(
        slang_device, shape=(width * height,), dtype='uint')
    color_image = buffer.create_texture_2d(\
        slang_device, width, height, num_channels=4)

    # HW 2.5 TODO: implement this (basically the same as 2.3 and 2.4)
    # change np.zeros into the right matrices
    # Perspective projection (Eq. 29). Applied to camera-space (x, y, z, 1) it gives
    # (x, y, 1, -z). After dividing by w = -z (the depth):
    #   x/-z, y/-z = point projected onto the image plane z = -1
    #   1/-z       = inverse depth, which the rasterizer needs for perspective-correct weights
    projection = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, -1, 0]], dtype=np.float32)               # z/w becomes 1/depth
    # Image plane -> pixel coordinates (Eq. 3): x in [-s*a, s*a] maps to [0, width],
    # y in [-s, s] maps to [height, 0] (minus sign = flip, since pixel y points DOWN).
    # The z row is identity so the inverse depth passes through untouched.
    camera_to_screen = np.array([[width / (2 * camera.s * a), 0, 0, width / 2],
                                 [0, -height / (2 * camera.s), 0, height / 2],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], dtype=np.float32)

    def render_frame(command_encoder, t):
        t = t % scene.duration                      # loop the animation
        # HW 2.5 TODO: implement this (basically the same as 2.4)
        cam_to_world = matrix_at(camera.cam_to_world, camera.transform_keyframes, t)   # camera at time t
        # view matrix = inverse of cam_to_world (world -> camera space).
        # Matrices apply right-to-left: world -> camera -> projected -> pixels.
        world_to_screen = camera_to_screen @ projection @ np.linalg.inv(cam_to_world)

        command_encoder.clear_texture_float(
            color_image,
            clear_value=spy.float4(scene.background[0],
                                   scene.background[1], 
                                   scene.background[2],
                                   1.0)
        )

        # Transform the vertices per mesh because each mesh
        # has a different model matrix
        for mesh, offset in zip(scene.meshes, vertex_offsets):
            # HW 2.5 TODO: implement this (basically the same as 2.4)
            # MVP = object -> world (model) -> pixels, all in one matrix
            MVP = world_to_screen @ matrix_at(mesh.model_matrix, mesh.transform_keyframes, t)   # mesh at time t
            MVP = np.asarray(MVP, dtype=np.float32)     # GPU expects 32-bit floats
            # Step 1: one GPU thread per vertex of THIS mesh applies MVP + divide by w
            transform_vertices_shader.kernel.dispatch(\
                thread_count=[mesh.vertices.shape[0], 1, 1],
                command_encoder=command_encoder,
                num_vertices=mesh.vertices.shape[0],
                vertex_offset=offset,
                MVP=spy.float4x4(MVP),
                vertices=vertices,
                projected_vertices=projected_vertices.storage
            )

        # Step 2: one thread per pixel resets depth/ID buffers to 0xffffffff
        clear_buffers_shader.kernel.dispatch(
            thread_count=[width * height, 1, 1],
            command_encoder=command_encoder,
            num_pixels=width * height,
            depth_buffer=depth_buffer.storage,
            triangle_id_buffer=triangle_id_buffer.storage)

        # Barrier = "wait until everything above has finished" before the next step
        # (the rasterizer must see fully projected vertices and cleared buffers)
        command_encoder.global_barrier()

        # Step 3: one thread per triangle -> closest depth per pixel
        rasterize_depth_shader.kernel.dispatch(\
            thread_count=[num_faces, 1, 1],
            command_encoder=command_encoder,
            num_triangles=num_faces,
            width=width,
            height=height,
            s=camera.s,
            a=a,
            faces=faces,
            projected_vertices=projected_vertices.storage,
            depth_buffer=depth_buffer.storage
        )

        # depth_buffer must be FINAL before the ID pass compares against it
        command_encoder.global_barrier()

        # Step 4: one thread per triangle -> which triangle is visible per pixel
        rasterize_id_shader.kernel.dispatch(\
            thread_count=[num_faces, 1, 1],
            command_encoder=command_encoder,
            num_triangles=num_faces,
            width=width,
            height=height,
            s=camera.s,
            a=a,
            faces=faces,
            projected_vertices=projected_vertices.storage,
            depth_buffer=depth_buffer.storage,
            triangle_id_buffer=triangle_id_buffer.storage
        )

        command_encoder.global_barrier()

        # Step 5: one thread per pixel -> color from the visible triangle
        shade_shader.kernel.dispatch(\
            thread_count=[width, height, 1],
            command_encoder=command_encoder,
            width=width,
            height=height,
            s=camera.s,
            a=a,
            faces=faces,
            vertex_colors=vertex_colors,
            projected_vertices=projected_vertices.storage,
            triangle_id_buffer=triangle_id_buffer.storage,
            image=color_image)

        command_encoder.global_barrier()

    if args.output is None:
        app = window.App(slang_device, color_image)
        vars = {'time': 0.0}
        def update(t, vars):
            vars['time'] = t
            return vars

        def render(command_encoder, vars, img):
            render_frame(command_encoder, vars['time'])

        app.run(vars, render, update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        command_encoder = slang_device.create_command_encoder()
        render_frame(command_encoder, t)
        slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        image.save_png(args.output, color_image.to_numpy())

def hw2_6(args):
    # HW 2.6 overview: use the GPU's built-in graphics pipeline instead of our own
    # rasterizer. Python sets up the pipeline (depth test, index buffers) and, each
    # frame, gives every mesh its MVP and calls draw_indexed. The GPU runs
    # vertex_main / fragment_main from hw2_6.slang and does clipping, rasterization,
    # interpolation, and depth testing itself.
    slang_device = spy.create_device(type=spy.DeviceType.metal)
    program = slang_device.load_program(
        "hw2/hw2_6.slang",
        ["vertex_main", "fragment_main"]
    )

    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    a = width / height

    # Upload meshes to GPU
    gpu_meshes = []
    for mesh in scene.meshes:
        vertices = buffer.structured_typed_buffer_from_numpy(
            slang_device, mesh.vertices, 'float3')
        vertex_colors = buffer.structured_typed_buffer_from_numpy(
            slang_device, mesh.vertex_colors, 'float3')
        indices_host = np.ascontiguousarray(
            mesh.faces.reshape(-1), dtype=np.uint32)
        indices = slang_device.create_buffer(
            size=indices_host.nbytes,
            usage=spy.BufferUsage.index_buffer,
            data=indices_host
        )
        render_state = spy.RenderState({
            'viewports': [spy.Viewport.from_size(width, height)],
            'scissor_rects': [spy.ScissorRect.from_size(width, height)],
            'index_buffer': spy.BufferOffsetPair(indices),
            'index_format': spy.IndexFormat.uint32,
        })
        gpu_meshes.append({
            'vertices': vertices,
            'indices': indices,
            'vertex_colors': vertex_colors,
            'render_state': render_state
        })

    color_image = slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba32_float,
        usage=(spy.TextureUsage.render_target |
               spy.TextureUsage.shader_resource |
               spy.TextureUsage.copy_source)
    )
    depth_image = slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.d32_float,
        usage=spy.TextureUsage.depth_stencil
    )
    color_view = color_image.create_view()
    depth_view = depth_image.create_view()

    pipeline = slang_device.create_render_pipeline(
        program=program,
        input_layout=None,
        primitive_topology=spy.PrimitiveTopology.triangle_list,
        targets=[
            spy.ColorTargetDesc({"format": color_image.format})
        ],
        depth_stencil=spy.DepthStencilDesc({
            "format": spy.Format.d32_float,
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.less,
        }),
        rasterizer=spy.RasterizerDesc({
            "cull_mode": spy.CullMode.none
        }),
    )

    render_pass_desc = spy.RenderPassDesc({
        'color_attachments': [
            spy.RenderPassColorAttachment({
                'view': color_view,
                'load_op': spy.LoadOp.clear,
                'store_op': spy.StoreOp.store,
                'clear_value': spy.float4(\
                    scene.background[0],
                    scene.background[1], 
                    scene.background[2], 1.0),
            })
        ],
        'depth_stencil_attachment': spy.RenderPassDepthStencilAttachment({
            'view': depth_view,
            'depth_load_op': spy.LoadOp.clear,
            'depth_store_op': spy.StoreOp.dont_care,
            'depth_clear_value': 1.0,
        }),
    })

    # (Provided) Projection to CLIP space instead of screen space. After the GPU
    # divides by w = -z: x, y land in [-1, 1] (visible area), depth lands in [0, 1]
    # (0 at the near plane, 1 infinitely far). The GPU then maps [-1, 1] to pixels
    # itself, so there is no camera_to_screen matrix in 2.6/2.7.
    clip_projection = np.array([
        [1 / (camera.s * a), 0, 0, 0],
        [0, 1 / camera.s, 0, 0],
        [0, 0, -1, -camera.z_near],
        [0, 0, -1, 0],
    ], dtype=np.float32)

    def render_frame(command_encoder, t):
        t = t % scene.duration                      # loop the animation
        # HW 2.6 TODO: compute the view matrix
        # remember to handle animation if keyframes exist
        cam_to_world = matrix_at(camera.cam_to_world, camera.transform_keyframes, t)   # camera at time t
        # cam_to_world places the camera in the world; its inverse (the view matrix)
        # moves the world so the camera sits at the origin looking down -z.
        view_matrix = np.linalg.inv(cam_to_world)                                      # world -> camera

        render_pass = command_encoder.begin_render_pass(render_pass_desc)

        for mesh, gpu_mesh in zip(scene.meshes, gpu_meshes):
            # HW 2.6 TODO: compute the model matrix
            # remember to handle animation if keyframes exist
            model_matrix = matrix_at(mesh.model_matrix, mesh.transform_keyframes, t)   # mesh at time t (object -> world)

            # object -> world -> camera -> clip (applied right to left)
            MVP = clip_projection @ view_matrix @ model_matrix
            MVP = np.asarray(MVP, dtype=np.float32)     # GPU expects 32-bit floats

            root = render_pass.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(root)

            cursor.find_field("MVP").write(spy.float4x4(MVP))
            cursor.find_field("vertices").write(gpu_mesh['vertices'])
            cursor.find_field("vertex_colors").write(gpu_mesh['vertex_colors'])

            render_pass.set_render_state(gpu_mesh['render_state'])
            render_pass.draw_indexed(spy.DrawArguments({
                "vertex_count": 3 * len(mesh.faces),
                "instance_count": 1,
                "start_index_location": 0,
            }))

        render_pass.end()

    if args.output is None:
        app = window.App(slang_device, color_image)
        vars = {'time': 0.0}
        def update(t, vars):
            vars['time'] = t
            return vars

        def render(command_encoder, vars, img):
            render_frame(command_encoder, vars['time'])

        app.run(vars, render, update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        command_encoder = slang_device.create_command_encoder()
        render_frame(command_encoder, t)
        slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        image.save_png(args.output, color_image.to_numpy())

def create_Kd_texture(slang_device, material, scene_dir):
    if 'texture' in material:
        filename = os.path.join(scene_dir, material['texture'])
        rgba = np.asarray(
            Image.open(filename).convert('RGBA'),
            dtype=np.uint8)
    else:
        color = material['Kd']
        rgba = np.array(
            [[[*color, 1.0]]], dtype=np.float32)
        rgba = np.uint8(np.round(np.clip(rgba, 0, 1) * 255))

    height, width = rgba.shape[:2]

    return slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba8_unorm_srgb,
        usage=spy.TextureUsage.shader_resource,
        data=np.array(rgba, dtype=np.uint8, copy=True, order='C').ravel()
    )

def hw2_7(args):
    # HW 2.7 overview: same GPU pipeline as 2.6, but each mesh also gets normals,
    # UVs, a Kd texture, and material/light values, and hw2_7.slang computes
    # Phong lighting (ambient + diffuse + specular) per pixel.
    slang_device = spy.create_device(type=spy.DeviceType.metal)
    program = slang_device.load_program(
        "hw2/hw2_7.slang",
        ["vertex_main", "fragment_main"]
    )

    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    a = width / height

    # Upload meshes to GPU
    gpu_meshes = []
    for mesh in scene.meshes:
        vertices = buffer.structured_typed_buffer_from_numpy(
            slang_device, mesh.vertices, 'float3')
        vertex_normals = buffer.structured_typed_buffer_from_numpy(
            slang_device, mesh.vertex_normals, 'float3')
        vertex_uvs = buffer.structured_typed_buffer_from_numpy(
            slang_device, mesh.vertex_uvs, 'float2')
        indices_host = np.ascontiguousarray(
            mesh.faces.reshape(-1), dtype=np.uint32)
        indices = slang_device.create_buffer(
            size=indices_host.nbytes,
            usage=spy.BufferUsage.index_buffer,
            data=indices_host
        )

        render_state = spy.RenderState({
            'viewports': [spy.Viewport.from_size(width, height)],
            'scissor_rects': [spy.ScissorRect.from_size(width, height)],
            'index_buffer': spy.BufferOffsetPair(indices),
            'index_format': spy.IndexFormat.uint32,
        })

        gpu_meshes.append({
            'vertices': vertices,
            'indices': indices,
            'vertex_normals': vertex_normals,
            'vertex_uvs': vertex_uvs,
            'render_state': render_state
        })

    color_image = slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba32_float,
        usage=(spy.TextureUsage.render_target |
               spy.TextureUsage.shader_resource |
               spy.TextureUsage.copy_source)
    )

    depth_image = slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.d32_float,
        usage=spy.TextureUsage.depth_stencil
    )
    color_view = color_image.create_view()
    depth_view = depth_image.create_view()

    pipeline = slang_device.create_render_pipeline(
        program=program,
        input_layout=None,
        primitive_topology=spy.PrimitiveTopology.triangle_list,
        targets=[
            spy.ColorTargetDesc({"format": color_image.format})
        ],
        depth_stencil=spy.DepthStencilDesc({
            "format": spy.Format.d32_float,
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.less,
        }),
        rasterizer=spy.RasterizerDesc({
            "cull_mode": spy.CullMode.none
        }),
    )

    render_pass_desc = spy.RenderPassDesc({
        'color_attachments': [
            spy.RenderPassColorAttachment({
                'view': color_view,
                'load_op': spy.LoadOp.clear,
                'store_op': spy.StoreOp.store,
                'clear_value': spy.float4(\
                    scene.background[0],
                    scene.background[1], 
                    scene.background[2], 1.0),
            })
        ],
        'depth_stencil_attachment': spy.RenderPassDepthStencilAttachment({
            'view': depth_view,
            'depth_load_op': spy.LoadOp.clear,
            'depth_store_op': spy.StoreOp.dont_care,
            'depth_clear_value': 1.0,
        }),
    })

    clip_projection = np.array([
        [1 / (camera.s * a), 0, 0, 0],
        [0, 1 / camera.s, 0, 0],
        [0, 0, -1, -camera.z_near],
        [0, 0, -1, 0],
    ], dtype=np.float32)

    Kd_textures = [
        create_Kd_texture(slang_device,
                          mesh.material,
                          os.path.dirname(args.scene))
        for mesh in scene.meshes
    ]

    Kd_sampler = slang_device.create_sampler()

    def render_frame(command_encoder, t):
        t = t % scene.duration
        # HW 2.7 TODO: compute cam_to_world and view matrix
        # remember to handle animation if keyframes exist
        # Same as 2.6. cam_to_world matters more here: its last column [:3, 3] is
        # the camera's world position, sent below as camera_position for specular.
        cam_to_world = matrix_at(camera.cam_to_world, camera.transform_keyframes, t)   # camera at time t
        view_matrix = np.linalg.inv(cam_to_world)                                      # world -> camera

        render_pass = command_encoder.begin_render_pass(render_pass_desc)

        for mesh, gpu_mesh, Kd_texture in zip(scene.meshes, gpu_meshes, Kd_textures):
            # HW 2.7 TODO: compute the model matrix
            # remember to handle animation if keyframes exist
            model_matrix = matrix_at(mesh.model_matrix, mesh.transform_keyframes, t)   # mesh at time t
            model_matrix = np.asarray(model_matrix, dtype=np.float32)                  # also sent to the GPU
            # (the vertex shader uses model_matrix to get world positions for lighting)

            # (Provided) Normal matrix N = (M3x3^-1)^T: transforms normals so they stay
            # perpendicular to the surface even under non-uniform scaling.
            normal_matrix = np.linalg.inv(model_matrix[:3, :3]).T
            normal_matrix = np.asarray(normal_matrix, dtype=np.float32)

            # object -> world -> camera -> clip (applied right to left)
            MVP = clip_projection @ view_matrix @ model_matrix
            MVP = np.asarray(MVP, dtype=np.float32)

            root = render_pass.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(root)

            cursor.find_field("MVP").write(spy.float4x4(MVP))
            cursor.find_field("model_matrix").write(spy.float4x4(model_matrix))
            cursor.find_field("normal_matrix").write(spy.float3x3(normal_matrix))
            cursor.find_field("vertices").write(gpu_mesh['vertices'])
            cursor.find_field("vertex_normals").write(gpu_mesh['vertex_normals'])
            cursor.find_field("vertex_uvs").write(gpu_mesh['vertex_uvs'])

            cursor.find_field("light_dir").write(spy.float3(scene.light_dir))
            cursor.find_field("light_color").write(spy.float3(scene.light_color))
            cursor.find_field("camera_position").write(
                spy.float3(cam_to_world[:3, 3]))

            cursor.find_field("Kd_texture").write(Kd_texture)
            cursor.find_field("Kd_sampler").write(Kd_sampler)
            cursor.find_field("Ks").write(spy.float3(mesh.material['Ks']))
            cursor.find_field("exponent").write(mesh.material['exponent'])

            render_pass.set_render_state(gpu_mesh['render_state'])
            render_pass.draw_indexed(spy.DrawArguments({
                "vertex_count": 3 * len(mesh.faces),
                "instance_count": 1,
                "start_index_location": 0,
            }))

        render_pass.end()

    if args.output == None:
        app = window.App(slang_device, color_image)
        vars = {'time': 0.0}
        def update(t, vars):
            vars['time'] = t
            return vars

        def render(command_encoder, vars, img):
            render_frame(command_encoder, vars['time'])

        app.run(vars, render, update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        command_encoder = slang_device.create_command_encoder()
        render_frame(command_encoder, t)
        slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        image.save_png(args.output, color_image.to_numpy())
