# Command: 'C:\Program Files\Blender Foundation\Blender 4.0\blender.exe' --background --python .\blender_get_images.py

import bpy
import mathutils
import math
import numpy as np

# Parameters
model_path = f'C:/_sw/eb_python/deep_learning/_dataset/NeRF/_models/helmet/Helmet.obj'
images_path = f'C:/_sw/eb_python/deep_learning/_dataset/NeRF/_images/helmet/400x400/imgs/'
intrinsics_path = f'C:/_sw/eb_python/deep_learning/_dataset/NeRF/_images/helmet/400x400/train/intrinsics/'
extrinsics_path = f'C:/_sw/eb_python/deep_learning/_dataset/NeRF/_images/helmet/400x400/train/pose/'
hres, vres = 400, 400
cam_fov = 20.
cam_distance  = (1./2.) / np.tan(cam_fov/2 * np.pi/180.) * 1.5
cam_azimuth_step = 5.
cam_altitude_step = 5.
world_light_intensity = 10.

def generate_intrinsic_matrix(camera, hres, vres):
    sensor_width_mm = camera.data.sensor_width
    focal_length_mm = camera.data.lens
    focal_length_x_pixels = (hres / sensor_width_mm) * focal_length_mm
    focal_length_y_pixels = focal_length_x_pixels * (vres / hres)
    cx, cy = hres / 2, vres / 2
    print(f'Sensor dimensions [mm]: {sensor_width_mm}')
    print(f'focal length: {focal_length_mm}mm')
    print(f'focal length [pixel]: {focal_length_x_pixels};{focal_length_y_pixels}')
    print(f'Optical center: {cx}; {cy}')
    # Construct the intrinsic matrix
    K = np.array([
        [focal_length_x_pixels, 0, cx, 0],
        [0, focal_length_y_pixels, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return K

def write_matrix_to_file(matrix, file_path):
    with open(file_path, 'w') as file:
        for row in matrix:
            for elem in row:
                file.write(f"{elem}\n")

def get_camera_extrinsic_matrix(camera):
    # The world matrix represents the transformation from local to world coordinates.
    # We take its inverse to transform from world coordinates to camera coordinates.
    world_matrix = camera.matrix_world
    inverse_matrix = world_matrix.inverted()
    #extrinsic_matrix = np.array(inverse_matrix) # Convert the Blender Matrix to a NumPy array
    extrinsic_matrix = np.array(world_matrix)
    
    return extrinsic_matrix

# Delete all objects in scene
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
    obj.select_set(True)
bpy.ops.object.delete()

bpy.ops.wm.obj_import(filepath=model_path)
obj = bpy.context.active_object

# Scale object so that biggest dimension is set to 1
max_dimension = max(obj.dimensions)
scale_factor = 1 / max_dimension
obj.scale *= scale_factor  # Scale the object
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # Apply the scale to make it permanent

bpy.ops.transform.rotate(value=90 * (math.pi / 180), orient_axis='X', orient_type='GLOBAL')

# Configure render settings
bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = hres, vres
bpy.context.scene.render.engine = 'CYCLES' # Or 'BLENDER_EEVEE'

# Define camera positions
thetas = np.arange(0, 360, cam_azimuth_step)*np.pi/180.0
phis = np.arange(30, 90 + cam_altitude_step, cam_altitude_step)*np.pi/180.0
camera_positions = []
for phi in phis:
    for theta in thetas:
        x = cam_distance*np.cos(theta)*np.sin(phi)
        y = cam_distance*np.sin(theta)*np.sin(phi)
        z = cam_distance*np.cos(phi)
        camera_positions.append(mathutils.Vector((x, y, z)))
print(f'Total views: {len(camera_positions)}')

# Add a camera object
camera = bpy.data.objects.get('Camera')
if not camera:  # If the camera does not exist, create one
    bpy.ops.object.camera_add(location=camera_positions[1])
    camera = bpy.context.active_object
    camera.name = 'Camera'  # Rename the new camera to 'Camera'
    cam = bpy.data.objects['Camera']
    cam.data.type = 'PERSP' # Perspective camera
    # Point the camera at the model
    look_at = mathutils.Vector((0, 0, 0)) # Assuming your model is centered at the origin
    direction = look_at - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    cam.data.angle = math.radians(20.)

# Add a light object
bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[1].default_value = world_light_intensity
#bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)  # RGB + Alpha
# Add a point light
light_data = bpy.data.lights.new(name="SideLight", type='POINT')
light_data.energy = 2000  # Adjust the energy to control the brightness
light_object = bpy.data.objects.new(name="light", object_data=light_data)
bpy.context.collection.objects.link(light_object)   # Link light object to the scene so it will be rendered
light_object.location = mathutils.Vector((0, -5, 5))


bpy.context.scene.render.engine = 'CYCLES'  # Set render engine to Cycles
bpy.context.scene.render.film_transparent = True  # Enable Transparent Background

# Set up the camera and render settings
cam = bpy.data.objects['Camera']
for index, position in enumerate(camera_positions):
    cam.location = position
    print(f'Camera #{index} position: {cam.location}')
    # Point the camera at the model
    look_at = mathutils.Vector((0, 0, 0)) # Assuming your model is centered at the origin
    direction = look_at - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    
    filename = f'train_{index:04}'
    bpy.context.scene.camera = cam
    
    intrinsic_camera = generate_intrinsic_matrix(cam, hres, vres)
    write_matrix_to_file(intrinsic_camera, f'{intrinsics_path}{filename}.txt')
    
    extrinsic_matrix = get_camera_extrinsic_matrix(cam)
    write_matrix_to_file(extrinsic_matrix, f'{extrinsics_path}{filename}.txt')
    
    bpy.context.scene.render.filepath = f'{images_path}{filename}.png'
    bpy.ops.render.render(write_still=True) # Render the image and save it

# Delete all objects in scene
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
    obj.select_set(True)
bpy.ops.object.delete()