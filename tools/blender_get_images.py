# Command: 'C:\Program Files\Blender Foundation\Blender 4.0\blender.exe' --background --log-level 4 --python .\blender_get_images.py

import bpy, mathutils, shutil, os, math, json, numpy as np

cam_parameters = {
  'Paths' : {
      '3d model': 'C:/_sw/eb_python/deep_learning/_dataset/NeRF/3d_models/helmet/Helmet.obj',
      'Output': 'C:/_sw/eb_python/deep_learning/_dataset/NeRF/images/helmet/_temp/'
  },
  '3d model': {
      'Size': 1
  },
  'Res': {
      'Hor': 800,
      'Ver': 800
  },
  'FoV': 20,
  'World light intensity': 10,
  'Distance to world center': 4.25,  #(1./2.) / np.tan(cam_parameters['FoV'] / 2.0 * np.pi/180.) * 1.5
  'Images' : {
      'Train': 100,
      'Test': 100
  }
}

def GenerateIntrinsicMatrix(camera, hres, vres):
    sensor_width_mm = camera.data.sensor_width
    focal_length_mm = camera.data.lens
    focal_length_x_pixels = (hres / sensor_width_mm) * focal_length_mm
    focal_length_y_pixels = focal_length_x_pixels * (vres / hres)
    cx, cy = hres / 2, vres / 2
    #print(f'Sensor dimensions [mm]: {sensor_width_mm}')
    #print(f'focal length: {focal_length_mm}mm')
    #print(f'focal length [pixel]: {focal_length_x_pixels};{focal_length_y_pixels}')
    #print(f'Optical center: {cx}; {cy}')
    K = np.array([
        [focal_length_x_pixels, 0, cx, 0],
        [0, focal_length_y_pixels, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return K

def WriteMatrixToFile(matrix, file_path):
    with open(file_path, 'w') as file:
        for row in matrix:
            for elem in row:
                file.write(f"{elem}\n")

def GetExtrinsicMatrix(camera):
    world_matrix = camera.matrix_world
    #inverse_matrix = world_matrix.inverted()
    #extrinsic_matrix = np.array(inverse_matrix) # Convert the Blender Matrix to a NumPy array
    extrinsic_matrix = np.array(world_matrix)
    
    return extrinsic_matrix

def SetCamera(cam, position):
    cam.location = position     # Place camera at predefined location
    look_at = mathutils.Vector((0, 0, 0)) # Pointing at model
    direction = look_at - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

def GeneratePointsUpperHemisphere(num_points, offset):
    indices = np.arange(0, num_points, dtype=float) + offset
    phi = np.arccos(1. - 2. * indices / num_points / 2.)
    theta = np.pi * (1. + 5.**0.5) * indices
    altitude = phi  # 0 is North Pole, 90 is equator, 180 is South Pole
    azimuth = np.degrees(theta) % (2. * np.pi)
    print(f'Altitude: min: {np.min(altitude):.2f}, max: {np.max(altitude):.2f}')
    print(f'Azimuth`: min: {np.min(azimuth):.2f}, max: {np.max(azimuth):.2f}')

    return zip(azimuth, altitude)

def GenerateCameraPoses(N, offset):
    points = GeneratePointsUpperHemisphere(N, offset)
    camera_positions = []
    for i, (theta, phi) in enumerate(points):
        x = cam_parameters['Distance to world center']*np.cos(theta)*np.sin(phi)
        y = cam_parameters['Distance to world center']*np.sin(theta)*np.sin(phi)
        z = cam_parameters['Distance to world center']*np.cos(phi)
        camera_positions.append(mathutils.Vector((x, y, z)))
        distance = np.sqrt(x**2 + y**2 + z**2)
        print(f'Camera #{i}  Theta: {theta:.3f}, Phi: {phi:.3f}, Distance: {distance:.1f}')
    print(f'Total views: {len(camera_positions)}')
    
    return camera_positions

def CaptureImages(path, type, camera_positions, res):
    cam = bpy.data.objects['Camera']
    for index, position in enumerate(camera_positions):
        SetCamera(cam, position)
        bpy.context.view_layer.update()     # Ensure Blender updates the scene and applies the transformation
        bpy.context.scene.camera = cam      # Update the scene's active camera to the current camera object
        
        filename = f'{type}_{index:04}'
        intrinsic_camera = GenerateIntrinsicMatrix(cam, res['Hor'], res['Ver'])
        WriteMatrixToFile(intrinsic_camera, f'{path}{type}/intrinsics/{filename}.txt')
        extrinsic_matrix = GetExtrinsicMatrix(cam)
        WriteMatrixToFile(extrinsic_matrix, f'{path}{type}/pose/{filename}.txt')
        bpy.context.scene.render.filepath = f'{path}{type}/imgs/{filename}.png'
        bpy.ops.render.render(write_still=True) # Render the image and save it

def DeleteAllObjects():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.delete()

def PrepareFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(f"{path}train/imgs", exist_ok=True)
    os.makedirs(f"{path}train/pose", exist_ok=True)
    os.makedirs(f"{path}train/intrinsics", exist_ok=True)
    os.makedirs(f"{path}test/imgs", exist_ok=True)
    os.makedirs(f"{path}test/pose", exist_ok=True)
    os.makedirs(f"{path}test/intrinsics", exist_ok=True)


DeleteAllObjects()
PrepareFolder(cam_parameters['Paths']['Output'])

# Load 3d model
bpy.ops.wm.obj_import(filepath=cam_parameters['Paths']['3d model'])
obj = bpy.context.active_object
max_dimension = max(obj.dimensions)     # Scale object so that biggest dimension is set to 1
scale_factor = cam_parameters['3d model']['Size'] / max_dimension
obj.scale *= scale_factor  # Scale the object
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # Apply the scale to make it permanent
bpy.ops.transform.rotate(value=90 * (math.pi / 180), orient_axis='X', orient_type='GLOBAL')

# Configure render settings
bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = cam_parameters['Res']['Hor'], cam_parameters['Res']['Ver']

# Add a camera object
#bpy.ops.object.camera_add(location=camera_positions[1])
bpy.ops.object.camera_add()
camera = bpy.context.active_object
camera.name = 'Camera'  # Rename the new camera to 'Camera'
cam = bpy.data.objects['Camera']
cam.data.type = 'PERSP' # Perspective camera
cam.data.lens_unit = 'FOV'
cam.data.angle = math.radians(cam_parameters['FoV'])

# Add a light object
bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[1].default_value = cam_parameters['World light intensity']

# Enable the first CUDA/OPTIX GPU found
bpy.context.scene.render.engine = 'CYCLES'  # Set render engine to Cycles
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # Use 'OPTIX' for RTX GPUs
cycles_preferences = bpy.context.preferences.addons['cycles'].preferences
cycles_preferences.get_devices()
for device in cycles_preferences.devices:
    if device.type == 'CUDA':  # Change to 'OPTIX' for RTX GPUs
        device.use = True
        break  # Enable the first CUDA device and break
bpy.context.scene.cycles.device = 'GPU'     # Set the scene's render device to GPU
bpy.context.scene.render.film_transparent = True  # Enable Transparent Background


with open(f"{cam_parameters['Paths']['Output']}info.json", 'w') as json_file:
    json.dump(cam_parameters, json_file, indent=4)
print(f"Distance camera-object: {cam_parameters['Distance to world center']}")

# Generate camera poses and capture images for training dataset
train_camera_positions = GenerateCameraPoses(cam_parameters['Images']['Train'], 0.5)
CaptureImages(cam_parameters['Paths']['Output'], 'train', train_camera_positions, cam_parameters['Res'])

# Generate camera poses and capture images for testing dataset
test_camera_positions = GenerateCameraPoses(cam_parameters['Images']['Test'], 0)
CaptureImages(cam_parameters['Paths']['Output'], 'test', test_camera_positions, cam_parameters['Res'])

DeleteAllObjects()