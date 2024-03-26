import bpy
import mathutils
import math
import numpy as np

# Parameters
hres, vres = 200, 200
cam_distance  = 2.
cam_azimuth_step = 1.
cam_altitude_step = 5.
world_light_intensity = 1.0

# Delete all objects in scene
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.scene.objects:
    obj.select_set(True)
bpy.ops.object.delete()

# Load 3D model
#filepath = f'C:/_sw/eb_python/blender/_models/beetle_obj/Beetle.obj'
#filepath = f'C:/_sw/eb_python/blender/_models/christmas_bear_obj/Christmas Bear.obj'
filepath = f'C:/_sw/eb_python/blender/_models/helmet/Helmet/Helmet.obj'
bpy.ops.wm.obj_import(filepath=filepath)

# Assuming the imported object is active
obj = bpy.context.active_object

# Scale object so that biggest dimension is set to 1
max_dimension = max(obj.dimensions)
print(max_dimension)
scale_factor = 1 / max_dimension
obj.scale *= scale_factor  # Scale the object
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # Apply the scale to make it permanent

bpy.ops.transform.rotate(value=90 * (math.pi / 180), orient_axis='X', orient_type='GLOBAL')

# Configure render settings
bpy.context.scene.render.resolution_x = hres # Example resolution
bpy.context.scene.render.resolution_y = vres
bpy.context.scene.render.engine = 'CYCLES' # Or 'BLENDER_EEVEE'


# Define camera positions
thetas = np.arange(0, 360, cam_azimuth_step)*np.pi/180.0
phis = np.arange(60, 135, cam_altitude_step)*np.pi/180.0
camera_positions = []
for phi in phis:
    for theta in thetas:
        #print(f'{theta}, {phi}')
        x = cam_distance*np.cos(theta)*np.sin(phi)
        y = cam_distance*np.sin(theta)*np.sin(phi)
        z = cam_distance*np.cos(phi)
        #print(f'{x}, {y}, {z}')
        camera_positions.append(mathutils.Vector((x, y, z)))
print(f'Total views: {len(camera_positions)}')

# Add a camera if not available
camera = bpy.data.objects.get('Camera')
if not camera:  # If the camera does not exist, create one
    # Create a new camera
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


bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[1].default_value = world_light_intensity
#bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)  # RGB + Alpha
# Add a point light
light_data = bpy.data.lights.new(name="SideLight", type='POINT')
light_data.energy = 2000  # Adjust the energy to control the brightness
light_object = bpy.data.objects.new(name="light", object_data=light_data)
bpy.context.collection.objects.link(light_object)   # Link light object to the scene so it will be rendered
light_object.location = mathutils.Vector((0, 5, 5))


bpy.context.scene.render.engine = 'CYCLES'  # Set render engine to Cycles
bpy.context.scene.render.film_transparent = True  # Enable Transparent Background

# Set up the camera and render settings
cam = bpy.data.objects['Camera']
cam.data.type = 'PERSP' # Perspective camera
for index, position in enumerate(camera_positions):
    cam.location = position
    print(cam.location)
    # Point the camera at the model
    look_at = mathutils.Vector((0, 0, 0)) # Assuming your model is centered at the origin
    direction = look_at - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    
    # Render the image
    bpy.context.scene.camera = cam
    #image_path = f"C:/_sw/eb_python/blender/_images/render_output_{position}.png" # Define your file path and naming convention
    image_path = f"C:/_sw/eb_python/blender/_images/out{index:04}.png" # Define your file path and naming convention
    bpy.context.scene.render.filepath = image_path
    bpy.ops.render.render(write_still=True) # Render the image and save it
