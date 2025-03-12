import numpy as np
import cv2
import open3d as o3d
import json

with open("camera_params.json", "r") as f:
    camera_data = json.load(f)

depth_intr = camera_data["depth_intrinsics"]
fx_d = depth_intr["fx"]  # 383.2601318359375
fy_d = depth_intr["fy"]  # 383.2601318359375
cx_d = depth_intr["ppx"] # 325.90032958984375
cy_d = depth_intr["ppy"] # 237.86862182617188

color_intr = camera_data["color_intrinsics"]
fx_c = color_intr["fx"]  # 608.2882690429688
fy_c = color_intr["fy"]  # 608.6083984375
cx_c = color_intr["ppx"] # 316.56170654296875
cy_c = color_intr["ppy"] # 231.01663208007812

R = np.array(camera_data["extrinsics"]["rotation"]).reshape(3, 3) 
T = np.array(camera_data["extrinsics"]["translation"])

def create_point_cloud(depth_path, color_path, depth_scale=1000.0):
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
    
    if depth_image is None or color_image is None:
        raise ValueError("Error loading images!")

    height, width = depth_image.shape

    # always thrown an exception because thầy linh huỳnh would be mad without passing null test case
    # so under this is, simply, overthinking!!!!!!!!!!!
    if height != 480 or width != 640:
        raise ValueError("Depth image must be 640x480 as per JSON!")
    if color_image.shape[:2] != (480, 640):
        raise ValueError("Color image must be 640x480 as per JSON!")

    points = []
    colors = []
    
    for v in range(height):
        for u in range(width):
            depth = depth_image[v, u] / depth_scale #đổi đơn vị =))))) tại sao lại bug vì không đổi đơn vị vậy, it just propotion but it works, idc
            if 0 < depth:

                # Step 1: Compute 3D point in depth camera coordinates
                x_d = (u - cx_d) * depth / fx_d
                y_d = (v - cy_d) * depth / fy_d
                z_d = depth

                # Step 2: Transform to color camera coordinates
                point_d = np.array([x_d, y_d, z_d])
                point_c = R @ point_d + T  # Apply rotation and translation

                # Step 3: Project to color image plane
                x_c = point_c[0]
                y_c = point_c[1]
                z_c = point_c[2]
                # if z_c <= 0:  # Skip points behind the camera
                #     continue

                # Perspective projection
                u_c = int((x_c * fx_c / z_c) + cx_c)
                v_c = int((y_c * fy_c / z_c) + cy_c)

                # Step 4: Check if the projected point is within the color image bounds
                if 0 <= u_c < width and 0 <= v_c < height:
                    b, g, r = color_image[v_c, u_c] / 255.0  # Normalize to [0, 1]
                    colors.append([r, g, b])
                    points.append(point_c)  # Use transformed coordinates

    # Here go the point cloud (please look good)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

    return point_cloud

depth_path = "depth.png"
color_path = "color.png"
point_cloud = create_point_cloud(depth_path, color_path)
o3d.visualization.draw_geometries([point_cloud])

# Save because debug is need not a should
o3d.io.write_point_cloud("point_cloud_w_RT.ply", point_cloud)
print("Point cloud saved as 'point_cloud_w_RT.ply'.")