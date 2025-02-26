import numpy as np
import cv2
import open3d as o3d

def create_point_cloud(depth_path, color_path, fx, fy, cx, cy, depth_scale=1000.0):
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
    
    if depth_image is None or color_image is None:
        raise ValueError("Error loading images!")

    height, width = depth_image.shape

    points = []
    colors = []
    
    for v in range(height):
        for u in range(width):
            depth = depth_image[v, u] / depth_scale 
            if depth > 0:
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth

                points.append([x, y, z])
                b, g, r = color_image[v, u] / 255.0
                colors.append([r, g, b])

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

    return point_cloud

depth_path = "depth.png"  # Path to depth image
color_path = "color.png"  # Path to color image

fx, fy = 608.2882690429688, 608.2882690429688  # Focal length
cx, cy = 383.2601318359375, 383.2601318359375  # Principal point

point_cloud = create_point_cloud(depth_path, color_path, fx, fy, cx, cy)

o3d.visualization.draw_geometries([point_cloud])

# # Save as PLY file (common format for point clouds)
# o3d.io.write_point_cloud("point_cloud.ply", point_cloud)
# print("Point cloud saved as 'point_cloud.ply'.")

# o3d.io.write_point_cloud("point_cloud.pcd", point_cloud)
# print("Point cloud saved as 'point_cloud.pcd'.")