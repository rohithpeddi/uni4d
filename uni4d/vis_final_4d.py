import numpy as np
import rerun as rr
import argparse

def log_cam_dyn(R,t,k,rgb, dyn_coords, dyn_rgb):

    h,w,_ = rgb.shape

    rr.log(
        f"/camera",
        rr.Pinhole(
            resolution=[w, h],
            image_from_camera=k,
        ),
        rr.Transform3D(translation=t, mat3x3=R),
    )

    rr.log(f"/image/rgb", rr.Image(rgb))
    rr.log(f"/dyn/pcd", rr.Points3D(dyn_coords, colors=dyn_rgb, radii=0.02))

def log_pcd(control_points, colors):

    rr.set_time_seconds("stable_time", 0)
    mask = control_points[:,2] < 50        # Filter out points that might be too far away
    rr.log(f"/static/pcd", rr.Points3D(control_points[mask], colors=colors[mask], radii=0.02))

# Set up argument parser
parser = argparse.ArgumentParser(description='Visualize 4D data from a npz file')
parser.add_argument('--results_path', type=str, default="fused_dyn_points.npz",
                    help='Path to the npz file containing results (default: fused_dyn_points.npz)')
args = parser.parse_args()

rr.init("vis")
rr.spawn()
rr.log("/", timeless=True)

RESULTS_PATH = args.results_path
results = np.load(RESULTS_PATH, allow_pickle=True)

fused_points_coords = results["static_points"]
fused_points_rgbs = results["static_rgbs"]
dyn_points_coords = results["dyn_points"]
dyn_points_rgbs = results["dyn_rgbs"]
poses = results["c2w"]
rgbs = results["rgbs"]
Ks = results["Ks"]

Rs = []
ts = []

log_pcd(fused_points_coords, fused_points_rgbs)
 
for i, pose in enumerate(poses):

    rr.set_time_seconds("stable_time", i)
    
    R = pose[:3, :3]
    t = pose[:3, 3]
    rgb = rgbs[i]

    log_cam_dyn(R,t,Ks[i],rgb, dyn_points_coords[i], dyn_points_rgbs[i])
