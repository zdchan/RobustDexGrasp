import pyrealsense2 as rs
import argparse
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import cv2

import os
import open3d as o3d

class Realsense:
    def __init__(self, camK_path, sample_pc_num, calculate_flag = False):
        print("test")

        self.calculate_flag = calculate_flag
        if calculate_flag:
            self.filter_time = 1000
            self.downsample = 1
        else:
            self.filter_time = 20
            self.downsample = 4

        self.mini_height = 0.015
        self.debug = False
        self.width = 640
        self.hight = 480
        self.rate = 30
        self.sample_pc_num = sample_pc_num
        self.camK_path = camK_path
        self.flat_npy_path = os.path.join(os.path.dirname(__file__), 'flat.npy')
        self.obj_ply_path = os.path.join(os.path.dirname(__file__), 'obj.ply')
        self.rgb_K = None
        self.depth_K = None
        
        self.all_pc = None
        
        self.init_hardware()

    # KD tree to calculate K-Nearest Neighbors for each point
    def remove_outliers(self, point_cloud, k=5, threshold=1.5):
        tree = KDTree(point_cloud)
        
        # search KNN for each point (including K itself, so it is k+1)
        distances, _ = tree.query(point_cloud, k=k+1)
        
        # mean distance of each KNN point
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # calculate Median of mean distance and MAD（Mean Absolute Deviation)
        median_distance = np.median(mean_distances)
        mad_distance = np.median(np.abs(mean_distances - median_distance))
        
        # use threshold to ignore outliers
        normalized_distances = np.abs(mean_distances - median_distance) / (mad_distance + 1e-8)
        inliers = normalized_distances < threshold
        
        return point_cloud[inliers]

    def depth2xyzmap(self, depth):
        K = self.rgb_K
        invalid_mask = (depth<0.1)
        H,W = depth.shape[:2]
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
        zs = depth[vs,us]
        xs = (us-K[0,2])*zs/K[0,0]
        ys = (vs-K[1,2])*zs/K[1,1]
        pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
        xyz_map = np.zeros((H,W,3), dtype=np.float32)
        xyz_map[vs,us] = pts
        xyz_map[invalid_mask] = 0

        mask = np.any(xyz_map != 0, axis=-1)
        filtered_points = xyz_map[mask]
        return filtered_points

    def calculate_mean_around(self, img, i, j):
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),         (0, 1),
                    (1, -1), (1, 0), (1, 1)]

        valid_values = []
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
        
            if 0 <= ni < img.shape[0] and 0 <= nj < img.shape[1]:
                if img[ni, nj] < 0.4 or img[ni, nj] > 0.8:
                    valid_values.append(img[ni, nj])

        if valid_values:
            return np.mean(valid_values)
        else:
            print(f"---error value {i}, {j}")
            return img[i, j] 

    def sample_pc(self):
        if self.all_pc.shape[0] > self.sample_pc_num*2:
            replace_flag=False
        else:
            replace_flag=True
        pose_pc_idx = np.random.choice(self.all_pc.shape[0], int(self.sample_pc_num*2), replace=replace_flag)
        pose_pc = self.all_pc[pose_pc_idx]
        pose = np.mean(pose_pc, axis=0).reshape(1,3)
        filtered_point_cloud = self.remove_outliers(pose_pc, k=15, threshold=3.0)
        if self.all_pc.shape[0] > self.sample_pc_num:
            replace_flag=False
        else:
            replace_flag=True
        output_pc_idx = np.random.choice(filtered_point_cloud.shape[0], self.sample_pc_num, replace=replace_flag)
        return filtered_point_cloud[output_pc_idx], pose
    
    def filter_pc(self, nparray): # shape is (50, self.hight, self.width)
        # rule1: ignore the value smaller then 0.4 and larger then 0.8
        rule1_mask = (nparray < 0.4) | (nparray > 0.8)
        masked_images = np.ma.masked_array(nparray, mask=rule1_mask)
        masked_images = masked_images.filled(np.nan)  
        output = np.zeros((self.hight, self.width))

        for i in range(int(self.hight/self.downsample)):
            for j in range(int(self.width/self.downsample)):
                int_i = int(self.downsample * i)
                int_j = int(self.downsample * j)
                #mask_origin = nparray[:,i,j]
                tmp = masked_images[:,i,j]
                valid_values = tmp[~np.isnan(tmp)] # ignore nan
                mask_data_value = np.sort(valid_values)
                data_num = len(mask_data_value)
                if data_num > self.filter_time / 2:
                    mask_max_min = mask_data_value[int(self.filter_time/10):-int(self.filter_time/10)]  # 去掉10%最值
                    max_value = np.max(mask_max_min)
                    min_value = np.min(mask_max_min)
                    diff = max_value - min_value
                    if diff < 0.05:
                        output[int_i][int_j] = np.mean(mask_max_min)
                    else:
                        bins = np.linspace(min_value, max_value, num=3) # devide into 2 parts
                        indices = np.digitize(mask_max_min, bins) # put data into 2 parts
                        max_mean = 0.0
                        max_cnt = 0
                        for k in range(1, len(bins)):  # check each parts
                            region_data = mask_max_min[indices == k] # get the data
                            if len(region_data) > 0:  # make sure there is data in this part
                                diff = np.max(region_data) - np.min(region_data)
                                if diff < 0.05:
                                    region_mean = np.mean(region_data)
                                    region_count = len(region_data)
                                    if max_cnt < region_count:
                                        max_cnt = region_count
                                        max_mean = region_mean
                        output[int_i][int_j] = max_mean
                        # # for debug
                        # if (max_mean > 0.1): 
                        #     plt.cla()
                        #     plt.plot(mask_max_min, 'rd')
                        #     plt.plot(mask_origin, 'bo')
                        #     plt.plot(mask_data_value, 'g*')
                        #     plt.show()
                        #     continue
                                
                if self.calculate_flag is False:
                    if abs(output[int_i][int_j] - self.flat_npy[int_i][int_j]) < self.mini_height:
                        output[int_i][int_j] = 0.0

        return output
    
    def raisim_frame_tf(self, cam_frame):
    
        Treal2sim = np.array([[  0., 1., 0., 0.],
                            [-1., 0., 0., 0.],
                            [ 0., 0., 1., 0.],
                            [ 0., 0., 0., 1.]])
        Tsim2real = np.linalg.inv(Treal2sim)
        Tsimbase = np.array([[   1., 0., 0., 0.],
                            [ 0., 1., 0., 0.],
                            [ 0., 0., 1., 0.771],
                            [ 0., 0., 0., 1.]])
        
        # tf matrix
        Tbase2cam = np.loadtxt(self.camK_path + "/base2cam.txt", delimiter=',')
        test = R.from_matrix(Tbase2cam[:3, :3])
        ang=test.as_euler('xyz', degrees=True)
        # https://support.intelrealsense.com/hc/en-us/community/posts/4405875311123-About-make-sure-FOV-specification-of-D435i 
        # tf from RGB to left-IR camera
        Tcamrgb2depth = np.array([[  1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [ 0., 0., 1., 0.0031],
                            [ 0., 0., 0., 1.]])

        T_depth = np.hstack((cam_frame, np.ones((cam_frame.shape[0], 1))))  # (N, 4)
        T_rgb = T_depth @ Tcamrgb2depth.T
        T_base = T_rgb @ Tbase2cam.T
        T_simbase = T_base @ Tsim2real.T
        T_simworld = T_simbase @ Tsimbase.T
        
        return T_simworld
    
    def init_hardware(self):
        if (os.path.exists(self.flat_npy_path)): 
            self.flat_npy = np.load(self.flat_npy_path)
        else:
            if self.calculate_flag is False:
                print(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
                print(" !!!!!!!!!!! need to calculate camera !!!!!!!!!!!!! ")
                print(" !!!!! Please clear the desktop and execute: !!!!!! ")
                print(" !!!!!!!!! python PointCloud.py -c True !!!!!!!!!!! ")
                print(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
                exit(0)

        # realsense get depth
        self.pipeline = rs.pipeline()
        config = rs.config()
        with open(self.camK_path + "/deviceid.txt",'r') as f:
            id=f.read().splitlines()[0]
            config.enable_device(id)
        config.enable_stream(rs.stream.depth, self.width, self.hight, rs.format.z16, self.rate)
        config.enable_stream(rs.stream.color, self.width, self.hight, rs.format.rgb8, self.rate)
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        wait_cnt = 0
        while True:
            frames = self.pipeline.wait_for_frames()
            wait_cnt += 1
            if wait_cnt > 50:
                aligned_frames = self.align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_intrin = aligned_frames.get_profile().as_video_stream_profile().get_intrinsics()
                depth_intrin = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
                self.rgb_K = np.array([[color_intrin.fx, 0., color_intrin.ppx], [0., color_intrin.fy, color_intrin.ppy], [0, 0, 1.]])
                self.depth_K = np.array([[depth_intrin.fx, 0., depth_intrin.ppx], [0., depth_intrin.fy, depth_intrin.ppy], [0, 0, 1.]])
                break
    
    def get_point_from_image(self, color_frame):
        points = []
        def select_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow("Image", image_display)

        # Convert image to numpy array
        image = color_frame
        image_display = image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", select_points)

        print("Click on the image to select points. Press Enter when done.")

        while True:
            cv2.imshow("Image", image_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break

        print("------------------get point: " + str(points))
        cv2.destroyAllWindows()

        return points

    def get_rgbd_frame(self):
        while True:  
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            rgb_frame = aligned_frames.get_color_frame()
            if not depth_frame or not rgb_frame:
                continue
                
            # Convert image to numpy array
            depth_image = np.asanyarray(depth_frame.get_data())/1e3
            color_image = np.asanyarray(rgb_frame.get_data())
            depth_image_scaled = (depth_image * self.depth_scale * 1000).astype(np.float32)
            H, W = color_image.shape[:2]
            color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
            depth[(depth<0.2) | (depth>=np.inf)] = 0
            
            return color, depth


    def GetPointCloud(self, mask = None):
        log_time1 = time.time()
        wait_cnt = 0
        pointcloud_xyz_list = []
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame:
                continue

            wait_cnt += 1

            depth_image = np.asanyarray(depth_frame.get_data())
            if mask is not None:
                masked_image = np.zeros_like(depth_image)
                masked_image[mask] = depth_image[mask]
                depth_image = masked_image
            downsampled_depth_image = depth_image[::self.downsample, ::self.downsample]
            depth_image_scaled = (downsampled_depth_image * self.depth_scale).astype(np.float32)
            pointcloud_xyz_list.append(depth_image_scaled)
            if wait_cnt < self.filter_time:
                continue
            
            log_time2 = time.time()
            #print(f"-------------get depth time = {log_time2 - log_time1}")
            
            output = self.filter_pc(np.array(pointcloud_xyz_list))

            log_time3 = time.time()
            #print(f"-------------filter depth time = {log_time3 - log_time2}")

            # get point cloud
            pointcloud_xyz = self.depth2xyzmap(output)
            self.all_pc = pointcloud_xyz.reshape(-1, 3).astype(np.float32)

            log_time4 = time.time()
            #print(f"-------------get point cloud time = {log_time4 - log_time3}")
            
            if self.calculate_flag is False:
                if self.debug:
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(self.all_pc)
                    o3d.visualization.draw_geometries([cloud])
                    o3d.io.write_point_cloud(self.obj_ply_path, cloud)
                pass
            else:
                output_fill = output.copy()
                for i in range(int(self.hight)):
                    for j in range(int(self.width)):
                        value = output[i][j]
                        if value < 0.4 or value > 0.8:
                            output_fill[i, j] = self.calculate_mean_around(output, i, j)

                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(self.all_pc)
                o3d.visualization.draw_geometries([cloud])
                np.save(self.flat_npy_path, output_fill)            
            break

        mask_pcd_new, mean_pose = self.sample_pc()
        tsim2realpc = self.raisim_frame_tf(mask_pcd_new)
        tsim2realpose = self.raisim_frame_tf(mean_pose)
        

        log_time5 = time.time()
        #print(f"-------------get calculate tf time = {log_time5 - log_time4}")
        print(f"-------------point cloud center: camera_frame={mean_pose}, raisim_world_frame={tsim2realpose}")
        return tsim2realpose[:, :3], tsim2realpc[:, :3]
        
def main() -> None:
    print("test ...")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--calculate_flag", help="check the table", type=bool, default=True)
    args = parser.parse_args()
    test = Realsense("/home/ubuntu/hand/calculate/0_datasets_allegro_hand_topview", 200, args.calculate_flag) # 0_datasets_allegro_hand_topview 0_datasets_allegro_hand
    pose, pc = test.GetPointCloud()

if __name__ == "__main__":
    main()
