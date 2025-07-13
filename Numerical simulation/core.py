import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math
import cv2
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import copy

def compute_similarity_transform_matrix(points_source, points_target):
    # Convert points to numpy arrays of type float64
    points_source = np.array(points_source, dtype=np.float64, copy=True)
    points_target = np.array(points_target, dtype=np.float64, copy=True)

    n_dims = points_source.shape[0]  # Number of dimensions

    # Translate centroids of the points to the origin
    centroid_source = -np.mean(points_source, axis=1)
    transform_source = np.identity(n_dims + 1)
    transform_source[:n_dims, n_dims] = centroid_source
    points_source += centroid_source.reshape(n_dims, 1)

    centroid_target = -np.mean(points_target, axis=1)
    transform_target = np.identity(n_dims + 1)
    transform_target[:n_dims, n_dims] = centroid_target
    points_target += centroid_target.reshape(n_dims, 1)

    # Compute the rigid transformation using SVD of the covariance matrix
    u, _, vh = np.linalg.svd(np.dot(points_target, points_source.T))
    # Compute the rotation matrix from SVD orthonormal bases
    rotation_matrix = np.dot(u, vh)

    # Ensure the rotation matrix forms a right-handed coordinate system
    if np.linalg.det(rotation_matrix) < 0.0:
        rotation_matrix -= np.outer(u[:, n_dims - 1], vh[n_dims - 1, :] * 2.0)

    # Construct the homogeneous transformation matrix
    transform_matrix = np.identity(n_dims + 1)
    transform_matrix[:n_dims, :n_dims] = rotation_matrix

    # Apply affine transformation: scale is the ratio of RMS deviations from the centroid
    points_source_squared = points_source ** 2
    points_target_squared = points_target ** 2
    scale_factor = math.sqrt(np.sum(points_target_squared) / np.sum(points_source_squared))
    transform_matrix[:n_dims, :n_dims] *= scale_factor

    # Translate centroids back to original locations
    transform_matrix = np.dot(np.linalg.inv(transform_target), np.dot(transform_matrix, transform_source))
    transform_matrix /= transform_matrix[n_dims, n_dims]

    return transform_matrix


def affine_matrix_from_points(v0, v1):
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)


    # Rigid transformation via SVD of covariance matrix
    u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
    # rotation matrix from SVD orthonormal bases
    R = np.dot(u, vh)
    if np.linalg.det(R) < 0.0:
        # R does not constitute right handed system
        R -= np.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
        s[-1] *= -1.0
    # homogeneous transformation matrix
    M = np.identity(ndims+1)
    M[:ndims, :ndims] = R

    # Affine transformation; scale is ratio of RMS deviations from centroid
    v0 *= v0
    v1 *= v1
    M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M



def simulate_reconstruction_error_detection(flight, times=10, error_levels=None):
    """
    模拟检测误差对轨迹重建误差的影响，并绘制误差分布直方图。
    
    参数：
    flight: 包含点和相机数据的对象
    times: 模拟次数，默认为10
    error_levels: 不同的误差级别，默认为[1, 2, 4, 6, 8, 10]
    output_file: 保存图形的文件名，默认为'Reconstruction-Error-of-Trajectorybydetection.pdf'
    
    返回：
    all_rmse_lists: 包含不同误差级别下的RMSE列表的列表
    mean_rmses: 每个误差级别下的平均RMSE
    percentile_95_rmses: 每个误差级别下的95%分位数RMSE
    """
    if error_levels is None:
        error_levels = [1, 2, 4, 6, 8, 10]
    
    np.random.seed(0)
    all_error_arrays = []

    error_shape = flight.cameras[0].imagepoints.shape
    
    def homogeneous(x):
        return np.vstack((x, np.ones(x.shape[1])))

    p2 = []
    for camera in flight.cameras:
        p2.append(camera.position)
    p2 = np.array(p2).T

    for error in error_levels:
        print('现在开始进行'+str(error)+'的误差计算')
        # rmse_list = []
        error_arrays_list = []
        for _ in tqdm(range(times), desc=f"Simulating for error level {error}"):
            for i in range(flight.numCam):
                flight.cameras[i].imagepoints_err = flight.cameras[i].imagepoints + np.random.uniform(-error, error, error_shape)
            t1, t2 = flight.sequence[0], flight.sequence[1]
            K1, K2 = flight.cameras[t1].K, flight.cameras[t2].K
            d1, d2 = flight.cameras[t1].imagepoints_err.T, flight.cameras[t2].imagepoints_err.T
            pts1 = d1[1:][:2].T
            pts2 = d2[1:][:2].T
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 10)
            E = np.dot(np.dot(K2.T, F), K1)
            x1, x2 = homogeneous(d1[1:]), homogeneous(d2[1:])

            X, P = triangulate_from_E(E, K1, K2, x1, x2)
            flight.cameras[t1].P = np.dot(K1, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
            flight.cameras[t2].P = np.dot(K2, P)
            flight.cameras[t1].decompose()
            flight.cameras[t2].decompose()

            for cameraid in range(2, flight.numCam):
                flight.get_camera_pose(cameraid, X[:-1].T, flight.cameras[cameraid].imagepoints_err[:, 1:])
            # 获取每台相机的location的矩阵
            p1 = []
            for i in range(flight.numCam):
                p1.append(-np.linalg.inv(flight.cameras[i].R_camera0).dot(np.array(flight.cameras[i].t_camera0)))
            p1 = np.array(p1).T
            M = compute_similarity_transform_matrix(p1, p2)
            result = np.dot(M, X)[:3]
            # Calculate RMSE and add to list
            error_arr = flight.points - result
            error_arrays_list.append(error_arr)
        all_error_arrays.append((error, error_arrays_list))
    # return all_rmse_lists, mean_rmses, percentile_95_rmses
    return all_error_arrays


def simulate_reconstruction_error_position(flight, p1, X, times=10000, error_levels=None):
    """
    模拟不同相机定位误差对轨迹重建误差的影响，并绘制误差分布直方图。
    
    参数：
    flight: 包含点和相机数据的对象
    p1: 原始相机位置的矩阵
    X: 需要变换的点
    times: 模拟次数，默认为10000
    error_levels: 不同的误差级别，默认为[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    返回：
    all_rmse_lists: 包含不同误差级别下的RMSE列表的列表
    mean_rmses: 每个误差级别下的平均RMSE
    percentile_95_rmses: 每个误差级别下的95%分位数RMSE
    """
    if error_levels is None:
        error_levels = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    np.random.seed(0)
    all_error_arrays = []

    for error in error_levels:
        # rmse_list = []
        error_arrays_list = []
        for _ in range(times):
            p2 = []
            for camera in flight.cameras:
                p2.append(camera.position)
            p2 = np.array(p2).T
            # Simulate error matrix
            error_matrix = np.random.uniform(-error, error, p2.shape)
            p2 = p2 + error_matrix
            # Compute affine transformation matrix
            M = compute_similarity_transform_matrix(p1, p2)
            # Apply transformation to points
            result = np.dot(M, X)[:3]
            # Calculate RMSE
            error_arr = flight.points - result
            error_arrays_list.append(error_arr)
        all_error_arrays.append((error, error_arrays_list))

    return all_error_arrays

def simulate_reconstruction_error_size(times=100,pixel_error=4,position_error=0.5, size_ratio=[0.25,0.5,1,1.5,2]):
    """
    模拟检测误差对轨迹重建误差的影响，并绘制误差分布直方图。
    
    参数：
    flight: 包含点和相机数据的对象
    times: 模拟次数，默认为10
    error_levels: 不同的误差级别，默认为[1, 2, 4, 6, 8, 10]
    output_file: 保存图形的文件名，默认为'Reconstruction-Error-of-Trajectorybydetection.pdf'
    
    返回：
    all_rmse_lists: 包含不同误差级别下的RMSE列表的列表
    mean_rmses: 每个误差级别下的平均RMSE
    percentile_95_rmses: 每个误差级别下的95%分位数RMSE
    """
    if size_ratio is None:
        size_ratio = [0.25,0.5,1,1.5,2]
    
    np.random.seed(0)
    all_error_arrays = []

    
    def homogeneous(x):
        return np.vstack((x, np.ones(x.shape[1])))

    for ratio in size_ratio:
        print('现在开始进行size比例为'+str(ratio)+'的误差计算')
        # rmse_list = []
        error_arrays_list = []
        pathway = ratio*np.array([
        [70, 0, 30],[0, 65, 35],[-60, 0, 40],[0, -55, 45],
        [50, 0, 50],[0, 45, 55],[-40, 0, 60],[0, -35, 65],
        [30, 0, 70],[0, 25, 75],[-20, 0, 80],[0, -15, 85],
        [10, 0, 90],[0, 5, 95],[0, 0, 100]])
        flight_origin=create_scene(numCam=3,camera_radius=ratio*100,camera_height=0,camera_toward_height=ratio*65,pathway=pathway,interp_points_num=300)
        for _ in tqdm(range(times), desc=f"Simulating for error level {ratio}"):
            flight=copy.deepcopy(flight_origin)
            p2 = []
            for camera in flight.cameras:
                p2.append(camera.position)
            p2 = np.array(p2).T
            # Simulate error matrix
            error_matrix = np.random.uniform(-position_error, position_error, p2.shape)
            p2 = p2 + error_matrix
            x1,x2,d1,d2,X = flight.init_traj()
            pixel_error_shape=flight.cameras[0].imagepoints.shape
            for i in range(flight.numCam):
                flight.cameras[i].imagepoints_err = flight.cameras[i].imagepoints + np.random.uniform(-pixel_error, pixel_error, pixel_error_shape)
            t1, t2 = flight.sequence[0], flight.sequence[1]
            K1, K2 = flight.cameras[t1].K, flight.cameras[t2].K
            d1, d2 = flight.cameras[t1].imagepoints_err.T, flight.cameras[t2].imagepoints_err.T
            pts1 = d1[1:][:2].T
            pts2 = d2[1:][:2].T
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 10)
            E = np.dot(np.dot(K2.T, F), K1)
            x1, x2 = homogeneous(d1[1:]), homogeneous(d2[1:])
            X, P = triangulate_from_E(E, K1, K2, x1, x2)
            flight.cameras[t1].P = np.dot(K1, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
            flight.cameras[t2].P = np.dot(K2, P)
            flight.cameras[t1].decompose()
            flight.cameras[t2].decompose()
            
            for cameraid in range(2, flight.numCam):
                flight.get_camera_pose(cameraid, X[:-1].T, flight.cameras[cameraid].imagepoints_err[:, 1:])
            # 获取每台相机的location的矩阵
            p1 = []
            for i in range(flight.numCam):
                p1.append(-np.linalg.inv(flight.cameras[i].R_camera0).dot(np.array(flight.cameras[i].t_camera0)))
            p1 = np.array(p1).T
            M = compute_similarity_transform_matrix(p1, p2)
            result = np.dot(M, X)[:3]
            # Calculate RMSE and add to list
            error_arr = flight.points - result
            error_arrays_list.append(error_arr)
        all_error_arrays.append((ratio, error_arrays_list))
    return all_error_arrays


import copy
def simulate_reconstruction_error_intrinsic(flight_origin,times=100, size_ratio=[0.25,0.5,1,1.5,2]):
    """
    模拟检测误差对轨迹重建误差的影响，并绘制误差分布直方图。
    
    参数：
    flight: 包含点和相机数据的对象
    times: 模拟次数，默认为10
    error_levels: 不同的误差级别，默认为[1, 2, 4, 6, 8, 10]
    output_file: 保存图形的文件名，默认为'Reconstruction-Error-of-Trajectorybydetection.pdf'
    
    返回：
    all_rmse_lists: 包含不同误差级别下的RMSE列表的列表
    mean_rmses: 每个误差级别下的平均RMSE
    percentile_95_rmses: 每个误差级别下的95%分位数RMSE
    """

    np.random.seed(0)
    all_error_arrays = []

    
    def homogeneous(x):
        return np.vstack((x, np.ones(x.shape[1])))

    for ratio in size_ratio:
        print('现在开始进行size比例为'+str(ratio)+'的误差计算')
        # rmse_list = []
        error_arrays_list = []
        for _ in tqdm(range(times), desc=f"Simulating for error level {ratio}"):
            flight=copy.deepcopy(flight_origin)
            
            p2 = []
            #对每个相机的K进行扰动
            for camera in flight.cameras:
                p2.append(camera.position)
                fx_err=camera.K[0,0]*np.random.uniform(-ratio, ratio)/100
                fy_err=camera.K[1,1]*np.random.uniform(-ratio, ratio)/100
                cw_err=np.random.uniform(-ratio, ratio)
                cy_err=np.random.uniform(-ratio, ratio)
                camera.K=camera.K+np.array([
                    [fx_err,0,cw_err],
                    [0,fy_err,cy_err],
                    [0,0,0]], dtype=np.float32)
            p2 = np.array(p2).T
            x1,x2,d1,d2,X = flight.init_traj()
            pixel_error_shape=flight.cameras[0].imagepoints.shape
            t1, t2 = flight.sequence[0], flight.sequence[1]
            K1, K2 = flight.cameras[t1].K, flight.cameras[t2].K
            d1, d2 = flight.cameras[t1].imagepoints.T, flight.cameras[t2].imagepoints.T
            pts1 = d1[1:][:2].T
            pts2 = d2[1:][:2].T
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 10)
            E = np.dot(np.dot(K2.T, F), K1)
            x1, x2 = homogeneous(d1[1:]), homogeneous(d2[1:])
            X, P = triangulate_from_E(E, K1, K2, x1, x2)
            flight.cameras[t1].P = np.dot(K1, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
            flight.cameras[t2].P = np.dot(K2, P)
            flight.cameras[t1].decompose()
            flight.cameras[t2].decompose()
            
            for cameraid in range(2, flight.numCam):
                flight.get_camera_pose(cameraid, X[:-1].T, flight.cameras[cameraid].imagepoints[:, 1:])
            # 获取每台相机的location的矩阵
            p1 = []
            for i in range(flight.numCam):
                p1.append(-np.linalg.inv(flight.cameras[i].R_camera0).dot(np.array(flight.cameras[i].t_camera0)))
            p1 = np.array(p1).T
            M = compute_similarity_transform_matrix(p1, p2)
            result = np.dot(M, X)[:3]
            # Calculate RMSE and add to list
            error_arr = flight.points - result
            error_arrays_list.append(error_arr)

        all_error_arrays.append((ratio, error_arrays_list))
    return all_error_arrays



def calculate_similarity_transformation_matrix(flight):
    """
    计算相似变换矩阵。
    
    参数：
    flight: 包含相机数据的对象
    
    返回：
    M: 相似变换矩阵
    """
    # 获取每台相机的location的矩阵
    p2 = []
    for camera in flight.cameras:
        p2.append(camera.position)
    p2 = np.array(p2).T
    
    # 获取每台相机在原始坐标系中的位置
    p1 = []
    for i in range(flight.numCam):
        camera = flight.cameras[i]
        location = -np.linalg.inv(camera.R_camera0).dot(np.array(camera.t_camera0))
        p1.append(location)
    p1 = np.array(p1).T
    
    # 计算相似变换矩阵
    M = compute_similarity_transform_matrix(p1, p2)
    
    return M


def calculate_rmse(estimated_points, true_points):
    """
    计算估计点和真实点之间的均方根误差 (RMSE)。
    
    参数：
    estimated_points: 估计点的坐标，形状为 (3, N)
    true_points: 真实点的坐标，形状为 (3, N)
    
    返回：
    rmse: 均方根误差
    """
    # 计算误差数组
    err_arr = estimated_points - true_points
    
    # 计算每个点的欧几里得距离的平方
    squared_distances = np.sum(err_arr**2, axis=0)
    
    # 计算均方误差
    mse = np.mean(squared_distances)
    
    # 计算 RMSE
    rmse = np.sqrt(mse)
    
    return rmse

def compute_Rt_from_E(E):
    '''
    Compute the camera matrix P2, where P1=[I 0] assumed

    Return 4 possible combinations of R and t as a list
    '''

    U,S,V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U,V))<0:
        V = -V
    E = np.dot(U,np.dot(np.diag([1,1,0]),V))

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1 = np.dot(U,np.dot(W,V))
    R2 = np.dot(U,np.dot(W.T,V))
    R1 = R1 * np.linalg.det(R1)
    R2 = R2 * np.linalg.det(R2)
    t1 = U[:,2].reshape((-1,1))
    t2 = -U[:,2].reshape((-1,1))

    Rt = [np.hstack((R1,t1)),np.hstack((R1,t2)),np.hstack((R2,t1)),np.hstack((R2,t2))]    
    return Rt

def triangulate_matlab(x1,x2,P1,P2):
    X = np.zeros((4,x1.shape[1]))
    for i in range(x1.shape[1]):
        r1 = np.array(x1[0,i]*P1[2] - P1[0])
        r2 = np.array(x1[1,i]*P1[2] - P1[1])
        r3 = np.array(x2[0,i]*P2[2] - P2[0])
        r4 = np.array(x2[1,i]*P2[2] - P2[1])
        A = np.array([r1,r2,r3,r4])
        U,S,V = np.linalg.svd(A)

        X[:,i] = V[-1]/V[-1,-1]
    return X


def triangulate_from_E(E,K1,K2,x1,x2):
    #把像素坐标转变为归一化坐标
    x1n = np.dot(np.linalg.inv(K1),x1)
    x2n = np.dot(np.linalg.inv(K2),x2)
    infront_max = 0
    Rt = compute_Rt_from_E(E)
    #P表示扩展矩阵Rt
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    for i in range(4):
        P2_temp = Rt[i]
        X = triangulate_matlab(x1n,x2n,P1,P2_temp)
        d1 = np.dot(P1,X)[2]
        d2 = np.dot(P2_temp,X)[2]

        if sum(d1>0)+sum(d2>0) > infront_max:
            infront_max = sum(d1>0)+sum(d2>0)
            P2 = P2_temp
            # print(i,np.all(d1>0),np.all(d2>0))
    # P1=Rt1
    # P2=Rt2
    X = triangulate_matlab(x1n,x2n,P1,P2) 
    return X[:,:], P2


def computeFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, error=3, inliers=True):
    pts1 = pts1[:2].T
    pts2 = pts2[:2].T

    F, mask = cv2.findFundamentalMat(pts1,pts2,method,error)

    if inliers:
        return F, mask.reshape(-1,)
    else:
        return F

def homogeneous(x):
    return np.vstack((x,np.ones(x.shape[1])))

class Scene:
    def __init__(self):
        self.numCam = 0
        self.cameras = []
        self.points = []
        self.sequence=None
        self.traj=None
        
    def create_camera_points(self,num_points, radius, height):
        # 计算每个点在圆周上的角度
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        # 计算每个点的坐标
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        # 将每个点的高度调整为给定的高度
        z = np.full_like(x, height)
        return np.array([x, y, z])
    

    def addCamera(self,*camera):
        for i in camera:
            self.cameras.append(i)

    def get_camera_pose(self,cam_id,objectPoints,imagePoints, error=8):
        # PnP solution from OpenCV
        K=self.cameras[cam_id].K
        distCoeffs=self.cameras[cam_id].distCoeffs
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, K, distCoeffs, reprojectionError=error)
        self.cameras[cam_id].R_camera0 = cv2.Rodrigues(rvec)[0]
        self.cameras[cam_id].t_camera0 = tvec.reshape(-1,)
        self.cameras[cam_id].compose()
    def init_traj(self,error=10,inlier_only=False):

        t1, t2 = self.sequence[0], self.sequence[1]
        K1, K2 = self.cameras[t1].K, self.cameras[t2].K

        # Find correspondences
       
        d1,d2=self.cameras[t1].imagepoints.T,self.cameras[t2].imagepoints.T


        pts1 = self.cameras[t1].imagepoints[:,1:]
        pts2 = self.cameras[t2].imagepoints[:,1:]
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC,10)
        E = np.dot(np.dot(K2.T,F),K1)
        x1, x2 = homogeneous(d1[1:]), homogeneous(d2[1:])

        # Triangulte points
        X, P = triangulate_from_E(E,K1,K2,x1,x2)
        self.traj=np.vstack((d1[0],X[:-1]))

        # Assign the camera matrix for these two cameras
        self.cameras[t1].P = np.dot(K1,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
        self.cameras[t2].P = np.dot(K2,P)
        self.cameras[t1].decompose()
        self.cameras[t2].decompose()
        return x1,x2,d1,d2,X

class Camera:
    def __init__(self,**kwargs):
        self.P = kwargs.get('P')
        self.K = np.array([[1000, 0, 960],
                           [0, 1000, 540],
                           [0, 0, 1]], dtype=np.float32)
        self.distCoeffs = None
        self.R_camera2world = None
        self.R_camera0 = None
        self.t_camera2world = None
        self.t_camera0 = None
        self.T_camera2world = None
        self.T_camera0 = None
        self.position = kwargs.get('position')
        self.imagepoints = None
        self.imagepoints_err = None
        self.P = None
    def calculate_camera_transform(self, camera_target):
        # 相机主光轴的方向
        camera_axis = camera_target - self.position
        camera_axis = camera_axis/np.linalg.norm(camera_axis)  # 归一化向量
        camera_x_axis = np.cross(camera_axis, [0, 0, 1])  # 叉乘得到垂直于 z 轴和主光轴方向的向量
        camera_x_axis = camera_x_axis/ np.linalg.norm(camera_x_axis)
        rotation_matrix = np.column_stack((camera_x_axis, -np.cross(camera_x_axis, camera_axis), camera_axis))
        rotation_matrix = np.column_stack((rotation_matrix,self.position.T))
        rotation_matrix = np.vstack((rotation_matrix, np.array([0, 0, 0, 1])))
        return rotation_matrix
    #这个T是相机到世界坐标系的转移矩阵，我们需要算的是世界坐标，因此需要求取它的逆运算
    def UAV_position_world2camera(self,uav_points):
        UAVs_position_camera_uv=[]
        i=0
        for uav_position in uav_points.T:
            uav_position=np.vstack((uav_position.reshape(3,1),np.array([[1]])))
            UAV_position_camera=np.linalg.inv(self.T_camera2world).dot(uav_position)
            UAV_position_camera=UAV_position_camera[:3]
            UAV_position_camera_uv=self.K.dot(UAV_position_camera)
            UAV_position_camera_uv=UAV_position_camera_uv/UAV_position_camera_uv[2]
            UAV_position_camera_uv=UAV_position_camera_uv[:2]
            UAVs_position_camera_uv.append(np.hstack((np.array([i]), UAV_position_camera_uv.reshape(2))))
            i=i+1
        UAVs_position_camera_uv=np.array(UAVs_position_camera_uv)
        return UAVs_position_camera_uv
    def decompose(self):
        M = self.P[:,:3]
        R,K = np.linalg.qr(np.linalg.inv(M))
        R = np.linalg.inv(R)
        K = np.linalg.inv(K)

        T = np.diag(np.sign(np.diag(K)))
        if np.linalg.det(T) < 0:
            T[1,1] *= -1

        self.K = np.dot(K,T)
        self.R_camera0 = np.dot(T,R)
        self.t_camera0 = np.dot(np.linalg.inv(self.K),self.P[:,3])
        self.T_camera0 = np.vstack((np.column_stack((self.R_camera0,self.t_camera0)), np.array([0, 0, 0, 1])))
        self.K /= self.K[-1,-1]
        return self.K, self.R_camera0, self.t_camera0 ,self.T_camera0
    def compose(self):
        self.P = np.dot(self.K,np.hstack((self.R_camera0,self.t_camera0.reshape((-1,1)))))
        
def create_scene(numCam,camera_toward_height,camera_radius,camera_height,pathway,interp_points_num):
    flight = Scene()
    flight.numCam=numCam
    # Define the parameter t for the points and interpolated points
    t = np.arange(len(pathway))
    t_interp = np.linspace(0, t[-1], interp_points_num)
    # Create and apply the cubic splines for each dimension
    splines = [CubicSpline(t, pathway[:, i])(t_interp) for i in range(pathway.shape[1])]
    # Combine the interpolated points into a single array
    interpolated_points = np.vstack(splines)
    flight.points=interpolated_points
    

    camera_positions = flight.create_camera_points(numCam, camera_radius, camera_height).T
    for i in range(numCam):
        camera=Camera(position=camera_positions[i])
        camera.T_camera2world=camera.calculate_camera_transform(np.array([0, 0, camera_toward_height]))
        camera.R_camera2world=camera.T_camera2world[:3,:3]
        camera.t_camera2world=camera.T_camera2world[:3,-1]
        camera.imagepoints=camera.UAV_position_world2camera(flight.points)
        flight.addCamera(camera)
    flight.sequence=np.arange(flight.numCam)
    return flight