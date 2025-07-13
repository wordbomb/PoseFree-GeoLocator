import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2
import json
import math
import transformation

def affine_matrix_from_points(v0, v1):
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        print(ndims,v0.shape[1],v0.shape,v1.shape)
        raise ValueError('input arrays are of wrong shape or type')

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

    v0 *= v0
    v1 *= v1
    M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M

def calculate_rmse(err_arr):
    # 计算每个点的欧几里得距离的平方
    squared_distances = np.sum(err_arr**2, axis=0)
    
    # mse
    mse = np.mean(squared_distances)
    
    # rmse
    rmse = np.sqrt(mse)

    rmse_x = np.sqrt(np.mean(err_arr[0]**2))
    rmse_y = np.sqrt(np.mean(err_arr[1]**2))
    rmse_z = np.sqrt(np.mean(err_arr[2]**2))
    
    return rmse,rmse_x,rmse_y,rmse_z

def reprojection_error(x,x_p):
    return np.sqrt((x[0]-x_p[0])**2 + (x[1]-x_p[1])**2)

def find_intervals(x,gap=5,idx=False):
    '''
    Given indices of detections, return a matrix that contains the start and the end of each
    continues part.
    
    Input indices must be in ascending order. 
    
    The gap defines the maximal interruption, with which it's still considered as continues. 
    '''

    assert len(x.shape)==1 and (x[1:]>x[:-1]).all(), 'Input must be an ascending 1D-array'

    # Compute start and end
    x_s, x_e = np.append(-np.inf,x), np.append(x,np.inf)
    start = x_s[1:] - x_s[:-1] >= gap
    end = x_e[:-1] - x_e[1:] <= -gap
    interval = np.array([x[start],x[end]])
    int_idx = np.array([np.where(start)[0],np.where(end)[0]])

    # Remove intervals that are too short
    mask = interval[1]-interval[0] >= gap
    interval = interval[:,mask]
    int_idx = int_idx[:,mask]

    assert (interval[0,1:]>interval[1,:-1]).all()

    if idx:
        return interval, int_idx
    else:
        return interval

def sampling(x,interval,belong=False):
    '''
    Sample points from the input which are inside the given intervals
    '''
    
    # Define timestamps
    if len(x.shape)==1:
        timestamp = x
    elif len(x.shape)==2:
        timestamp = x[0]

    # Sample points from each interval
    idx_ts = np.zeros_like(timestamp, dtype=int)
    for i in range(interval.shape[1]):
        mask = np.logical_xor(timestamp-interval[0,i] >= 0, timestamp-interval[1,i] >= 0)
        idx_ts[mask] = i+1

    if not belong:
        idx_ts = idx_ts.astype(bool)
    if len(x.shape)==1:
        return x[idx_ts.astype(bool)], idx_ts
    elif len(x.shape)==2:
        return x[:,idx_ts.astype(bool)], idx_ts
    else:
        raise Exception('The shape of input is wrong')

def homogeneous(x):
    return np.vstack((x,np.ones(x.shape[1])))

def match_overlap(x,y):
    '''
    Given two inputs in the same timeline (global), return the parts of them which are temporally overlapped

    Important: it's assumed that x has a higher frequency (fps) so that points are interpolated in y
    '''
    interval = find_intervals(y[0])
    x_s, _ = sampling(x, interval)

    tck, u = interpolate.splprep(y[1:],u=y[0],s=0,k=3)
    y_s = np.asarray(interpolate.splev(x_s[0],tck))
    y_s = np.vstack((x_s[0],y_s))

    return x_s, y_s

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
    x1n = np.dot(np.linalg.inv(K1),x1)
    x2n = np.dot(np.linalg.inv(K2),x2)
    infront_max = 0
    Rt = compute_Rt_from_E(E)
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    for i in range(4):
        P2_temp = Rt[i]
        X = triangulate_matlab(x1n,x2n,P1,P2_temp)
        d1 = np.dot(P1,X)[2]
        d2 = np.dot(P2_temp,X)[2]

        if sum(d1>0)+sum(d2>0) > infront_max:
            infront_max = sum(d1>0)+sum(d2>0)
            P2 = P2_temp
    X = triangulate_matlab(x1n,x2n,P1,P2) 
    return X[:,:], P2

def computeFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransac_thresh=15):
    '''
    Function:
            compute fundamental matrix given correspondences (at least 8)
    Input:
            pts1, pts2 = list of pixel coordinates of corresponding features
            method = cv2.FM_RANSAC: Using RANSAC algorithm
            error = reprojection threshold that describes maximal distance from a 
                    point to a epipolar line
    Output:
            F = Fundamental matrix with size 3*3
            mask = index for inlier correspondences (optional)
    '''

    pts1 = pts1[:2].T
    pts2 = pts2[:2].T

    F, mask = cv2.findFundamentalMat(pts1,pts2,method,ransac_thresh)
    return F, mask.reshape(-1,)



class Scene:
    def __init__(self):
        self.numCam = 0
        self.cameras = []
        self.numpoints = 0
        self.points = []
        self.sequence=None
        self.detections = []
        self.detections_global = []
        self.ref_cam = 0
        self.alpha = []
        self.beta = []
        self.spline = {'tck':[], 'int':[]}
        self.gt_ori=None

    def addDetection(self,*detection):
        for i in detection:
            self.detections.append(i)

    def create_camera_points(self,num_points, radius, height):
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.full_like(x, height)
        return np.array([x, y, z])

    def addCamera(self,*camera):
        for i in camera:
            assert type(i) is Camera, "camera is not an instance of Camera"
            self.cameras.append(i)

    def get_camera_pose(self, cam_id, error=8):
        '''
        Get the absolute pose of a camera by solving the PnP problem.

        Take care with DISTORSION model!
        '''
        
        tck, interval = self.spline['tck'], self.spline['int']
        self.detection_to_global(cam_id)

        _, idx = sampling(self.detections_global[cam_id], interval, belong=True)
        detect = np.empty([3,0])
        point_3D = np.empty([3,0])
        for i in range(interval.shape[1]):
            detect_part = self.detections_global[cam_id][:,idx==i+1]
            if detect_part.size:
                detect = np.hstack((detect,detect_part))
                point_3D = np.hstack((point_3D, np.asarray(interpolate.splev(detect_part[0], tck[i]))))

        # PnP solution from OpenCV
        N = point_3D.shape[1]
        objectPoints = np.ascontiguousarray(point_3D.T).reshape((N,1,3))
        imagePoints  = np.ascontiguousarray(detect[1:].T).reshape((N,1,2))
        distCoeffs = self.cameras[cam_id].d
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, self.cameras[cam_id].K, distCoeffs, reprojectionError=error)

        self.cameras[cam_id].R = cv2.Rodrigues(rvec)[0]
        self.cameras[cam_id].t = tvec.reshape(-1,)
        self.cameras[cam_id].compose()

    def triangulate(self, cam_id, cams, thres=20, refit=True):
        '''
        Triangulate new points to the existing 3D spline and optionally refit it
        cam_id is the new camera
        cams must be an iterable that contains cameras that have been processed to build the 3D spline
        '''
        assert self.cameras[cam_id].P is not None, 'The camera pose must be computed first'
        tck, interval = self.spline['tck'], self.spline['int']
        self.detection_to_global(cam_id)

        # Find detections from this camera that haven't been triangulated yet
        _, idx_ex = sampling(self.detections_global[cam_id], interval)
        detect_new = self.detections_global[cam_id][:, np.logical_not(idx_ex)]

        # Matching these detections with detections from previous cameras and triangulate them
        X_new = np.empty([4,0])
        for i in cams:
            self.detection_to_global(i)
            detect_ex = self.detections_global[i]

            # Detections of previous cameras are interpolated, no matter the fps
            try:
                x1, x2 = match_overlap(detect_new, detect_ex)
            except:
                continue
            else:
                P1, P2 = self.cameras[cam_id].P, self.cameras[i].P
                X_i = triangulate_matlab(x1[1:], x2[1:], P1, P2)
                X_i = np.vstack((x1[0], X_i[:-1]))

                # Check reprojection error directly after triangulation, preserve those with small error
                
                err_1 = reprojection_error(x1[1:], self.cameras[cam_id].projectPoint(X_i[1:]))
                err_2 = reprojection_error(x2[1:], self.cameras[i].projectPoint(X_i[1:]))
                mask = np.logical_and(err_1<thres, err_2<thres)
                X_i = X_i[:, mask]
                    
                X_new = np.hstack((X_new, X_i))
                
        # Add these points to the discrete 3D trajectory
        self.spline_to_traj(sampling_rate=0.5)
        self.traj = np.hstack((self.traj, X_new))
        _, idx = np.unique(self.traj[0], return_index=True)
        self.traj = self.traj[:, idx]

        # refit the 3D spline if wanted
        if refit:
            self.traj_to_spline()

        return X_new
    
    def init_traj(self):
        '''
        Select the first two cams in the sequence, compute fundamental matrix, triangulate points
        '''
        t1, t2 = self.sequence[0], self.sequence[1]
        K1, K2 = self.cameras[t1].K, self.cameras[t2].K

        if self.cameras[t1].fps > self.cameras[t2].fps:
            d1, d2 = match_overlap(self.detections_global[t1], self.detections_global[t2])
        else:
            d2, d1 = match_overlap(self.detections_global[t2], self.detections_global[t1])

        F, inlier = computeFundamentalMat(d1[1:], d2[1:])
        E = K2.T @ F @ K1
        inlier = inlier.reshape(-1)

        pts1 = d1[1:, inlier == 1].T.reshape(1, -1, 2)
        pts2 = d2[1:, inlier == 1].T.reshape(1, -1, 2)

        m1, m2 = cv2.correctMatches(F, pts1, pts2)
        x1 = homogeneous(m1.reshape(-1, 2).T)
        x2 = homogeneous(m2.reshape(-1, 2).T)

        mask = ~np.isnan(x1[0])
        x1, x2 = x1[:, mask], x2[:, mask]
        timestamps = d1[0][inlier == 1][mask]

        X, P = triangulate_from_E(E, K1, K2, x1, x2)
        self.traj = np.vstack((timestamps, X[:-1]))

        self.cameras[t1].P = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.cameras[t2].P = K2 @ P
        self.cameras[t1].decompose()
        self.cameras[t2].decompose()

    def init_alpha(self):
        '''Initialize alpha for each camera based on the ratio of fps'''
        self.alpha = np.ones(self.numCam)
        fps_ref = self.cameras[self.ref_cam].fps
        for i in range(self.numCam):
            self.alpha[i] = fps_ref / self.cameras[i].fps

    def time_shift(self, iter=False):
        '''
        This function computes relative time shifts of each camera to the ref camera using the given corresponding frame numbers

        If the given frame indices are precise, then the time shifts are directly transformed from them.
        '''
        assert len(self.cf)==self.numCam, 'The number of frame indices should equal to the number of cameras'

        self.beta = self.cf[self.ref_cam] - self.alpha*self.cf

    def detection_to_global(self,*cam):
        '''
        Convert frame indices of raw detections into the global timeline.

        Input is an iterable that specifies which detection(s) to compute.

        If no input, all detections will be converted.
        '''

        assert len(self.alpha)==self.numCam and len(self.beta)==self.numCam, 'The Number of alpha and beta is wrong'

        if len(cam):
            cams = cam
            if type(cams[0]) != int:
                cams = cams[0]
        else:
            cams = range(self.numCam)
            self.detections_global = [[] for i in cams]

        for i in cams:
            timestamp = self.alpha[i] * self.detections[i][0] + self.beta[i]
            detect = self.cameras[i].undist_point(self.detections[i][1:])
            
            self.detections_global[i] = np.vstack((timestamp, detect))
            
    def traj_to_spline(self,smooth_factor=[20,10]):
        '''
        Convert discrete 3D trajectory into spline representation

        A single spline is built for each interval
        '''
        assert len(smooth_factor)==2, 'Smoothness should be defined by two parameters (min, max)'
        timestamp = self.traj[0]
        interval, idx = find_intervals(timestamp,idx=True)
        tck = [None] * interval.shape[1]
        for i in range(interval.shape[1]):
            part = self.traj[:,idx[0,i]:idx[1,i]+1]
            measure = part[0,-1] - part[0,0]
            s = (1e-3)**2*measure
            thres_min, thres_max = min(smooth_factor), max(smooth_factor)
            prev = 0
            t = 0
            try:
                while True:
                    tck[i], u = interpolate.splprep(part[1:],u=part[0],s=s,k=3)
                    numKnot = len(tck[i][0])-4
                    if numKnot == prev and numKnot==4 and t==2:
                        break
                    else:
                        prev = numKnot
                    if measure/numKnot > thres_max:
                        s /= 1.5
                        t = 1
                    elif measure/numKnot < thres_min:
                        s *= 2
                        t = 2
                    else:
                        break
                dist = np.sum(np.sqrt(np.sum((part[1:,1:]-part[1:,:-1])**2,axis=0)))
            except:
                tck[i], u = interpolate.splprep(part[1:],u=part[0],s=s,k=1)
        self.spline['tck'], self.spline['int'] = tck, interval

    
    def spline_to_traj(self,sampling_rate=1,t=None):

        tck, interval = self.spline['tck'], self.spline['int']
        self.traj = np.empty([4,0])
        if t is not None:
            assert len(t.shape)==1, 'Input timestamps must be a 1D array'
            timestamp = t
        else:
            timestamp = np.arange(interval[0,0], interval[1,-1], sampling_rate)

        for i in range(interval.shape[1]):
            t_part = timestamp[np.logical_and(timestamp>=interval[0,i], timestamp<=interval[1,i])]
            try:
                traj_part = np.asarray(interpolate.splev(t_part, tck[i]))
            except:
                continue
            self.traj = np.hstack((self.traj, np.vstack((t_part,traj_part))))
        assert (self.traj[0,1:] >= self.traj[0,:-1]).all()
    
    def getSimilarityTransformationMatrix0w(self):
        p2=[]
        for camera in self.cameras:
            p2.append(camera.location)
        p2=np.array(p2).T
        p1=[]
        for i in range(self.numCam):
            p1.append(-np.linalg.inv(self.cameras[i].R).dot(np.array(self.cameras[i].t)))
        p1=np.array(p1).T
        print(p1)
        print(p2)
        M = affine_matrix_from_points(p1, p2)

        p1_transformed = (M @ np.vstack((p1, np.ones(p1.shape[1]))))[:3]
        rmse = np.sqrt(np.mean(np.sum((p1_transformed - p2)**2, axis=0)))
        print(f"RMSE = {rmse:.4f}")

        return M



    def pre_process_and_find_best_match(self, M):
    # Pre-processing
        alpha = self.cameras[self.ref_cam].fps / self.gt['frequency']
        self.spline_to_traj(sampling_rate=alpha)
        reconst = self.traj
        t0 = reconst[0, 0]
        reconst = np.vstack(((reconst[0] - t0) / alpha, reconst[1:]))
        gt = np.vstack((np.arange(len(self.gt_ori[0])), self.gt_ori))
    
        if int(gt[0, -1] - reconst[0, -1] / 2) < 0:
            raise Exception('Ground truth too short!')

        thres = int(reconst[0, -1] / 2)
        error_min = np.inf
        best_offset = None

        for i in range(-thres, int(gt[0, -1] - thres)):
            reconst_i = np.vstack((reconst[0] + i, reconst[1:]))
            p1, p2 = match_overlap(reconst_i, gt)
            tran = np.dot(M, homogeneous(p1[1:])) / homogeneous(p1[1:])[-1]
            error = np.mean(np.sqrt((p2[1] - tran[0])**2 + (p2[2] - tran[1])**2 + (p2[3] - tran[2])**2))

            if error < error_min:
                error_min = error
                best_offset = i

        beta = t0 - alpha * best_offset
        t_gt = alpha * np.arange(self.gt_ori.shape[1]) + beta
        return t_gt


    def align_gt(self, f_gt, gt_path):
        if not len(gt_path):
            print('No ground truth data provided\n')
            return
        else:
            try:
                gt_ori = np.loadtxt(gt_path)
            except:
                print('Ground truth not correctly loaded')
                return
        if gt_ori.shape[0] == 3 or gt_ori.shape[0] == 4:
            pass
        elif gt_ori.shape[1] == 3 or gt_ori.shape[1] == 4:
            gt_ori = gt_ori.T
        else:
            raise Exception('Ground truth data have an invalid shape')

        # Pre-processing
        f_reconst = self.cameras[self.ref_cam].fps
        alpha = f_reconst/f_gt


        self.spline_to_traj(sampling_rate=alpha)
        reconst = self.traj
        t0 = reconst[0,0]
        reconst = np.vstack(((reconst[0]-t0)/alpha,reconst[1:]))
        if gt_ori.shape[0] == 3:
            gt = np.vstack((np.arange(len(gt_ori[0])),gt_ori))
        else:
            gt = np.vstack((gt_ori[0]-gt_ori[0,0],gt_ori[1:]))

        # Coarse search
        thres = int(reconst[0,-1] / 2)
        if int(gt[0,-1]-thres) < 0:
            raise Exception('Ground truth too short!')

        error_min = np.inf

        for i in range(-thres, int(gt[0,-1]-thres)):
            reconst_i = np.vstack((reconst[0]+i,reconst[1:]))
            p1, p2 = match_overlap(reconst_i, gt)
            M = transformation.affine_matrix_from_points(p1[1:], p2[1:], shear=False, scale=True)
            tran = np.dot(M, homogeneous(p1[1:]))
            tran /= tran[-1]
            error_all = np.sqrt((p2[1]-tran[0])**2 + (p2[2]-tran[1])**2 + (p2[3]-tran[2])**2)
            error = np.mean(error_all)
            if error < error_min:
                error_min = error
                j = i
                self.M=M


def match_overlap(x,y):
    '''
    Given two inputs in the same timeline (global), return the parts of them which are temporally overlapped

    Important: it's assumed that x has a higher frequency (fps) so that points are interpolated in y
    '''
    interval = find_intervals(y[0])
    x_s, _ = sampling(x, interval)


    tck, u = interpolate.splprep(y[1:],u=y[0],s=0,k=3)
    y_s = np.asarray(interpolate.splev(x_s[0],tck))
    y_s = np.vstack((x_s[0],y_s))

    assert (x_s[0] == y_s[0]).all(), 'Both outputs should have the same timestamps'

    return x_s, y_s


class Camera:
    def __init__(self,**kwargs):
        self.P = kwargs.get('P')
        self.K = kwargs.get('K')
        self.d = kwargs.get('d')
        self.R = kwargs.get('R')
        self.t = kwargs.get('t')
        self.location = kwargs.get('location')
        self.distCoeffs = None
        self.R_camera2world = None
        self.R_camera0 = None
        self.t_camera2world = None
        self.T_camera2world = None
        self.P = None
        self.fps = kwargs.get('fps')
        self.resolution = kwargs.get('resolution')

    def undist_point(self,points):
        
        assert points.shape[0]==2, 'Input must be a 2D array'

        num = points.shape[1]

        src = np.ascontiguousarray(points.T).reshape((num,1,2))
        dst = cv2.undistortPoints(src, self.K, self.d)
        dst_unnorm = np.dot(self.K, homogeneous(dst.reshape((num,2)).T))

        return dst_unnorm[:2]
        
    def decompose(self):
        M = self.P[:,:3]
        R,K = np.linalg.qr(np.linalg.inv(M))
        R = np.linalg.inv(R)
        K = np.linalg.inv(K)

        T = np.diag(np.sign(np.diag(K)))
        if np.linalg.det(T) < 0:
            T[1,1] *= -1

        self.K = np.dot(K,T)
        self.R = np.dot(T,R)
        self.t = np.dot(np.linalg.inv(self.K),self.P[:,3])
        self.K /= self.K[-1,-1]
        return self.K, self.R, self.t
    
    def compose(self):
        self.P = np.dot(self.K,np.hstack((self.R,self.t.reshape((-1,1)))))

    def projectPoint(self,X):
        assert self.P is not None, 'The projection matrix P has not been calculated yet'
        if X.shape[0] == 3:
            X = homogeneous(X)
        x = np.dot(self.P,X)
        x /= x[2]
        return x

def create_scene(path_input):
    flight = Scene()

    with open(path_input, 'r') as file:
        config = json.load(file)
    
    # Load detections
    path_detect = config['detection_inputs']['path_detections']
    flight.numCam = len(path_detect)
    for i in path_detect:
        detect = np.loadtxt(i,usecols=(0,1,2)).T

        frames = detect[0].astype(int)
        coords = detect[1:, :]

        diffs = np.diff(frames)
        split_idx = np.where(diffs != 1)[0] + 1
        split_points = np.split(np.arange(len(frames)), split_idx)

        valid_indices = []
        for seg in split_points:
            if len(seg) >= 15:
                valid_indices.extend(seg)

        if len(valid_indices) == 0:
            continue
        detect = detect[:, valid_indices]
        flight.addDetection(detect)

    # Load cameras
    path_cam = config['detection_inputs']['path_cameras']
    i=0
    for i in range(len(path_cam)):
        path=path_cam[i]
        try:
            with open(path, 'r') as file:
                cam = json.load(file)
        except:
            raise Exception('Wrong input of camera',path)

        if len(cam['distCoeff']) == 4:
            cam['distCoeff'].append(0)
        flight.addCamera(Camera(K=np.asfarray(cam['K-matrix']), d=np.asfarray(cam['distCoeff']),fps=cam['fps'], resolution=cam['resolution'],location=cam['location']))
    
    #  Load sequence
    flight.sequence = [i for i in range(len(path_cam))]

    # Load corresponding frames
    flight.cf = np.asfarray(config['detection_inputs']['corresponding_frames'])

    # Load ground truth setting
    flight.gt = config['detection_inputs']['ground_truth']
    flight.gt_ori=np.loadtxt(flight.gt['filepath']).T


    return flight
