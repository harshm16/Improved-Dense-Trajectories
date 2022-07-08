import os
import numpy as np
from cffi import FFI
import cv2

from scipy.cluster.vq import kmeans,kmeans2,vq

# For trajectory storage
import h5py
import uuid
import densetrack
import re

# OpenBLAS affects CPU affinity
os.sched_setaffinity(0,range(os.cpu_count()))
def setaff():
    os.sched_setaffinity(0,range(os.cpu_count()))
    
# for Multi-threading
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(5, setaff)


def read_video(file):
    """
    Reads the frames from a video file.
    :param file: the filename in the data directory.
    :type file: String
    :return: the gray video as a numpy array.
    :rtype: array_like, shape (D, H, W) where D is the number of frames,
            H and W are the resolution.
    """
    vidcap = cv2.VideoCapture(file)

    # print("vidcap",vidcap.shape)

    video = []
    success, image = vidcap.read()
    while success:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        video.append(gray)
        success, image = vidcap.read()

    return np.array(video)


def read_sparse_video(data):
    """
    Reads the frames from a video file.
    :param file: the filename in the data directory.
    :type file: String
    :return: the gray video as a numpy array.
    :rtype: array_like, shape (D, H, W) where D is the number of frames,
            H and W are the resolution.
    """
    # vidcap = cv2.VideoCapture(file)

    # print("vidcap",vidcap.shape)

    video = []
    # success, image = vidcap.read()

    for i in range(data.shape[0]):
        # print("data[i]",data[i].shape)
        # print(sd)
        gray = cv2.cvtColor(data[i], cv2.COLOR_BGR2GRAY)
        video.append(gray)
        # success, image = vidcap.read()

    return np.array(video)


# =======================================================================
def filter_trajs_displacement(trajs):
    #print(trajs.shape)
    num_trajs = len(trajs)
    disp_stor = np.empty((num_trajs,), np.float32)
    for ii in range(num_trajs):
        disp_stor[ii] = np.sum(np.sqrt(np.sum((trajs[ii,1:,:]-trajs[ii,0:-1,:])**2,1)))
    # Remove trajectories that have very low displacement
    good_trajs = np.flatnonzero(disp_stor>5)
    
    return good_trajs
    

# =======================================================================
def filter_trajs_kmeans(trajs, dec_frames, num_centroids):
    num_trajs = len(trajs)
    traj_vec_stor = np.empty((num_trajs, (dec_frames-1)*2), np.float32)
    disp_stor = np.empty((num_trajs,), np.float32)
        
    for ii in range(num_trajs):
        traj = trajs[ii,0:dec_frames,:]  # n-by-2
        traj_vec_stor[ii,:] = (traj[1:,:] - traj[0,:]).flatten() # substract start point        
        disp_stor[ii] = np.sum(np.sqrt(np.sum((traj[1:,:]-traj[0:-1,:])**2,1)))
    # Remove trajectories that have very low displacement
    good_trajs = np.flatnonzero(disp_stor>0.4)
    traj_vec_stor = traj_vec_stor[good_trajs,:]
    
    if traj_vec_stor.shape[0] < num_centroids: # too few points
        print("kmeans: TOO FEW USABLE KEYPOINTS")
        return good_trajs[np.arange(0,traj_vec_stor.shape[0]-1)] # try to use all of them
        
    # k-means on vectors
    #num_centroids = 10
    #centroids,_ = kmeans(traj_vec_stor,k_or_guess=num_centroids, iter=100)
    centroids,_ = kmeans(traj_vec_stor,num_centroids, iter=100)
    
    # Find the nearest vectors to centroids
    rep = np.argmin(np.sum((traj_vec_stor[:,np.newaxis,:]-centroids[:,:])**2,2),0) # 10-dim
    
    rep = good_trajs[rep]
    
    return rep # return the index of K most representative trajectories
    
# ==========================================================================

CLIP_LENGTH = 16

# Load video...
#for vid_idx in range(NUM_VIDEOS):
def worker(idx):
    # print("Processing %d/%d" % (idx, len(job_stor)))
    video_path, length, offset,video_file_path,video_target_path = job_stor[idx]
    
    # print("video_path",video_path)
    # print("length",length)
    # print("offset",offset)
    #start_frame = random.randint(0,length-CLIP_LENGTH+1-1)
    start_frame = offset
    # sparse_video = []
    
    # out = cv2.VideoWriter(video_target_path+ '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, (256,192)) 
    

    for fram_no in range(CLIP_LENGTH):
        # print("video_target_path",video_target_path)
        # while start_frame+fram_no < 2:
            # print("video_path",video_path+'/'+str(start_frame+fram_no)+'.jpg')
            
            frame = cv2.imread(video_path+'/'+str(start_frame+fram_no)+'.jpg')
            
            img = cv2.resize(frame, (256,192), interpolation=cv2.INTER_AREA)

            # out.write(img)

            if fram_no == 0:
                height = img.shape[0]
                width = img.shape[1]

                vid_seq = np.empty([CLIP_LENGTH,height,width,3], dtype=np.uint8)

                
            # sparse_video.append(img)
            vid_seq[fram_no,:,:,:] = img[:,:,:]
    
    # print(out)
   
    # out.release()
    
    
    # print("sparse_video",np.array(sparse_video).shape)
    # print("vid_seq",vid_seq.shape)
    # print(sd)

    # print("video_target_path+ '.avi'",video_target_path+ '.avi')
    # sparse_video = read_video(video_target_path+ '.avi')



    # print("sparse_video",sparse_video)  

    # out.write(sparse_video)

    # print(sd)

    # print("video_path",video_file_path)
    
    sparse_video = read_sparse_video(vid_seq)

    # print("video",video.shape)
   
    tracks = densetrack.densetrack(sparse_video, adjust_camera=True)
    # print("tracks",tracks)

    # print(outy)
    np.save(video_target_path, tracks)

    # print(type(tracks))

    trajectory = []

    for each_trajectory in tracks:
        trajectory.append(each_trajectory[10]) 

    np_trajectory = np.array(trajectory)
    # print("np_trajectory",np_trajectory)

    # print("traj", np.array(trajectory))

    # print(sd)
    # print('------------------------------------------')
    # print(f'Running: {file} of shape {video.shape}')

    
    # print("video_target_path",video_target_path)
    # np.save(video_target_path, tracks)

    # print("tracks",tracks)
      

    # print("vid_seq", vid_seq.ctypes.data)
    # # Calculate trajectories
    # vid_seq_cptr = ffi.cast("char *", vid_seq.ctypes.data)
    # traj_ret = ffi.new("Ret[]", 1)
    # # note that a lot more parameters are hard-coded in DenseTrackStab.cpp due to laziness.
    # libtest.main_like(vid_seq_cptr, width, height, CLIP_LENGTH, traj_ret)
    # #print(traj_ret[0].traj_length)
    # #print(traj_ret[0].num_trajs)
    # #print(traj_ret[0].out_trajs[0])
    # trajs = np.frombuffer(ffi.buffer(traj_ret[0].out_trajs, traj_ret[0].traj_length*traj_ret[0].num_trajs*2*4), dtype=np.float32)
    # trajs = np.resize(trajs,[traj_ret[0].num_trajs,traj_ret[0].traj_length,2])
    # #print(trajs.shape)
    # libtest.free_mem()

    # print(len(vid_seq))


  
    #filtered_trajs = filter_trajs_kmeans(trajs, 15, 10)
    filtered_trajs = filter_trajs_displacement(np_trajectory)

    # print("filtered_trajs",filtered_trajs)
    # del trajectory
    # del tracks

    # f = open("check.txt", "a")
    if len(filtered_trajs) == 0:
        # f.write(video_path + "\n")
        print('No Trajectory detected!!!')
        
    else:

        # print("DETCTED filtered_trajs.size",filtered_trajs.size)
        # f2 = open("check_true.txt", "a")
        # f2.write(video_path + "\n")



        # Write result to HDF5
        # %06d_%04d_%04d_uuid1(startFrame, trajLen, trajCount)
        h5_ucf_bc_traj = h5_ucf_bc.require_dataset('%06d_%04d_%04d_%s' % (start_frame+1, CLIP_LENGTH, filtered_trajs.size, uuid.uuid1()), shape=(filtered_trajs.size, CLIP_LENGTH, 2), dtype='float32')
        h5_ucf_bc_traj[:,:,:] = np_trajectory[filtered_trajs[:],:,:]
        h5_ucf_bc_traj.attrs['VidPath'] = video_path
        h5_ucf_bc_traj.attrs['StartFrame'] = start_frame
        h5_ucf_bc_traj.attrs['TrajLen'] = CLIP_LENGTH
        h5_ucf_bc_traj.attrs['TrajCount'] = filtered_trajs.size
        h5_ucf_bc_traj.attrs['VidResH'] = height
        h5_ucf_bc_traj.attrs['VidResW'] = width
        # f.flush()

    # print("done")
    
if __name__ == "__main__":
    # ========================================================================
    # Load UCF101 dataset
    DATASET_DIR = "/densetrack/data/cat_dog_seq"            # [EDIT ME!]
    # "/densetrack/data/UCF101seq"
    # "/densetrack/data/cat_dog_seq"# [EDIT ME!]
    
    VIDEO_DIR = "/densetrack/data/UCF-cat_dog"          # [EDIT ME!]
    # "/densetrack/data/UCF-101"
    # "/densetrack/data/UCF-cat_dog"

    TARGET_DIRECTORY = "/densetrack/data/cat_dog_dense"     # [EDIT ME!]
    # "/densetrack/data/cat_dog_dense"
    # Load split file:
    f = open('testlist_cat_dog.txt','r') # Sample: ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi # [EDIT ME!]
    job_stor = []
    for line in f:
        # print("line.split(),",len(line.split()))
        vid_name = line.split()[0]
        video_path = os.path.join(DATASET_DIR, vid_name)
        video_file_path = os.path.join(VIDEO_DIR, vid_name)

        target_path = os.path.join(TARGET_DIRECTORY, vid_name.split("/")[0])

        if os.path.isdir(target_path) == False:
            os.mkdir(target_path)

        video_target_path = os.path.join(target_path,vid_name.split(".avi")[0].split("/")[1])

        img_list = os.listdir(video_path)
        frame_count = 0
        for filename in img_list:
            frame_count = max(frame_count, int(filename.split('.')[0]))
        frame_count += 1
        # print("frame_count",CLIP_LENGTH)
        # print("frame_count - CLIP_LENGTH + 1",frame_count - CLIP_LENGTH + 1)
        for offset in range(0, frame_count - CLIP_LENGTH + 1, 8): # Stride = 8
            job_stor.append((video_path, frame_count, offset,video_file_path,video_target_path))
        
    f.close()
            
    print('Job count: {:d}'.format(len(job_stor))) # 13320, or 9537



    # Load HDF5 database......
    f = h5py.File("traj_stor_test_cat_dog.h5", 'a', libver='latest') # Supports Single-Write-Multiple-Read  # [EDIT ME!]
    h5_ucf = f.require_group("UCFTraj")
    #h5_kt_bv = h5_pa.require_group("by_video") # /KITTITraj/by_video/%04d(videoNo)/%06d_%04d_%04d_uuid1(startFrame, trajLen, trajCount)
    h5_ucf_bc = h5_ucf.require_group("by_clip") # /KITTITraj/by_clip/%02d_%04d_%04d_uuid1(video, startframe, len)
    f.swmr_mode = True

    pool.map(worker, range(len(job_stor)))
    # pool.map(worker, range(1))

