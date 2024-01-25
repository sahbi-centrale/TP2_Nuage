#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from visu import show_ICP

import sys


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#




def best_rigid_transform(data: np.ndarray, ref: np.ndarray):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
         dat = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    mean_data = np.mean(data, axis=1, keepdims = True)
    mean_ref = np.mean(ref, axis=1, keepdims=True)
    new_data = data - mean_data
    new_ref = ref - mean_ref
    H = new_data.dot(new_ref.T)
    u, _, vt = np.linalg.svd(H)
    R = (vt.T).dot(u.T)
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = (vt.T).dot(u.T)
    T = mean_ref - R.dot(mean_data)
    return R , T 




def icp_point_to_point(data, ref, max_iter, RMS_threshold):
  
    data_aligned = np.copy(data)
    leaf_size = 20
    tree = KDTree(ref.T, leaf_size=leaf_size)

    # Initialize lists to store transformations and errors
    R_list, T_list, neighbors_list, RMS_list = [], [], [], []
    R_prev = np.eye(ref.shape[0])
    T_prev = np.zeros((ref.shape[0], 1))
    
    rms_current = np.inf

    for it in range(max_iter):
        # Break loop if RMS error is below threshold
        if rms_current < RMS_threshold:
            break

        # Find nearest neighbors in ref for each point in data_aligned
        ref_nearest_index = tree.query(data_aligned.T, k=1, return_distance=False)[:, 0]
        ref_nearest = ref[:, ref_nearest_index]

        # Store the indices of nearest neighbors
        neighbors_list.append(ref_nearest_index.copy())

        # Compute the best rigid transformation
        R, T = best_rigid_transform(data_aligned, ref_nearest)

        # Update data_aligned with the current transformation
        data_aligned = np.dot(R, data_aligned) + T

        # Update the transformation matrices
        T = np.dot(R, T_prev) + T
        R = np.dot(R, R_prev)

        # Store the current transformation
        R_prev, T_prev = R, T
        R_list.append(R.copy())
        T_list.append(T.copy())

        # Compute and store current RMS error
        rms_current = np.sqrt(np.mean(np.linalg.norm(data_aligned - ref_nearest, axis=0)))
        RMS_list.append(rms_current)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list






def icp_point_to_point_fast(data, ref, max_iter, RMS_threshold, sampling_limit):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        dat = (d x N_dat) matrix where "N_dat" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
        Tree = pre-buit KDTree (leaf size=150 is a good value)
        true_rmse = Full RMSE computation
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
        total = total time spent in the function (for benchmarking)

    '''

    # Variable for aligned data
    data_aligned = np.copy(data)
    leaf_size = 20
    tree = KDTree(ref.T, leaf_size=leaf_size)

    R_list, T_list, neighbors_list, RMS_list = [], [], [], []
    R_prev = np.eye(ref.shape[0])
    T_prev = np.zeros((ref.shape[0], 1))
    
    rms_current = np.inf
    total = 0.
    for it in range(max_iter):
        if rms_current < RMS_threshold:
            break
        selection_indexes = np.random.choice(data_aligned.shape[-1], size=sampling_limit, replace=False)
        data_selection = data_aligned[:, selection_indexes]
        ref_nearest_index = tree.query(data_selection.T, k=1, return_distance=False)[:, 0]
        ref_nearest = ref[:, ref_nearest_index]
        neighbors_list.append(ref_nearest_index.copy())
        R, T = best_rigid_transform(data_selection, ref_nearest)
        data_aligned = np.dot(R, data_aligned) + T
        T = np.dot(R, T_prev) + T
        R = np.dot(R, R_prev)
        R_prev = R
        T_prev = T
        R_list.append(R.copy())
        T_list.append(T.copy())
        rms = np.sqrt(np.linalg.norm(data_aligned[:, selection_indexes] - ref_nearest, axis=0).mean())
        RMS_list.append(rms)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list










#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #
    


    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = 'bunny_original.ply'
        bunny_r_path = 'bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        print("before")
        R, T = best_rigid_transform(bunny_r, bunny_o)
        print("after")

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if  False:

        # Cloud paths
        ref2D_path = 'ref2D.ply'
        data2D_path = 'data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)
        
        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
        

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = 'bunny_original.ply'
        bunny_p_path = 'bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        
        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
      # If statement to skip this part if wanted

    if True:

        # Cloud paths
        Notre_Dame_Des_Champs_1_path = 'Notre_Dame_Des_Champs_1.ply'
        Notre_Dame_Des_Champs_2_path = 'Notre_Dame_Des_Champs_2.ply'
        
        # Load clouds
        Notre_Dame_Des_Champs_1_ply = read_ply(Notre_Dame_Des_Champs_1_path)
        Notre_Dame_Des_Champs_2_ply = read_ply(Notre_Dame_Des_Champs_2_path)
        Notre_Dame_Des_Champs_1 = np.vstack((Notre_Dame_Des_Champs_1_ply['x'], Notre_Dame_Des_Champs_1_ply['y'], Notre_Dame_Des_Champs_1_ply['z']))
        Notre_Dame_Des_Champs_2 = np.vstack((Notre_Dame_Des_Champs_2_ply['x'], Notre_Dame_Des_Champs_2_ply['y'], Notre_Dame_Des_Champs_2_ply['z']))

        # Apply ICP
        Notre_Dame_Des_Champs_2_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(Notre_Dame_Des_Champs_2, Notre_Dame_Des_Champs_1, 25, 1e-4 , 10000)
        x = np.linspace(0,25,25)
        
        # Plot RMS
        print(len(RMS_list))
        plt.plot(x ,RMS_list)
        plt.show()

