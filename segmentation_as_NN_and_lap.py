"""White Matter Tract Segmentation as Multiple Linear Assignment Problems
"""

import numpy as np
from dissimilarity import compute_dissimilarity, dissimilarity
from dipy.tracking.distances import bundles_distances_mam
from sklearn.neighbors import KDTree
from nibabel import trackvis
from dipy.tracking.utils import length
from dipy.viz import fvtk
import os
import vtk.util.colors as colors

try:
    from linear_assignment import LinearAssignment
except ImportError:
    print("WARNING: Cythonized LAPJV not available. Falling back to Python.")
    print("WARNING: See README.txt")
   
   
try:
    from joblib import Parallel, delayed
    joblib_available = True
except ImportError:
    joblib_available = False
   

def show_tract(segmented_tract, color):
   """Visualization of the segmented tract.
   """ 
   ren = fvtk.ren()           
   fvtk.add(ren, fvtk.line(segmented_tract.tolist(),
                           colors=color,
                           linewidth=2,
                           opacity=0.3))
   fvtk.show(ren)
   fvtk.clear(ren)



def ranking_schema(superset_estimated_target_tract_idx,superset_estimated_target_tract_cost):
    """ Rank all the extracted streamlines estimated by the LAP with different examples (superset)   
    accoring to the number of times it selected and the total cost
    """
    idxs = np.unique(superset_estimated_target_tract_idx)
    how_many_times_selected = np.array([(superset_estimated_target_tract_idx == idx).sum() for idx in idxs])
    how_much_cost = np.array([((superset_estimated_target_tract_idx == idx)*superset_estimated_target_tract_cost).sum() for idx in idxs])
    ranking = np.argsort(how_many_times_selected)[::-1]
    tmp = np.unique(how_many_times_selected)[::-1]
    for i in tmp:
        tmp1 = (how_many_times_selected == i)
        tmp2 = np.where(tmp1)[0]
        if tmp2.size > 1:
            tmp3 = np.argsort(how_much_cost[tmp2])
            ranking[how_many_times_selected[ranking]==i] = tmp2[tmp3]
 
    return idxs[ranking]

def load(T_filename, threshold_short_streamlines=10.0):
    """Load tractogram from TRK file and remove short streamlines with
    length below threshold.
    """
    print("Loading %s" % T_filename)
    T, hdr = trackvis.read(T_filename, as_generator=False)
    T = np.array([s[0] for s in T], dtype=np.object)
    print("%s: %s streamlines" % (T_filename, len(T)))

    # Removing short artifactual streamlines
    print("Removing (presumably artifactual) streamlines shorter than %s" % threshold_short_streamlines)
    T = np.array([s for s in T if length(s) >= threshold_short_streamlines], dtype=np.object)
    print("%s: %s streamlines" % (T_filename, len(T)))
    return T, hdr
    
    
def compute_kdtree_and_dr_tractogram( tractogram, num_prototypes=None):
    """Compute the dissimilarity representation of the target tractogram and 
    build the kd-tree.
    """
    tractogram = np.array(tractogram, dtype=np.object)
   
    print("Computing dissimilarity matrices")
    if num_prototypes is None:
        num_prototypes = 40
        print("Using %s prototypes as in Olivetti et al. 2012"
              % num_prototypes)

    print("Using %s prototypes" % num_prototypes)
    dm_tractogram, prototype_idx = compute_dissimilarity(tractogram,
                                                         num_prototypes=num_prototypes,
                                                         distance= bundles_distances_mam,
                                                         prototype_policy='sff',
                                                         n_jobs=-1,
                                                         verbose=False)
    
    prototypes = tractogram[prototype_idx]
    

    print("Building the KD-tree of tractogram")
    kdt = KDTree(dm_tractogram)
    
    return kdt, prototypes    
    
def NN(kdt, dm_E_t, num_NN ):
    """Code for efficient nearest neighbors computation.
    """
    D, I = kdt.query(dm_E_t, k=num_NN)
    
  
    if num_NN==1:     
      return I.squeeze(), D.squeeze(), dm_E_t.shape[0]
    else:    
      return  np.unique(I.flat)


def bundles_distances_mam_smarter_faster(A, B, n_jobs=-1, chunk_size=100):
    """Parallel version of bundles_distances_mam that also avoids
    computing distances twice.
    """
    lenA = len(A)
    chunks = chunker(A, chunk_size)
    if B is None:
        dm = np.empty((lenA, lenA), dtype=np.float32)
        dm[np.diag_indices(lenA)] = 0.0
        results = Parallel(n_jobs=-1)(delayed(bundles_distances_mam)(ss, A[i*chunk_size+1:]) for i, ss in enumerate(chunks))
        # Fill triu
        for i, res in enumerate(results):
            dm[(i*chunk_size):((i+1)*chunk_size), (i*chunk_size+1):] = res
            
        # Copy triu to trid:
        rows, cols = np.triu_indices(lenA, 1)
        dm[cols, rows] = dm[rows, cols]

    else:
        dm = np.vstack(Parallel(n_jobs=n_jobs)(delayed(bundles_distances_mam)(ss, B) for ss in chunks))

    return dm


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))
 

def tract_segmentation_single_example_lap (kdt_T_A,  prototypes_T_A,sid, num_NN,T_A ):                            
    """ step 1: tract segmentation from a single example using Jonker-Volgenant algorithm (LAPJV)
    """        
    E_t_filename= 'data/example/'+ str(sid) +'_'+str(tract_name)+'.trk'
    
    print("Loading Example tract: %s" % E_t_filename)
    
    E_t, hdr=  load(E_t_filename, threshold_short_streamlines=threshold_short_streamlines)                          
    
    dm_E_t= dissimilarity(E_t, prototypes_T_A,bundles_distances_mam)
    
    #compute the NN of the example tract in order to construcse the cost matrix
    NN_E_t_NN_Idx= NN (kdt_T_A, dm_E_t,num_NN)    
    
   
    print("Computing the cost matrix with mam distance (%s x %s) for RLAP " % (len(E_t),
                                                             len( NN_E_t_NN_Idx)))
    
    cost_matrix = bundles_distances_mam_smarter_faster(E_t, 
                                                       T_A[NN_E_t_NN_Idx])
    
    print("Computing optimal assignmnet with LAPJV")
    assignment = LinearAssignment(cost_matrix).solution
    
    
    min_cost_values=  cost_matrix[np.arange(len(cost_matrix)), assignment]

    
   
    return NN_E_t_NN_Idx[assignment], min_cost_values, len(E_t)
    


def tract_correspondence_multiple_example_lap (kdt_T_A,  prototypes_T_A,example_sunject_id_list, num_NN ):
    """ step:2 tracts generated from each example are merged together and then filtered 
        in order to obtain the final segmentation of the desired tract
    """    
    print("Extracting the estimated target tract (superset) using the RLAP")
    n_jobs=-1
   
      
    result_LAP= np.array(Parallel(n_jobs=n_jobs)(delayed(tract_segmentation_single_example_lap)(kdt_T_A,  prototypes_T_A,sid, num_NN,T_A ) for sid in   example_sunject_id_list ))                        


    superset_estimated_correspondence_tract_idx= np.hstack(result_LAP[:,0]) 
    superset_estimated_correspondence_tract_cost= np.hstack(result_LAP[:,1])
    example_tract_len_med=np.median(np.hstack(result_LAP[:,2]))
    
    print("Ranking the estimated target (superset) tract.")
    superset_estimated_correspondence_tract_idx_ranked=ranking_schema(superset_estimated_correspondence_tract_idx,
                                                           superset_estimated_correspondence_tract_cost)
                                                           
    

    print("Extracting the estimated target tract (until the median size (in terms of number of streamlines) of all the tracts from the example).")
    superset_estimated_correspondence_tract_idx_ranked_med=superset_estimated_correspondence_tract_idx_ranked[0:int(example_tract_len_med)]

   
    segmented_tract_LAP=T_A [ superset_estimated_correspondence_tract_idx_ranked_med]
    
    
  
      
    print("Saving the estimated target (superset) (.trk)")
    prefix="lap"
    save_trk( tract_name, 
              test_tractogram, 
              segmented_tract_LAP,
              hdr, 
              prefix)
     
    
    print("Show the tract")
    color= colors.blue
    show_tract(segmented_tract_LAP,color)                                                     
    



def tract_segmentation_single_example_NN (kdt_T_A,  prototypes_T_A,sid, num_NN,T_A ):                            
    """ step:1 tract segmentation from single example using lapjv
    """        
    E_t_filename= 'data/example/'+ str(sid) +'_'+str(tract_name)+'.trk'
    
    print("Loading Example tract: %s" % E_t_filename)
    
    E_t, hdr=  load(E_t_filename, threshold_short_streamlines=threshold_short_streamlines)                          
    
    dm_E_t= dissimilarity(E_t, prototypes_T_A,bundles_distances_mam)
    
    #compute the NN of the example tract in order to construcse the cost matrix
    assignmnet, min_cost_value, len_E_T = NN (kdt_T_A, dm_E_t,num_NN)    
  
   
    return  assignmnet, min_cost_value, len_E_T
    


def tract_correspondence_multiple_example_NN (kdt_T_A,  prototypes_T_A,example_subject_id_list,num_NN ):
    """ step:2 tract segmentation using multiple example
    """    
    print("Extracting the estimated target tract (superset) using the RLAP")
    n_jobs=-1
   
    
    result_NN= np.array(Parallel(n_jobs=n_jobs)(delayed(tract_segmentation_single_example_NN)(kdt_T_A,  prototypes_T_A,sid, num_NN,T_A ) for sid in   example_subject_id_list ))#euclidean                            


    superset_estimated_correspondence_tract_idx= np.hstack(result_NN[:,0]) 
    superset_estimated_correspondence_tract_cost= np.hstack(result_NN[:,1])
    example_tract_len_med=np.median(np.hstack(result_NN[:,2]))
    
    print("Ranking the estimated target (superset) tract.")
    superset_estimated_correspondence_tract_idx_ranked=ranking_schema(superset_estimated_correspondence_tract_idx,
                                                           superset_estimated_correspondence_tract_cost)
                                                           
    

    print("Extracting the estimated target tract (until the median size (in terms of number of streamlines) of all the tracts from the example).")
    superset_estimated_correspondence_tract_idx_ranked_med=superset_estimated_correspondence_tract_idx_ranked[0:int(example_tract_len_med)]

   
    segmented_tract_NN=T_A [ superset_estimated_correspondence_tract_idx_ranked_med]
    
    
    print len (segmented_tract_NN)
      
    print("Saving the estimated target (superset) (.trk)")
    prefix="NN"
    save_trk(tract_name, 
             test_tractogram, 
             segmented_tract_NN,
             hdr,
             prefix)
     
    
    print("Show the tract")
    color= colors.green
    show_tract(segmented_tract_NN,
               color)                                                     
    


    
    
def save_trk(tract_name, test_tractogram, segmented_tract_LAP,  hdr, prefix):
    """Save the segmented tract estimated from the LAP 
    """      
    filedir = os.path.dirname('data/segmented_tract/')
    if not os.path.exists(filedir):
            os.makedirs(filedir) 
            
    save_segmented_tract_LAP_filename = '%s/%s_%s_%s.trk'%\
                                            (filedir, test_tractogram, tract_name, prefix)
    
    strmR_A = ((sl, None, None) for sl in  segmented_tract_LAP )               
    trackvis.write(  save_segmented_tract_LAP_filename ,strmR_A ,  hdr)
    
if __name__ == '__main__':
    
    print(__doc__)
    np.random.seed(0)

    # test tractogram
    test_tractogram = "100307" 
    T_A_filename = 'data/test_tractogram/tractogram_b1k_1.25mm_csd_wm_mask_eudx1M.trk'
    

    # Main parameters:
    
    threshold_short_streamlines = 0.0  # Beware: discarding streamlines affects IDs    
    num_NN_lap = 500  # number of nesrest neighbour in order to sparsify the cost matrix. 
    num_example= 3
    num_prototypes=40
    num_NN=1      
    
    tract_name= "uf.left"
    
    example_subject_id_list= ["100408", "128632", "103414"]
    
    # 1) load test tractogram, T_A
    
    T_A, hdr = load(T_A_filename, threshold_short_streamlines=threshold_short_streamlines)    
    
    
    
    # 2) Compute the dissimilarity representation of T_A
    
    print("Computing the dissimilarity representation and KD-tree.")
    kdt_T_A, prototypes_T_A = compute_kdtree_and_dr_tractogram( T_A, 
                                                       num_prototypes)
   
   
    print("Segmenting tract with NN")                                             
    tract_correspondence_multiple_example_NN (kdt_T_A,  
                                              prototypes_T_A,
                                              example_subject_id_list,
                                              num_NN=num_NN )

    print("Segmenting tract with lap")                                                  
    tract_correspondence_multiple_example_lap (kdt_T_A,  
                                               prototypes_T_A,
                                               example_subject_id_list,
                                               num_NN=num_NN_lap )
    
   
