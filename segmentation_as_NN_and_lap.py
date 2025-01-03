"""White Matter Tract Segmentation as Multiple Linear Assignment
Problems (LAPs).
"""

from dipy.tracking.distances import bundles_distances_mam
from sklearn.neighbors import KDTree
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.streamline import set_number_of_points
import nibabel as nib
import numpy as np
import random
from typing import Any, Literal

from .dissimilarity import compute_dissimilarity, dissimilarity

try:
    from .linear_assignment import LinearAssignment
except ImportError:
    print("WARNING: Cythonized LAPJV not available. Falling back to Python.")
    print("WARNING: See README.txt")
    from .linear_assignment_numpy import LinearAssignment

try:
    from joblib import Parallel, delayed

    joblib_available = True
except ImportError:
    joblib_available = False


def ranking_schema(
    superset_estimated_target_tract_idx, superset_estimated_target_tract_cost
):
    """Rank all the extracted streamlines estimated by the LAP with
    different examples (superset) accoring to the number of times it
    selected and the total cost
    """
    idxs = np.unique(superset_estimated_target_tract_idx)
    how_many_times_selected = np.array(
        [(superset_estimated_target_tract_idx == idx).sum() for idx in idxs]
    )
    how_much_cost = np.array(
        [
            (
                (superset_estimated_target_tract_idx == idx)
                * superset_estimated_target_tract_cost
            ).sum()
            for idx in idxs
        ]
    )
    ranking = np.argsort(how_many_times_selected)[::-1]
    tmp = np.unique(how_many_times_selected)[::-1]
    for i in tmp:
        tmp1 = how_many_times_selected == i
        tmp2 = np.where(tmp1)[0]
        if tmp2.size > 1:
            tmp3 = np.argsort(how_much_cost[tmp2])
            ranking[how_many_times_selected[ranking] == i] = tmp2[tmp3]

    return idxs[ranking]


def compute_kdtree_and_dr_tractogram(tractogram, num_prototypes=None):
    """Compute the dissimilarity representation of the target tractogram and
    build the kd-tree.
    """
    tractogram = np.array(tractogram, dtype=object)

    print("Computing dissimilarity matrices")
    if num_prototypes is None:
        num_prototypes = 40
        print("Using %s prototypes as in Olivetti et al. 2012" % num_prototypes)

    print("Using %s prototypes" % num_prototypes)
    dm_tractogram, prototype_idx = compute_dissimilarity(
        tractogram,
        num_prototypes=num_prototypes,
        distance=bundles_distances_mam,
        prototype_policy="sff",
        n_jobs=-1,
        verbose=False,
    )

    prototypes = tractogram[prototype_idx]

    print("Building the KD-tree of tractogram")
    kdt = KDTree(dm_tractogram)

    return kdt, prototypes


def NN(kdt, dm_E_t, num_NN=1):
    """Code for efficient nearest neighbors computation."""
    D, I = kdt.query(dm_E_t, k=num_NN)

    if num_NN == 1:
        return I.squeeze(), D.squeeze(), dm_E_t.shape[0]
    else:
        return np.unique(I.flat)


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def bundles_distances_mam_smarter_faster(A, B, n_jobs=-1, chunk_size=100):
    """Parallel version of bundles_distances_mam that also avoids
    computing distances twice.
    """
    lenA = len(A)
    chunks = chunker(A, chunk_size)
    if B is None:
        dm = np.empty((lenA, lenA), dtype=np.float32)
        dm[np.diag_indices(lenA)] = 0.0
        results = Parallel(n_jobs=-1)(
            delayed(bundles_distances_mam)(ss, A[i * chunk_size + 1 :])
            for i, ss in enumerate(chunks)
        )
        # Fill triu
        for i, res in enumerate(results):
            dm[(i * chunk_size) : ((i + 1) * chunk_size), (i * chunk_size + 1) :] = res

        # Copy triu to trid:
        rows, cols = np.triu_indices(lenA, 1)
        dm[cols, rows] = dm[rows, cols]

    else:
        dm = np.vstack(
            Parallel(n_jobs=n_jobs)(
                delayed(bundles_distances_mam)(ss, B) for ss in chunks
            )
        )

    return dm


def tract_segmentation_single_example_lap(
    kdt_T_A, prototypes_T_A, train_tract, num_NN, T_A
):
    """step 1: tract segmentation from a single example using Jonker-Volgenant algorithm (LAPJV)"""
    dm_E_t = dissimilarity(train_tract, prototypes_T_A, bundles_distances_mam)

    # compute the NN of the example tract in order to construct the cost matrix
    NN_E_t_NN_Idx = NN(kdt_T_A, dm_E_t, num_NN)

    print(
        "Computing the cost matrix with mam distance (%s x %s) for RLAP "
        % (len(train_tract), len(NN_E_t_NN_Idx))
    )

    cost_matrix = bundles_distances_mam_smarter_faster(train_tract, T_A[NN_E_t_NN_Idx])

    print("Computing optimal assignment with LAPJV")
    assignment = LinearAssignment(cost_matrix).solution

    min_cost_values = cost_matrix[np.arange(len(cost_matrix)), assignment]

    return NN_E_t_NN_Idx[assignment], min_cost_values, len(train_tract)


def tract_correspondence_multiple_example_lap(streamlines, train_tracts, num_NN):
    """step:2 tracts generated from each example are merged together and then filtered
    in order to obtain the final segmentation of the desired tract
    """
    kdt_T_A, prototypes_T_A = compute_kdtree_and_dr_tractogram(streamlines, 40)

    print("Extracting the estimated target tract (superset) using the RLAP")
    n_jobs = -1

    result_LAP = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(tract_segmentation_single_example_lap)(
                kdt_T_A, prototypes_T_A, train_tract, num_NN, streamlines
            )
            for train_tract in train_tracts
        ),
        dtype=object,
    )

    superset_estimated_correspondence_tract_idx = np.hstack(result_LAP[:, 0])
    superset_estimated_correspondence_tract_cost = np.hstack(result_LAP[:, 1])
    example_tract_len_med = np.median(np.hstack(result_LAP[:, 2]))

    print("Ranking the estimated target (superset) tract.")
    superset_estimated_correspondence_tract_idx_ranked = ranking_schema(
        superset_estimated_correspondence_tract_idx,
        superset_estimated_correspondence_tract_cost,
    )

    print(
        "Extracting the estimated target tract (until the median size (in terms of number of streamlines) of all the tracts from the example)."
    )
    superset_estimated_correspondence_tract_idx_ranked_med = (
        superset_estimated_correspondence_tract_idx_ranked[
            0 : int(example_tract_len_med)
        ]
    )

    return streamlines[superset_estimated_correspondence_tract_idx_ranked_med]


def tract_segmentation_single_example_NN(kdt_T_A, prototypes_T_A, train_tract):
    """step:1 tract segmentation from single example using lapjv"""
    dm_E_t = dissimilarity(train_tract, prototypes_T_A, bundles_distances_mam)

    # compute the NN of the example tract in order to construct the cost matrix
    return NN(kdt_T_A, dm_E_t)


def tract_correspondence_multiple_example_NN(streamlines, train_tracts):
    """step:2 tract segmentation using multiple example"""
    kdt_T_A, prototypes_T_A = compute_kdtree_and_dr_tractogram(streamlines, 40)

    print("Extracting the estimated target tract (superset) using the RLAP")
    n_jobs = -1

    result_NN = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(tract_segmentation_single_example_NN)(
                kdt_T_A, prototypes_T_A, train_tract
            )
            for train_tract in train_tracts
        )
    )  # euclidean

    superset_estimated_correspondence_tract_idx = np.hstack(result_NN[:, 0])
    superset_estimated_correspondence_tract_cost = np.hstack(result_NN[:, 1])
    example_tract_len_med = np.median(np.hstack(result_NN[:, 2]))

    print("Ranking the estimated target (superset) tract.")
    superset_estimated_correspondence_tract_idx_ranked = ranking_schema(
        superset_estimated_correspondence_tract_idx,
        superset_estimated_correspondence_tract_cost,
    )

    print(
        "Extracting the estimated target tract (until the median size (in terms of number of streamlines) of all the tracts from the example)."
    )
    superset_estimated_correspondence_tract_idx_ranked_med = (
        superset_estimated_correspondence_tract_idx_ranked[
            0 : int(example_tract_len_med)
        ]
    )

    return streamlines[superset_estimated_correspondence_tract_idx_ranked_med]


def segment_lap(
    test_tractogram: str | Any,
    train_tracts: list[str | Any],
    test_reference: str | nib.nifti1.Nifti1Image,
    train_references: list[str | nib.nifti1.Nifti1Image],
    output: str,
    max_train_streamlines: int = 5000,
    algorithm: Literal["LAP", "NN"] = "LAP",
):
    """
    Executes tract segmentation using LAP.

    :param test_tractogram: The input tractogram file.
    :param train_tracts: The reference tract files.
    :param test_reference: A NIfTI file to use as spatial reference for the input tractogram file.
    :param train_references: NIfTI files to use as spatial reference for the reference tract files.
    :param output: The output tract file.
    :param max_train_streamlines: The maximum number of streamlines to use from each reference. Will be randomly sampled.
    :param algorithm: The algorithm used for the segmentation. If set to "NN", uses nearest neighbor instead of LAP.
    """
    train_streamlines = [
        set_number_of_points(
            (
                load_tractogram(
                    train_tract, train_references[i], bbox_valid_check=False
                ).streamlines
                if isinstance(train_tract, str)
                else train_tract
            ),
            nb_points=30,
        )
        for i, train_tract in enumerate(train_tracts)
    ]
    train_streamlines = [
        (
            train_tract
            if len(train_tract) <= max_train_streamlines
            else random.sample(
                [train_streamline for train_streamline in train_tract],
                max_train_streamlines,
            )
        )
        for train_tract in train_streamlines
    ]

    test_streamlines = set_number_of_points(
        (
            load_tractogram(
                test_tractogram, test_reference, bbox_valid_check=False
            ).streamlines
            if isinstance(test_tractogram, str)
            else test_tractogram
        ),
        nb_points=30,
    )

    segmented_tract = (
        tract_correspondence_multiple_example_lap(
            test_streamlines,
            train_streamlines,
            num_NN=max(
                500,
                round(max([len(train_tract) for train_tract in train_streamlines]) / 4),
            ),
        )
        if algorithm == "LAP"
        else tract_correspondence_multiple_example_NN(
            test_streamlines, train_streamlines
        )
    )

    save_tractogram(
        StatefulTractogram(segmented_tract, test_reference, space=Space.RASMM), output
    )
