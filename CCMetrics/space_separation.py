import cc3d
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def compute_voronoi_regions(labels):
    """
    Compute Voronoi regions for the given labels.

    Parameters:
        labels (ndarray): Input label array.

    Returns:
        ndarray: Array of Voronoi region assignments.

    """
    cc_labels = cc3d.connected_components(labels)
    current_assignment = np.zeros_like(cc_labels, dtype="int")
    current_mins = np.ones_like(cc_labels, dtype="float") * np.inf
    for idx, cc in enumerate(np.unique(cc_labels)):
        if cc == 0:
            pass
        else:
            # Compute distance transforms from current cc
            cur_dt = distance_transform_edt(np.logical_not(cc_labels == cc))
            # Update the cc_asignment and previous minimas
            msk = cur_dt < current_mins
            current_mins[msk] = cur_dt[msk]
            current_assignment[msk] = idx
    cc_asignment = current_assignment
    return cc_asignment


def compute_voronoi_kdtree(labels):
    """
    Computes the Voronoi diagram using a KDTree for a given label image.

    Parameters:
        labels (ndarray): The label image.

    Returns:
        ndarray: Array of Voronoi region assignments.
    """
    cc_labels = cc3d.connected_components(labels)
    output = np.zeros_like(cc_labels, dtype=np.int32)

    coords = np.column_stack(np.nonzero(cc_labels))
    cc_ids = cc_labels[cc_labels > 0]
    unique_ccs = np.unique(cc_ids)

    # Map each cc_id to its voxel coordinates
    cc_points = {cc: coords[cc_ids == cc] for cc in unique_ccs}

    # Build a KDTree using all foreground voxels, tagged with their cc_id
    all_pts = np.concatenate([cc_points[cc] for cc in unique_ccs])
    all_tags = np.concatenate([[cc] * len(cc_points[cc]) for cc in unique_ccs])

    tree = cKDTree(all_pts)

    # For each voxel in the volume, find the nearest foreground point and assign its cc_id
    all_voxels = np.indices(cc_labels.shape).reshape(3, -1).T
    dists, idxs = tree.query(all_voxels)
    nearest_ccs = all_tags[idxs]
    output = nearest_ccs.reshape(cc_labels.shape)

    return output
