import numpy as np

try:
    from .datasets.colmap import Parser
    from .datasets.nerf_synth import SimpleParser
    from .evaluation import umeyama_alignment
except ImportError:  # TODO Figure out a better way to do this
    from datasets.colmap import Parser
    from datasets.nerf_synth import SimpleParser
    from evaluation import umeyama_alignment


def copy_cameras(
    from_parser: Parser | SimpleParser,
    to_parser: Parser | SimpleParser,
    copy_extrinsics: bool,
    copy_intrinsics: bool,
    copy_points: bool = False,
):
    """
    Copy the camera extrinsics, intrinsics, or 3D points from one parser to another in place after performing umeyama alignment
    """
    # Convert set to a sorted list to guarantee deterministic iteration order for array alignment
    common_names = sorted(list(set(from_parser.image_names) & set(to_parser.image_names)))

    # Calculate Umeyama alignment if we are copying either extrinsics or points,
    # as the points also need to be projected into the estimated coordinate space.
    if copy_extrinsics or copy_points:
        from_c2w_dict = {
            name: c2w for c2w, name in zip(from_parser.camtoworlds, from_parser.image_names) if name in common_names
        }
        to_c2w_dict = {
            name: c2w for c2w, name in zip(to_parser.camtoworlds, to_parser.image_names) if name in common_names
        }

        from_centers = np.array([from_c2w_dict[name] for name in common_names])[:, :3, 3]
        to_centers = np.array([to_c2w_dict[name] for name in common_names])[:, :3, 3]
        s, R, t = umeyama_alignment(from_centers, to_centers)
    else:
        s, R, t = 1.0, np.eye(3), np.zeros(3)

    if copy_extrinsics:
        assert len(common_names) == len(to_parser.image_names)

        for name in common_names:
            from_c2w = from_c2w_dict[name]
            to_i = to_parser.image_names.index(name)
            new_to_c2w = from_c2w.copy()
            # Apply Umeyama alignment to translate the GT camera into the Estimated coordinate space
            new_to_c2w[:3, 3] = s * R @ new_to_c2w[:3, 3] + t
            new_to_c2w[:3, :3] = R @ new_to_c2w[:3, :3]
            to_parser.camtoworlds[to_i] = new_to_c2w

    if copy_intrinsics:
        for name in common_names:
            from_i = from_parser.image_names.index(name)
            to_i = to_parser.image_names.index(name)

            from_cam_id = from_parser.camera_ids[from_i]
            to_cam_id = to_parser.camera_ids[to_i]

            # Copy all relevant intrinsic dictionaries
            to_parser.Ks_dict[to_cam_id] = from_parser.Ks_dict[from_cam_id].copy()
            to_parser.params_dict[to_cam_id] = from_parser.params_dict[from_cam_id].copy()
            to_parser.imsize_dict[to_cam_id] = from_parser.imsize_dict[from_cam_id]

            if from_parser.mask_dict[from_cam_id] is not None:
                to_parser.mask_dict[to_cam_id] = from_parser.mask_dict[from_cam_id].copy()
            else:
                to_parser.mask_dict[to_cam_id] = None

            # Copy undistortion maps if they exist
            if hasattr(from_parser, "mapx_dict") and from_cam_id in from_parser.mapx_dict:
                to_parser.mapx_dict[to_cam_id] = from_parser.mapx_dict[from_cam_id].copy()
                to_parser.mapy_dict[to_cam_id] = from_parser.mapy_dict[from_cam_id].copy()
                to_parser.roi_undist_dict[to_cam_id] = list(from_parser.roi_undist_dict[from_cam_id])

    if copy_points:
        # Transform the GT 3D points into the Estimated coordinate space
        # R is applied via transpose because from_parser.points is shape (N, 3)
        to_parser.points = (s * from_parser.points @ R.T) + t

        # Copy colors
        to_parser.points_rgb = from_parser.points_rgb.copy()

        # Scale the reprojection errors (depth uncertainty proxy) by the spatial scale factor
        to_parser.points_err = from_parser.points_err.copy() * s

        # Update image-to-point indices, mapping only for common images
        new_indices = {}
        for name in common_names:
            if name in from_parser.point_indices:
                new_indices[name] = from_parser.point_indices[name].copy()
        to_parser.point_indices = new_indices
