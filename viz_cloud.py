import os
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import trimesh
import cv2
from typing import Optional, Union, Tuple, Dict, List, Any
from examples.datasets.colmap import Parser
from evaluation import umeyama_alignment, calculate_metrics


def create_frustum_traces(c2w: np.ndarray, color: str, name: str, size: float = 0.1) -> List[go.Scatter3d]:
    """Creates Plotly 3D traces for a camera frustum."""
    pts = np.array([[0, 0, 0], [1, 1, 2], [-1, 1, 2], [-1, -1, 2], [1, -1, 2]]) * size
    # Transform to world coordinates
    pts_world = (c2w[:3, :3] @ pts.T + c2w[:3, 3:4]).T

    # Define the 8 lines of the frustum
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

    traces = []
    for line in lines:
        p0, p1 = pts_world[line[0]], pts_world[line[1]]
        traces.append(
            go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False,
                hoverinfo="name",
                name=name,
            )
        )
    return traces


def create_point_cloud_trace(
    points: np.ndarray, color: Union[str, np.ndarray], name: str, subsample: int = 1
) -> go.Scatter3d:
    """Creates a Scatter3d trace for a point cloud with subsampling for performance."""
    # Subsample points to avoid freezing the browser (Plotly can struggle with >50k points)
    pts_sub = points[::subsample]

    # Handle color: if it's an array (RGB/Confidence), subsample and format for Plotly
    if isinstance(color, np.ndarray):
        col_sub = color[::subsample]
        # Ensure we just take the first 3 channels (RGB) and cast to integer for formatting
        col_sub = col_sub[:, :3].astype(int)
        marker_color = [f"rgb({r},{g},{b})" for r, g, b in col_sub]
    else:
        marker_color = color

    return go.Scatter3d(
        x=pts_sub[:, 0],
        y=pts_sub[:, 1],
        z=pts_sub[:, 2],
        mode="markers",
        marker=dict(size=3.0, color=marker_color, opacity=0.6),
        name=name,
    )


def project_points(points_3d: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D points into 2D image coordinates using a 3x3 Intrinsic Matrix.
    """
    # w2c transform
    w2c = np.linalg.inv(c2w)
    pts_cam = (w2c[:3, :3] @ points_3d.T).T + w2c[:3, 3]

    # Perspective projection using Matrix K
    # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Z-normalization
    x = (pts_cam[:, 0] * fx / pts_cam[:, 2]) + cx
    y = (pts_cam[:, 1] * fy / pts_cam[:, 2]) + cy

    return np.stack([x, y], axis=1), pts_cam[:, 2]


def load_parser_data(path: str) -> Tuple[Optional[Parser], Optional[Dict], Optional[str]]:
    """Safe wrapper to initialize Parser and extract poses."""
    try:
        # Parser expects data_dir. Normalize=False to keep raw scale for alignment.
        parser = Parser(data_dir=path, normalize=False)
        # Create dict {image_name: c2w} for metrics calculation
        poses = {name: c2w for name, c2w in zip(parser.image_names, parser.camtoworlds)}
        return parser, poses, None
    except Exception as e:
        return None, None, str(e)


def run_gradio_eval(
    pred_path: str, gt_path: str, show_gt_pts: bool, show_pred_pts: bool, pred_color_mode: str
) -> Tuple[str, Optional[go.Figure], Dict, Optional[Parser], Optional[Parser]]:
    # 1. Load data using Parser
    gt_parser, gt_poses, gt_err = load_parser_data(gt_path)
    if gt_err or gt_parser is None or gt_poses is None:
        return f"Error loading GT: {gt_err}", None, {}, None, None

    pred_parser, pred_poses, pred_err = load_parser_data(pred_path)
    if pred_err or pred_parser is None or pred_poses is None:
        return f"Error loading Pred: {pred_err}", None, {}, None, None

    # 2. Calculate Metrics
    metrics = calculate_metrics(pred_poses, gt_poses)
    if "error" in metrics:
        return f"Error: {metrics['error']}", None, metrics, None, None

    common_names = sorted(list(set(pred_poses.keys()) & set(gt_poses.keys())))
    p_centers = np.array([pred_poses[n][:3, 3] for n in common_names])
    g_centers = np.array([gt_poses[n][:3, 3] for n in common_names])

    # 3. Alignment
    s, R, t = umeyama_alignment(p_centers, g_centers)

    fig = go.Figure()

    # 4. GT Points Visualization
    if show_gt_pts:
        if len(gt_parser.points) > 0:
            fig.add_trace(create_point_cloud_trace(gt_parser.points, "green", "GT Points"))

    # 5. Pred Points Visualization
    if show_pred_pts:
        try:
            # Special handling for confidence PLY in prediction folder (retained from original)
            ply_filename = "points_conf.ply" if pred_color_mode == "Confidence" else "points.ply"
            ply_file_path = os.path.join(pred_path, "sparse", ply_filename)

            pred_pts = None
            pred_colors = "red"

            # Try loading specific PLY if exists (for confidence visualization)
            if os.path.exists(ply_file_path):
                pc = trimesh.load(ply_file_path)
                if isinstance(pc, trimesh.PointCloud):
                    pred_pts = np.array(pc.vertices)
                    if hasattr(pc, "colors") and len(pc.colors) > 0:
                        pred_colors = np.array(pc.colors)

            if pred_pts is None and len(pred_parser.points) > 0:
                pred_pts = pred_parser.points
                if len(pred_parser.points_rgb) > 0:
                    pred_colors = pred_parser.points_rgb

            if pred_pts is not None and len(pred_pts) > 0:
                # Apply the alignment transform to the prediction point cloud
                # transform: x' = s * R * x + t
                pred_pts_aligned = (s * (R @ pred_pts.T)).T + t.T

                # Update name based on mode
                trace_name = (
                    f"Pred Points ({pred_color_mode})"
                    if isinstance(pred_colors, np.ndarray)
                    else "Pred Points (Aligned)"
                )
                fig.add_trace(create_point_cloud_trace(pred_pts_aligned, pred_colors, trace_name))
        except Exception as e:
            print(f"Could not load Pred points: {e}")

    # 6. Frustum Visualization
    for name in common_names:
        # Align Pred
        p_c2w = pred_poses[name].copy()
        p_c2w[:3, 3] = s * R @ p_c2w[:3, 3] + t
        p_c2w[:3, :3] = R @ p_c2w[:3, :3]

        g_c2w = gt_poses[name]

        gt_traces = create_frustum_traces(g_c2w, color="green", name=f"GT_{name}")
        pred_traces = create_frustum_traces(p_c2w, color="red", name=f"Pred_{name}")

        for t_trace in gt_traces + pred_traces:
            t_trace.customdata = [name] * len(t_trace.x)
            t_trace.hoverinfo = "name"

        fig.add_traces(gt_traces)
        fig.add_traces(pred_traces)

        # Add Error Vector (RTE)
        fig.add_trace(
            go.Scatter3d(
                x=[g_c2w[0, 3], p_c2w[0, 3]],
                y=[g_c2w[1, 3], p_c2w[1, 3]],
                z=[g_c2w[2, 3], p_c2w[2, 3]],
                mode="lines",
                line=dict(color="yellow", width=2),
                showlegend=False,
            )
        )

    fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, b=0, t=0), template="plotly_dark")

    summary_text = (
        f"### Alignment Success\n"
        f"- **Matched Images:** {metrics['num_aligned']}\n"
        f"- **Mean RRE:** {metrics['mean_rre_deg']:.3f}°\n"
        f"- **Mean RTE:** {metrics['mean_rte']:.5f}"
    )

    return summary_text, fig, metrics, gt_parser, pred_parser


def run_gradio_eval_with_names(
    pred_path: str, gt_path: str, show_gt_pts: bool, show_pred_pts: bool, pred_color_mode: str
) -> Tuple:
    summary, fig, metrics, gt_parser, pred_parser = run_gradio_eval(
        pred_path, gt_path, show_gt_pts, show_pred_pts, pred_color_mode
    )

    if metrics is None or gt_parser is None or pred_parser is None:
        return summary, fig, {}, gr.update(), gr.update(), [], None, None

    # Get names from loaded parser
    gt_poses = {n: c2w for n, c2w in zip(gt_parser.image_names, gt_parser.camtoworlds)}
    pred_poses = {n: c2w for n, c2w in zip(pred_parser.image_names, pred_parser.camtoworlds)}
    common_names = sorted(list(set(pred_poses.keys()) & set(gt_poses.keys())))

    if common_names:
        return (
            summary,
            fig,
            metrics,
            gr.update(choices=common_names, value=common_names[0]),
            gr.update(maximum=len(common_names) - 1, value=0, visible=True),
            common_names,
            gt_parser,
            pred_parser,
        )
    else:
        return summary, fig, metrics, gr.update(), gr.update(), [], gt_parser, pred_parser


def render_depth_overlay(img_ref: np.ndarray, points_3d: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Projects 3D points onto image and draws them with depth coloring."""
    h, w = img_ref.shape[:2]

    # Project points and get depths
    pts_2d, depths = project_points(points_3d, pose, K)

    # Filter valid points (in front of camera and inside image)
    valid = (depths > 0) & (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)

    pts_valid = pts_2d[valid]
    depths_valid = depths[valid]

    # Sort by depth (farthest first) so closer points overwrite
    sort_idx = np.argsort(depths_valid)[::-1]
    pts_valid = pts_valid[sort_idx]
    depths_valid = depths_valid[sort_idx]

    # Normalize depth for colormap
    if len(depths_valid) > 0:
        d_min, d_max = depths_valid.min(), depths_valid.max()
        norm_depth = (depths_valid - d_min) / (d_max - d_min + 1e-8)
        norm_depth = (norm_depth * 255).astype(np.uint8)

        cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        colors = cv2.applyColorMap(norm_depth[:, None], cmap)  # Shape (N, 1, 3)
    else:
        colors = []

    img_viz = img_ref.copy()

    stride = 5
    for pt, color in zip(pts_valid[::stride], colors[::stride]):
        cv2.circle(img_viz, (int(pt[0]), int(pt[1])), 2, color[0].tolist(), -1)

    return img_viz


def update_projection_comparison(
    camera_name: str, gt_parser_state: Optional[Parser], pred_parser_state: Optional[Parser]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Update function utilizing the cached Parser objects."""
    if not camera_name:
        return None, None, "Select a camera."

    if not gt_parser_state or not pred_parser_state:
        return None, None, "Run evaluation first."

    gt_parser = gt_parser_state
    pred_parser = pred_parser_state

    # 1. Setup Alignment
    gt_poses = {n: c2w for n, c2w in zip(gt_parser.image_names, gt_parser.camtoworlds)}
    pred_poses = {n: c2w for n, c2w in zip(pred_parser.image_names, pred_parser.camtoworlds)}

    common_names = sorted(list(set(pred_poses.keys()) & set(gt_poses.keys())))
    if camera_name not in common_names:
        return None, None, f"Camera {camera_name} not found in common set."

    p_centers = np.array([pred_poses[n][:3, 3] for n in common_names])
    g_centers = np.array([gt_poses[n][:3, 3] for n in common_names])
    s, R, t = umeyama_alignment(p_centers, g_centers)

    # 2. Get Image and Intrinsics from Parser
    if camera_name in gt_parser.image_names:
        idx = gt_parser.image_names.index(camera_name)
        img_path = gt_parser.image_paths[idx]
        cam_id = gt_parser.camera_ids[idx]
        gt_K = gt_parser.Ks_dict[cam_id]

        if not os.path.exists(img_path):
            return None, None, f"Image {img_path} not found."

        img_ref = cv2.imread(img_path)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

        # Undistort if Parser has computed undistortion maps
        if cam_id in gt_parser.mapx_dict:
            mapx = gt_parser.mapx_dict[cam_id]
            mapy = gt_parser.mapy_dict[cam_id]
            img_ref = cv2.remap(img_ref, mapx, mapy, interpolation=cv2.INTER_LINEAR)

    else:
        return None, None, "Camera not found in GT parser."

    # For Pred intrinsics, we assume matching camera ID or look up by name
    if camera_name in pred_parser.image_names:
        idx_p = pred_parser.image_names.index(camera_name)
        cam_id_p = pred_parser.camera_ids[idx_p]
        pred_K = pred_parser.Ks_dict[cam_id_p]
    else:
        pred_K = gt_K  # Fallback

    # 3. Generate GT Projection
    gt_viz = render_depth_overlay(img_ref, gt_parser.points, gt_poses[camera_name], gt_K)

    # 4. Generate Pred Projection (Aligned)
    # Align Pred Points to GT World
    pred_points = pred_parser.points
    if len(pred_points) > 0:
        pred_pts_aligned = (s * (R @ pred_points.T)).T + t.T
        pred_viz = render_depth_overlay(img_ref, pred_pts_aligned, gt_poses[camera_name], pred_K)
    else:
        pred_viz = img_ref

    return gt_viz, pred_viz, f"Visualizing: {camera_name} (Scale: {s:.2f})"


def sync_slider_to_dropdown(idx: Union[int, float], names: List[str]) -> Optional[str]:
    """Updates the dropdown selection based on slider index."""
    if names and 0 <= int(idx) < len(names):
        return names[int(idx)]
    return None


with gr.Blocks(title="SfM Evaluation Suite") as demo:
    gr.Markdown("# 3D Reconstruction Evaluator")
    camera_names_state = gr.State([])

    # Store Parser objects to avoid reloading on every dropdown change
    gt_parser_state = gr.State(None)
    pred_parser_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            pred_input = gr.Textbox(
                label="Prediction Path",
                value="../vggt/vggt_outputs/bonsai_2_n100_s42_c1.0",
            )
            gt_input = gr.Textbox(
                label="Ground Truth Path",
                value="../vggt/colmap_outputs/bonsai_2_n100_s42_c5.0",
                # value="./data/360_v2/bonsai/sparse/0",
            )

            with gr.Row():
                chk_gt = gr.Checkbox(label="Show GT Cloud", value=True)
                chk_pred = gr.Checkbox(label="Show Pred Cloud", value=True)

            radio_color = gr.Radio(
                choices=["RGB", "Confidence"], label="Pred Point Color", value="RGB", interactive=True
            )

            btn = gr.Button("Evaluate & Visualize", variant="primary")

            camera_slider = gr.Slider(label="Scroll Cameras", minimum=0, maximum=1, step=1, visible=False)
            camera_dropdown = gr.Dropdown(label="Select Camera to Project", choices=[], allow_custom_value=True)

            output_metrics = gr.Markdown("Results will appear here...")
            output_json = gr.JSON(label="Full Metrics JSON")
            img_info = gr.Markdown("")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="3D Trajectory Comparison")
            with gr.Row():
                with gr.Column():
                    gt_img_display = gr.Image(label="GT Projection")
                with gr.Column():
                    pred_img_display = gr.Image(label="Pred Projection")

    btn.click(
        fn=run_gradio_eval_with_names,
        inputs=[pred_input, gt_input, chk_gt, chk_pred, radio_color],
        outputs=[
            output_metrics,
            plot_output,
            output_json,
            camera_dropdown,
            camera_slider,
            camera_names_state,
            gt_parser_state,
            pred_parser_state,
        ],
    )

    camera_slider.change(
        fn=sync_slider_to_dropdown, inputs=[camera_slider, camera_names_state], outputs=[camera_dropdown]
    )

    camera_dropdown.change(
        fn=update_projection_comparison,
        inputs=[camera_dropdown, gt_parser_state, pred_parser_state],
        outputs=[gt_img_display, pred_img_display, img_info],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
