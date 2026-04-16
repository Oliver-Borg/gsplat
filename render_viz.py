import re
from pathlib import Path
from collections import defaultdict
import dataclasses

import gradio as gr

from experiment_runner import experiments, datasets, experiment_dict, Config


def get_step_renders(renders_folder: str, target_step: int) -> dict[str, Path]:
    """
    Finds all renders for the specified validation step in the given folder.
    Returns a dictionary mapping the image index to its file path.
    """
    p = Path(renders_folder)
    if not p.exists() or target_step is None:
        return {}

    step_files = list(p.glob(f"val_step{target_step}_*.jpg"))

    renders = {}
    for f in step_files:
        # Extract the image index (e.g. '0027' from 'val_step0_0027_psnr...')
        match = re.search(r"val_step\d+_(\d+)_", f.name)
        if match:
            idx = match.group(1)
            renders[idx] = f

    return renders


def update_configs(exp_name: str, ds_name: str):
    """
    Called when the Experiment or Dataset dropdown changes.
    Groups runs by their varying parameters (excluding choice and seed).
    """
    if not exp_name or not ds_name:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            {},
        )

    exp = experiment_dict[exp_name]

    # Identify which parameters vary, pulling 'choice' and 'seed' out of the signature
    varying_keys = [k for k, v in exp.config_dict.items() if isinstance(v, list) and k not in ["choice", "seed"]]

    # We enforce include_gt=True here to fetch the pseudo GT models if they exist
    exp_gt = dataclasses.replace(exp, include_gt=True)
    configs = exp_gt.get_configs(ds_name)

    # Dictionary format: choice -> signature -> seed -> config
    groups = defaultdict(lambda: defaultdict(dict))
    seeds = set()

    for config in configs:
        if not config.is_splatted:
            continue

        sig_parts = [f"{k}={getattr(config, k)}" for k in sorted(varying_keys) if hasattr(config, k)]
        sig = " | ".join(sig_parts) if sig_parts else "Default"

        # Group by the unique signature, storing the seed map
        groups[config.choice][sig][config.seed] = config
        seeds.add(config.seed)

    vggt_choices = list(groups["vggt"].keys())
    vggt_val = vggt_choices[0] if vggt_choices else None

    colmap_choices = list(groups["colmap"].keys())
    colmap_val = colmap_choices[0] if colmap_choices else None

    # Use a sorted list of valid seeds for the dropdown
    seed_choices = sorted(list(seeds))
    seed_val = seed_choices[0] if seed_choices else None

    # Convert defaultdict to standard dict for Gradio state serialization
    state_groups = {
        "vggt": {k: dict(v) for k, v in groups["vggt"].items()},
        "colmap": {k: dict(v) for k, v in groups["colmap"].items()},
        "gt": {k: dict(v) for k, v in groups["gt"].items()},
    }

    return (
        gr.update(choices=vggt_choices, value=vggt_val),
        gr.update(choices=colmap_choices, value=colmap_val),
        gr.update(choices=seed_choices, value=seed_val),
        state_groups,
    )


def update_steps(vggt_sig: str, colmap_sig: str, seed_val: int, groups: dict):
    """
    Triggered when config changes. Finds all available validation steps across selected configs.
    """
    if not groups or seed_val is None:
        return gr.update(choices=[], value=None)

    vggt_config = groups.get("vggt", {}).get(vggt_sig, {}).get(seed_val)
    colmap_config = groups.get("colmap", {}).get(colmap_sig, {}).get(seed_val)

    gt_sigs = list(groups.get("gt", {}).keys())
    gt_config = groups.get("gt", {}).get(gt_sigs[0], {}).get(seed_val) if gt_sigs else None

    steps = set()
    for config in [vggt_config, colmap_config, gt_config]:
        if config:
            p = Path(config.renders_folder)
            if p.exists():
                for f in p.glob("val_step*.jpg"):
                    match = re.search(r"val_step(\d+)_", f.name)
                    if match:
                        steps.add(int(match.group(1)))

    step_choices = sorted(list(steps))
    step_val = (14999 if 14999 in step_choices else max(step_choices)) if step_choices else None
    return gr.update(choices=step_choices, value=step_val)


def on_config_change(vggt_sig: str, colmap_sig: str, seed_val: int, step_val: int, groups: dict):
    """
    Triggered when the step dropdown changes.
    Scans the disk ONCE to pre-load all image paths for the slider to consume instantly.
    """
    if not groups or seed_val is None or step_val is None:
        return gr.update(maximum=0, value=0), [], [], None, "N/A", None, "N/A", None, "N/A"

    # Fetch configurations for the specifically requested seed
    vggt_config = groups.get("vggt", {}).get(vggt_sig, {}).get(seed_val)
    colmap_config = groups.get("colmap", {}).get(colmap_sig, {}).get(seed_val)

    gt_sigs = list(groups.get("gt", {}).keys())
    gt_config = groups.get("gt", {}).get(gt_sigs[0], {}).get(seed_val) if gt_sigs else None

    all_indices = set()
    group_renders = {}

    # Read the file paths from the disk once
    for choice, config in [("vggt", vggt_config), ("colmap", colmap_config), ("gt", gt_config)]:
        if config:
            renders = get_step_renders(config.renders_folder, step_val)
            group_renders[choice] = renders
            all_indices.update(renders.keys())
        else:
            group_renders[choice] = {}

    sorted_indices = sorted(list(all_indices))

    if not sorted_indices:
        return (
            gr.update(maximum=0, value=0),
            [],
            [],
            None,
            "No renders found",
            None,
            "No renders found",
            None,
            "No renders found",
        )

    # PRE-LOAD LOOP: Build a cache of all images and metrics for every slider step
    preloaded_frames = []
    for idx_str in sorted_indices:
        frame_data = []
        for choice in ["gt", "vggt", "colmap"]:
            renders = group_renders.get(choice, {})
            if idx_str in renders:
                img_path = renders[idx_str]
                match = re.search(r"psnr([0-9.]+)_lpips([0-9.]+)\.jpg", img_path.name)
                metrics = (
                    f"PSNR: {match.group(1)} | LPIPS: {match.group(2)}" if match else "Metrics missing from filename"
                )
                frame_data.extend([str(img_path), metrics])
            else:
                frame_data.extend([None, "N/A"])
        preloaded_frames.append(frame_data)

    first_frame = preloaded_frames[0]

    # Return slider update, state updates, and the first frame's images/metrics
    return [gr.update(maximum=len(sorted_indices) - 1, value=0), sorted_indices, preloaded_frames] + first_frame


def on_slider_change(slider_val: int, preloaded_frames: list):
    """
    Triggered when scrubbing the slider.
    Does zero logic—just fetches the pre-loaded image paths from memory.
    """
    if not preloaded_frames or slider_val >= len(preloaded_frames):
        return [None, "N/A"] * 3

    return preloaded_frames[slider_val]


with gr.Blocks(title="3DGS Experiment Results Viewer", fill_width=True) as app:
    gr.Markdown("# 🔍 Experiment Render Visualizer")

    with gr.Row():
        exp_dropdown = gr.Dropdown(choices=list(experiment_dict.keys()), label="1. Select Experiment")
        ds_dropdown = gr.Dropdown(choices=list(datasets.keys()), label="2. Select Dataset")
        seed_dropdown = gr.Dropdown(choices=[], label="3. Select Seed")

    with gr.Row():
        vggt_dropdown = gr.Dropdown(choices=[], label="4. Select VGGT Config")
        colmap_dropdown = gr.Dropdown(choices=[], label="5. Select COLMAP Config")
        step_dropdown = gr.Dropdown(choices=[], label="6. Select Validation Step")

    index_slider = gr.Slider(minimum=0, maximum=0, step=1, label="Image Index (Scrub through camera poses)")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### VGGT")
            vggt_img = gr.Image(label="VGGT", interactive=False)
            vggt_metrics = gr.Textbox(label="VGGT Metrics")

            gr.Markdown("---")

            gr.Markdown("### COLMAP")
            colmap_img = gr.Image(label="COLMAP", interactive=False)
            colmap_metrics = gr.Textbox(label="COLMAP Metrics")

            gr.Markdown("---")

            gr.Markdown("### Ground Truth (GT)")
            gt_img = gr.Image(label="GT", interactive=False)
            gt_metrics = gr.Textbox(label="GT Metrics")

    # Hidden states to store the current logic and cached images
    current_groups = gr.State({})
    current_indices = gr.State([])
    current_frames = gr.State([])

    # Event wiring
    exp_dropdown.change(
        fn=update_configs,
        inputs=[exp_dropdown, ds_dropdown],
        outputs=[vggt_dropdown, colmap_dropdown, seed_dropdown, current_groups],
    )

    ds_dropdown.change(
        fn=update_configs,
        inputs=[exp_dropdown, ds_dropdown],
        outputs=[vggt_dropdown, colmap_dropdown, seed_dropdown, current_groups],
    )

    # Link configuration UI directly to update steps first
    for ui_element in [vggt_dropdown, colmap_dropdown, seed_dropdown]:
        ui_element.change(
            fn=update_steps,
            inputs=[vggt_dropdown, colmap_dropdown, seed_dropdown, current_groups],
            outputs=[step_dropdown],
        )

    # Changing any dropdown triggers reading images and logic cache
    for ui_element in [vggt_dropdown, colmap_dropdown, seed_dropdown, step_dropdown]:
        ui_element.change(
            fn=on_config_change,
            inputs=[vggt_dropdown, colmap_dropdown, seed_dropdown, step_dropdown, current_groups],
            outputs=[
                index_slider,
                current_indices,
                current_frames,
                gt_img,
                gt_metrics,
                vggt_img,
                vggt_metrics,
                colmap_img,
                colmap_metrics,
            ],
        )

    # Slider only interacts with the cached frames for maximum speed
    index_slider.change(
        fn=on_slider_change,
        inputs=[index_slider, current_frames],
        outputs=[gt_img, gt_metrics, vggt_img, vggt_metrics, colmap_img, colmap_metrics],
        show_progress="hidden",
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7861, share=True)
