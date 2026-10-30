import re
from pathlib import Path


def update_experiment_names(dry_run=True):
    # 1. Configuration
    src_dir = Path("../gsplat/results/colmap_outputs")

    # 2. Pre-flight checks
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Error: Source directory '{src_dir}' does not exist.")
        return

    print(f"--- Starting update operation (Dry run: {dry_run}) ---")

    # Regex patterns for matching
    skip_pattern = re.compile(r"_steps\d+$")
    file_pattern = re.compile(r"val_step(\d+)\.json$")

    # 3. Iterate through folders and rename if conditions are met
    renamed_count = 0
    for folder in src_dir.iterdir():
        if folder.is_dir():
            # Skip if the folder already ends with _steps{n}
            if skip_pattern.search(folder.name):
                continue

            stats_dir = folder / "stats"
            if not stats_dir.exists() or not stats_dir.is_dir():
                continue

            highest_step = -1

            # Look for all files matching val_step*.json in the stats folder
            for stat_file in stats_dir.glob("val_step*.json"):
                match = file_pattern.search(stat_file.name)
                if match:
                    step_num = int(match.group(1))
                    if step_num > highest_step:
                        highest_step = step_num

            # If we found at least one matching file
            if highest_step != -1:
                new_step = highest_step + 1
                new_folder_name = f"{folder.name}_steps{new_step}"
                destination_path = src_dir / new_folder_name

                if dry_run:
                    print(f"[DRY RUN] Would rename '{folder.name}' -> '{new_folder_name}'")
                    renamed_count += 1
                else:
                    print(f"Renaming '{folder.name}' -> '{new_folder_name}'...")
                    try:
                        folder.rename(destination_path)
                        renamed_count += 1
                    except Exception as e:
                        print(f"Failed to rename '{folder.name}': {e}")

    if dry_run:
        print(f"\nDry run complete. Found {renamed_count} folder(s) to rename.")
        print("To apply changes, run the function with dry_run=False.")
    else:
        print(f"\nOperation complete. Renamed {renamed_count} folder(s).")


if __name__ == "__main__":
    # Change dry_run to False to execute the actual renaming
    update_experiment_names(dry_run=False)
