import os
from pathlib import Path
import sys
import argparse
import glob
import time
import yaml
import yaml_include

yaml.add_constructor("!include", yaml_include.Constructor(base_dir="."))

# CMS_TO_ROOT is assumed to be available in PATH or conda environment
COMMAND_DIR = Path(__file__).parent.parent.resolve()
CMS_TO_ROOT_CMD = "cms_to_root.py"


def submit_jobs(config, tag="tmp", dry_run=False):
    """Submit PBS jobs based on YAML configuration.

    Args:
        config: Dictionary loaded from YAML with keys:
            - source_top_dir: Base directory for input files
            - target_top_dir: Base directory for output files
            - files: Nested dict mapping parent_group -> subgroup -> list of files
            - processing_params: Dict with pt_cut, eta, num_events
            - pbs_params: Dict with walltime, mem, io, queue
            - log_dir: (optional) Directory for logs, default './pbs_logs'
        tag: Tag for organizing job logs and scripts
        dry_run: If True, only print commands without executing

    Returns:
        List of job commands submitted
    """
    source_top_dir = config["source_top_dir"]
    target_top_dir = config["target_top_dir"]
    file_groups = config["files"]
    processing_params = config.get("processing_params", {})
    pbs_params = config.get("pbs_params", {})

    # Get processing parameters with defaults
    pt_cut = processing_params.get("pt_cut", 0.0)
    eta = processing_params.get("eta", 2.7)
    num_events = processing_params.get("num_events", -1)

    # Get PBS parameters with defaults
    walltime = pbs_params.get("walltime", "4:00:00")
    mem = pbs_params.get("mem", "2g")
    io = pbs_params.get("io", "5")
    queue = pbs_params.get("queue", None)
    limit = f"walltime={walltime},mem={mem},io={io}"

    # Get log directory with default
    log_dir = config.get("log_dir", "./pbs_logs")

    all_jobs = []

    # Iterate over parent groups and subgroups
    for parent_group, subgroups in file_groups.items():
        for subgroup_name, file_list in subgroups.items():
            # Construct full group path: parent_group/subgroup_name
            full_group_path = os.path.join(parent_group, subgroup_name)

            # Sanitize group name for log/script file names (replace / with _)
            group_name_safe = full_group_path.replace("/", "_")

            # Create log and script directories for this group (using sanitized name)
            group_log_dir = os.path.join(log_dir, tag, group_name_safe)
            os.makedirs(group_log_dir, exist_ok=True)

            # Clean up old logs and scripts for this group
            os.system(f"rm -rf {group_log_dir}/*.log")
            os.system(f"rm -rf {group_log_dir}/*.sh")

            # Create output directory for this group (preserve nested structure)
            out_dir = os.path.join(target_top_dir, full_group_path)
            os.makedirs(out_dir, exist_ok=True)

            # Process each file in the subgroup
            for jobID, file_rel_path in enumerate(file_list):
                # Construct full input path
                input_file = os.path.join(source_top_dir, file_rel_path)

                # Extract filename and construct output path
                filename = os.path.basename(file_rel_path)
                output_filename = filename.replace(".root", "_conv.root")
                output_file = os.path.join(out_dir, output_filename)

                # Define log files
                out_log = os.path.join(group_log_dir, f"out_{jobID}.log")
                err_log = os.path.join(group_log_dir, f"err_{jobID}.log")

                # Clean up old output and logs
                os.system(f"rm -rf {out_log}")
                os.system(f"rm -rf {err_log}")
                os.system(f"rm -rf {output_file}")

                # Write job script
                run_script = os.path.join(group_log_dir, f"run_conv_{jobID}.sh")
                with open(run_script, "w") as f:
                    f.write(f"cd {COMMAND_DIR}\n")
                    f.write(
                        "source /usr/wipp/conda/24.5.0/bin/activate /usr/wipp/conda/24.5.0u/envs/common\n"
                    )
                    f.write(
                        f"python {CMS_TO_ROOT_CMD} -i {input_file} -o {output_file} "
                        f"-n {num_events} -pt {pt_cut} -eta {eta}\n"
                    )

                # Construct qsub command
                job = f"qsub -N {group_name_safe}_{jobID} -l {limit} -o {out_log} -e {err_log} -v jobID={jobID}"
                if queue is not None:
                    job += f" -q {queue} "
                job += f" {run_script}"

                all_jobs.append(job)
                print(job)

                if not dry_run:
                    os.system(job)
                    time.sleep(0.2)

    return all_jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit CMS-to-ROOT conversion jobs to PBS batch scheduler using YAML configuration"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        help="Dry run (print commands without executing)",
        action="store_true",
    )
    parser.add_argument(
        "-t", "--tag", type=str, help="Tag for the job directory", default="tmp"
    )
    parser.add_argument("-r", "--rerun", action="store_true", help="Rerun failed jobs")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run mode: commands will be printed but not executed.\n")

    # Load YAML configuration
    with open(args.config, "r") as f:
        config = yaml.full_load(f)

    # Validate required fields
    required_fields = ["source_top_dir", "target_top_dir", "files"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from config file")

    tag = args.tag
    log_dir = config.get("log_dir", "./pbs_logs")

    if args.rerun:
        # Rerun failed jobs
        jobs_file = f"jobs_{tag}.txt"
        if not os.path.exists(jobs_file):
            print(f"Error: {jobs_file} not found. Cannot rerun jobs.")
            sys.exit(1)

        with open(jobs_file, "r") as f:
            jobs = f.readlines()

        flag = True
        while flag:
            time.sleep(20)
            # Find all log files across all groups
            log_pattern = os.path.join(log_dir, tag, "**/out_*.log")
            out_files = glob.glob(log_pattern, recursive=True)
            jobs_to_rerun = []

            for out_file in out_files:
                if os.path.exists(out_file) and os.stat(out_file).st_size == 0:
                    # Extract job index from the jobs list based on the script path
                    err_file = out_file.replace("out_", "err_")
                    print(f"Job failed: {out_file}")
                    os.system(f"rm -rf {out_file}")
                    os.system(f"rm -rf {err_file}")

                    # Find corresponding job command
                    # This is a simplified approach - match by script path
                    script_path = out_file.replace("/out_", "/run_conv_").replace(
                        ".log", ".sh"
                    )
                    for idx, job in enumerate(jobs):
                        if script_path in job:
                            jobs_to_rerun.append((idx, job.strip()))
                            break

            if len(jobs_to_rerun) > 0:
                print(f"Rerunning {len(jobs_to_rerun)} failed jobs")
                for idx, job in jobs_to_rerun:
                    print(f"Rerunning: {job}")
                    os.system(job)
                    time.sleep(0.2)
            else:
                print("All jobs completed successfully")
                flag = False
    else:
        # Submit new jobs
        jobs = submit_jobs(config, tag=tag, dry_run=args.dry_run)

        # Save job commands for potential rerun
        jobs_file = f"jobs_{tag}.txt"
        with open(jobs_file, "w") as f:
            for job in jobs:
                f.write(job + "\n")

        print(f"\nSubmitted {len(jobs)} jobs. Job commands saved to {jobs_file}")
