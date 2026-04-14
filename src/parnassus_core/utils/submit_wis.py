import os
import time
import yaml
import argparse
from pathlib import Path
import shutil

IO_LIMIT = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit a training or evaluation job to the WIS cluster using qsub."
    )
    parser.add_argument("-c", "--config", type=str, default="particle.yaml")
    parser.add_argument(
        "-g", "--gpu", type=str, choices=["", "A5000", "A6000"], default=""
    )
    parser.add_argument("-ng", "--num-gpu", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--comet", type=str)
    parser.add_argument(
        "-d", "--dry-run", action="store_true", help="Only print the command"
    )
    parser.add_argument(
        "--move-files-temp",
        action="store_true",
        help="Whether to move data files to a temp directory ($TMPDIR) before training. By default false.",
    )
    parser.add_argument(
        "--ckpt", type=str, help="Optional checkpoint path to resume training from"
    )
    parser.add_argument(
        "--mem",
        type=str,
        default="130gb",
        help="Memory allocation for the job (e.g. '130gb'). Default is '130gb' for training and '30gb' for evaluation.",
    )
    parser.add_argument(
        "--walltime",
        type=str,
        default=None,
        help="Walltime for the job (e.g. '72:00:00'). If not specified, defaults to '72:00:00' for training and '32:00:00' for evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ncpus = 6 * int(args.num_gpu) + 1
    ngpus = str(args.num_gpu)
    mem = str(args.mem)

    if args.eval:
        walltime = args.walltime if args.walltime else "32:00:00"
        ncpus = "8"
        mem = "30gb"
    else:
        walltime = args.walltime if args.walltime else "72:00:00"

    top_dir = Path.cwd().resolve()
    pbs_dir = top_dir / "pbs"
    if not pbs_dir.exists():
        pbs_dir.mkdir(parents=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        name = config["name"]
    name = name + time.strftime("_%Y%m%d_T%H%M") + ("_eval" if args.eval else "")
    job_dir = pbs_dir / name
    if not job_dir.exists():
        job_dir.mkdir(parents=True)
    log_file = job_dir / "output.log"
    err_file = job_dir / "error.log"

    run_script = job_dir / "run.sh"

    config_file = Path(args.config).resolve()
    shutil.copy(config_file, job_dir / "config.yaml")

    job_command = f"python main.py {'eval' if args.eval else 'fit'} -c {config_file}"
    if args.ckpt:
        job_command += f" --ckpt {args.ckpt}"
    if args.num_gpu > 1:
        job_command += f" --trainer.devices {args.num_gpu}"
    if args.move_files_temp:
        job_command += " --move_files_temp $TMPDIR"

    with open(run_script, "w") as f:
        f.write("#!/bin/bash\n")
        if args.gpu:
            f.write(f"#PBS -l gputype={args.gpu}\n")
        f.write(f"IOTHROTTLE_LIMIT={IO_LIMIT}\n")
        f.write(f"export IOTHROTTLE_LIMIT={IO_LIMIT}\n")
        f.write("export IOTHROTTLE_VERBOSE=1\n")
        f.write(
            "source /usr/wipp/conda/24.5.0/bin/activate /usr/wipp/conda/24.5.0u/envs/common\n"
        )
        f.write(f"cd {top_dir}\n")
        f.write(f"{job_command}\n")

    command = (
        f"qsub -o {log_file} -e {err_file} -N {name}"
        f" -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},io=1"
    )
    if args.comet:
        command += f" -v COMET_EXP_ID={args.comet}"
    command += f" {run_script}"
    if args.dry_run:
        print("Dry run, command to be executed:")
        print(command)
    else:
        print("Submitting job with command:")
        print(command)
        os.system(command)


if __name__ == "__main__":
    main()
