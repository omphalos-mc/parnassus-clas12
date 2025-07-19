import os
import time
import yaml
import argparse
from pathlib import Path

top_dir = Path(__file__).parent.resolve()
pbs_dir = top_dir / "pbs"
if not pbs_dir.exists():
    pbs_dir.mkdir(parents=True)

ncpus = "6"
ngpus = "1"
mem = "130gb"


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="particle.yaml")
parser.add_argument("-g", "--gpu", type=str, choices=["", "A5000", "A6000"], default="")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--comet", type=str)
args = parser.parse_args()


if args.eval:
    walltime = "32:00:00"
    ncpus = "8"
    mem = "30gb"
else:
    walltime = "72:00:00"

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

with open(run_script, "w") as f:
    f.write("#!/bin/bash\n")
    if args.gpu:
        f.write(f"#PBS -l gputype={args.gpu}\n")
    f.write("IOTHROTTLE_LIMIT=0\n")
    f.write("export IOTHROTTLE_LIMIT=0\n")
    f.write("export IOTHROTTLE_VERBOSE=1\n")
    f.write(
        "source /usr/wipp/conda/24.5.0/bin/activate /usr/wipp/conda/24.5.0u/envs/common\n"
    )
    f.write(f"cd {top_dir}\n")
    f.write(f"python main.py fit -c {args.config}\n")

command = (
    f"qsub -o {log_file} -e {err_file} -N {name}"
    f" -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},io=5"
)
if args.comet:
    command += f" -v COMET_EXP_ID={args.comet}"
command += f" {run_script}"
print(command)

# command = f"qsub -o {os.getcwd()}/logs/output.log"
# command += f" -e {os.getcwd()}/logs/error.log"
# if args.comet:
#     command += f" -v COMET_EXP_ID={args.comet}"
# # command += f" -v COMET_EXP_ID=51fb246d9dba4805a95987e49e8d053e"
# command += f" -q gpu -N {args.name} -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},io=5"
# if args.eval:
#     command += f" {os.getcwd()}/eval_on_node.sh"
# else:
#     command += f" {os.getcwd()}/run_on_node.sh"

# print(command)
os.system(command)
