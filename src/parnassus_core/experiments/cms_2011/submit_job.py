import os
import argparse

ncpus = "6"
ngpus = "1"
mem = "130gb"


parser = argparse.ArgumentParser()
parser.add_argument("--eval", action="store_true")
parser.add_argument("--name", type=str, default="flow_cms")
parser.add_argument("--comet", type=str)
args = parser.parse_args()


if args.eval:
    walltime = "32:00:00"
    ncpus = "8"
    mem = "30gb"
else:
    walltime = "72:00:00"

command = f"qsub -o {os.getcwd()}/logs/output.log"
command += f" -e {os.getcwd()}/logs/error.log"
if args.comet:
    command += f" -v COMET_EXP_ID={args.comet}"
# command += f" -v COMET_EXP_ID=51fb246d9dba4805a95987e49e8d053e"
command += f" -q gpu -N {args.name} -l walltime={walltime},mem={mem},ncpus={ncpus},ngpus={ngpus},io=10"
if args.eval:
    command += f" {os.getcwd()}/eval_on_node.sh"
else:
    command += f" {os.getcwd()}/run_on_node.sh"

print(command)
os.system(command)
