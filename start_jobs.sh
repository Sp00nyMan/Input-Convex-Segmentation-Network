#!/bin/bash

sbatch --output=log_plain.txt job_script.sh plain
sbatch --output=log_flow.txt job_script.sh flow
# sbatch --output=log_convex.txt job_script.sh convex