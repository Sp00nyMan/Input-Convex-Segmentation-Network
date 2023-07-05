#!/bin/bash

sh scripts/clear.sh
sbatch --output=log_plain.txt scripts/job_script.sh plain
sbatch --output=log_flow.txt scripts/job_script.sh flow
# sbatch --output=log_convex.txt job_script.sh convex