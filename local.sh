#!/bin/bash


# Directory for logs
log_dir="log_experiments"

# Check if the directory exists; if not, create it
if [ ! -d "$log_dir" ]; then
  mkdir "$log_dir"
fi

# Get the current timestamp in format Year-Month-Day_Hour-Minute-Second
timestamp=$(date +"%Y%m%d_%H%M%S")

# Run the Python script and log the output to a file in the log_experiments directory
python task_training.py &> "$log_dir/log_$timestamp.txt"