#!/bin/bash
# you might need to use absolute paths in LOG_DIR
# cd tgi-swarm/

# Set job type: 'vllm' or 'tgi' (default)
JOB_TYPE={{job_type}}
echo "Using $JOB_TYPE"
JOB_ID={{job_id}}
# Set job-specific variables
if [ "$JOB_TYPE" = "vllm" ]; then
    # try absolute path if this fails
    LOG_DIR="slurm/logs_vllm"
    HOSTS_FILE={{slurm_hosts_path}}
    CURL_DATA='{
        "prompt": "What is Life?",
        "n": 1,
    }'
else
    # try absolute path if this fails
    LOG_DIR="slurm/logs"
    HOSTS_FILE={{slurm_hosts_path}}
    CURL_DATA='{"inputs":"What is Life?","parameters":{"max_new_tokens":10}}'
fi

# Function to find the latest log file
find_logs() {
    local job_id="$1"
    find "$LOG_DIR" -type f -name "*_${job_id}.out" -printf '%T+ %p\n' | sort -r | head -n 1 | cut -d' ' -f2
}
# Function to extract hostname and port
extract_hostname_port() {
    local log_file=$1
    if [ "$JOB_TYPE" = "vllm" ]; then
        PORT=8000
        HOSTNAME=$(echo $(squeue -j $JOB_ID  -o "%N") | cut -d' ' -f2)
        echo "Job $JOB_ID running on $HOSTNAME"
        if [ -z "$HOSTNAME" ]; then
            echo "Hostname not found."
        else
            echo "Hostname: $HOSTNAME"
        fi
    else
        # Extracting the port number
        PORT=$(grep -oP ' port: \K\d+' "$log_file")
        if [ -z "$PORT" ]; then
            echo "Port not found in log file."
        else
            echo "Port: $PORT"
        fi

        # Extracting the hostname
        HOSTNAME=$(grep -m 1 "hostname:" "$log_file" | awk -F '\"' '{print $4}')
        if [ -z "$HOSTNAME" ]; then
            echo "Hostname not found in log file."
        else
            echo "Hostname: $HOSTNAME"
        fi
    fi
}

# Function to test the endpoint
test_endpoint() {
    local address=$1

    # Determine the appropriate curl command based on job type
    if [ "$JOB_TYPE" = "vllm" ]; then
        # Execute the curl command for vllm job type
        if curl -m 10 $address/generate \
            -d '{
                "prompt": "What is Life?",
                "n": 1,
                "temperature": 0.2
            }'; then
            echo -e "\nThe vLLM endpoint works 🎉!"
        else
            echo "curl command failed for vLLM."
        fi
    else
        # Execute the curl command for other job types (e.g., tgi)
        if curl -m 10 $address/generate \
            -X POST \
            -d '{"inputs":"What is Life?","parameters":{"max_new_tokens":10}}' \
            -H 'Content-Type: application/json'; then
            echo -e "\nThe TGI endpoint works 🎉!"
        else
            echo "curl command failed for TGI."
        fi
    fi

}
# Main script
LOG_FILE_PATH=$(find_logs $JOB_ID)
echo "Corresponding log file is $LOG_FILE_PATH"

extract_hostname_port "$LOG_FILE_PATH"

ADDRESS="http://$HOSTNAME:$PORT"
echo "Saving address $ADDRESS in $HOSTS_FILE"

touch "$HOSTS_FILE"
echo "$ADDRESS" >> "$HOSTS_FILE"

test_endpoint "$ADDRESS"

exit
