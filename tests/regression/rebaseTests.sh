#!/bin/bash

# Function to replace files
replace_files() {
    local dir="${1%/}" # Remove trailing slash
    local base_dir="$dir/base"
    local data_out_dir="$dir/data_out"

    # Check if base and data_out directories exist
    if [[ ! -d "$base_dir" || ! -d "$data_out_dir" ]]; then
        echo "Skipping $dir: base or data_out directory does not exist."
        return
    fi

    # Replace files
    for file in "$base_dir"/*; do
        filename=$(basename "$file")
        if [[ -f "$data_out_dir/$filename" ]]; then
            cp "$data_out_dir/$filename" "$base_dir/"
            echo "Replaced $base_dir/$filename with version from $data_out_dir."
        else
            echo "No replacement found for $filename in $data_out_dir."
        fi
    done
}

# Main script logic
if [[ $# -eq 0 ]]; then
    # No arguments, apply to all subdirectories
    for dir in */; do
        replace_files "$dir"
    done
else
    # Apply to specified subdirectory
    for dir in "$@"; do
        if [[ -d "$dir" ]]; then
            replace_files "$dir"
        else
            echo "Directory $dir does not exist."
        fi
    done
fi
