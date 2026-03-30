#!/bin/bash

# Define source and destination directories
src_base="/home/youngju/ssd/datasets/DTU_TEST"
dest_base="/home/youngju/ssd/UFORecon-project-page/static/images/source_views"

# Define the list of scan IDs
scan_ids=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
img_ids=(1 16 36)  # These are the original image IDs

# Mapping for destination filenames: 1.png, 2.png, 3.png
declare -A dest_names=( [1]=1 [16]=2 [36]=3 )

# Loop over each scan ID
for id in "${scan_ids[@]}"; do
    for img_id in "${img_ids[@]}"; do
        # Format the source file with leading zeros
        formatted_src_id=$(printf "%06d" "$img_id")
        src_file="${src_base}/scan${id}/image/${formatted_src_id}.png"
        # Use the mapping for destination filename
        dest_file="${dest_base}/${id}/${dest_names[$img_id]}.png"

        # Check if the destination file already exists
        if [ ! -f "$dest_file" ]; then
            # Ensure the destination directory exists
            mkdir -p "$(dirname "$dest_file")"

            # Copy the file from source to destination
            cp "$src_file" "$dest_file"
            echo "Copied $src_file to $dest_file"
        else
            echo "File $dest_file already exists, skipping..."
        fi
    done
done
