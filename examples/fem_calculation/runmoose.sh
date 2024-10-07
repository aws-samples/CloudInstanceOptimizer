#!/bin/bash


# Function to check if a loop device exists
loop_exists() {
  [ -e "/dev/loop$1" ]
}

# Find the first available loop device number
LOOP_NUM=0
while loop_exists $LOOP_NUM; do
  LOOP_NUM=$((LOOP_NUM + 1))
done

# Create the new loop device
mknod -m 0660 /dev/loop$LOOP_NUM b 7 $LOOP_NUM

# Set the ownership to root:disk
chown root:disk /dev/loop$LOOP_NUM

echo "Loop device /dev/loop$LOOP_NUM has been created successfully"

# Optional: List all loop devices to verify
losetup -a


singularity exec -B /:/mnt moose.sif /mnt/runscript.sh $@