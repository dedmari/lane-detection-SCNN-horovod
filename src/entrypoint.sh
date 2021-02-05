#!/bin/sh

if [ -z "$RUNAI_MPI_NUM_WORKERS" ]; then
    echo "Environment variable 'RUNAI_MPI_NUM_WORKERS' does not exist"
    exit 1
fi

echo "Running with $RUNAI_MPI_NUM_WORKERS workers"

# Starting distributed training using Horovod
horovodrun -np $RUNAI_MPI_NUM_WORKERS -hostfile /etc/mpi/hostfile  python train.py --exp_dir ./experiments/exp0