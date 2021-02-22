from dataops.netapp_ops import create_snapshot
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_snapshot", default=True, type=lambda x: (str(x).lower() == "true"))
    parser.add_argument("--pvc_name", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    runai_job_uuid = os.environ['JOB_UUID']
    create_snapshot(runai_job_uuid, args.pvc_name)