from kubernetes import client, config


def create_snapshot(runai_job_uuid, pvc_name, volume_snapshot_class_name="csi-snapclass"):
    # Loading k8s config for accesing APIs
    config.load_incluster_config()
    api = client.CustomObjectsApi()


    # Snapshot name is linked with RUNAI job id for trackability
    snapshot_name = "snap" + "-" + pvc_name + "-" + runai_job_uuid
    persistent_volume_claim_name = pvc_name

    # Fetch current namespace
    current_namespace = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read()

    # it's volume snapshot resource defined as Dict
    snapshot_resource = {
        "apiVersion": "snapshot.storage.k8s.io/v1beta1",
        "kind": "VolumeSnapshot",
        "metadata": {"name": snapshot_name},
        "spec": {
 #           "volumeSnapshotClassName": volume_snapshot_class_name,
            "source": {"persistentVolumeClaimName": persistent_volume_claim_name}
        }
    }

    # creating snapshot
    api.create_namespaced_custom_object(
        group="snapshot.storage.k8s.io",
        version="v1beta1",
        namespace=current_namespace,
        plural="volumesnapshots",
        body=snapshot_resource,
    )
    print("Snapshot " + snapshot_name + "created in namespace " + current_namespace)


