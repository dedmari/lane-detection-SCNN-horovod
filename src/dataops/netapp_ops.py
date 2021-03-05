from kubernetes import client, config
import datetime


def create_snapshot(runai_job_name, pvc_name, volume_snapshot_class_name="csi-snapclass"):
    # Loading k8s config for accessing APIs
    config.load_incluster_config()
    api = client.CustomObjectsApi()

    # Snapshot name is linked with RUNAI job name and include timestamp for trackability
    snapshot_name = "snap" + "-" + pvc_name + "-" + runai_job_name + "-" + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H-%M-%S")
    persistent_volume_claim_name = pvc_name

    # Fetch current namespace belonging to the runai project
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

    # creating NetApp snapshot of pvc
    try:
        api.create_namespaced_custom_object(
            group="snapshot.storage.k8s.io",
            version="v1beta1",
            namespace=current_namespace,
            plural="volumesnapshots",
            body=snapshot_resource,
        )
        print("Snapshot " + snapshot_name + " created in namespace " + current_namespace)
    except Exception as e:
        print('Exception: ' + str(e))
