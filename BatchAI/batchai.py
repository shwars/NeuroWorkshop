from __future__ import print_function
from datetime import datetime
import os
import sys
import zipfile
from azure.storage.file import FileService
from azure.storage.blob import BlockBlobService
import azure.mgmt.batchai.models as models
import azure.mgmt.batchai as batchai
from azure.common.credentials import ServicePrincipalCredentials

tenant = "<Insert Correct Value Here>"
subscription = "<Insert Correct Value Here>"
resource_group_name = "batchai"

credentials = ServicePrincipalCredentials(client_id="<Insert Correct Value Here>",
                                          secret="<Insert Correct Value Here>",
                                          token_uri="https://login.microsoftonline.com/{0}/oauth2/token".format(tenant))
client = batchai.BatchAIManagementClient(
    credentials=credentials,
    subscription_id=subscription,
    base_url="")

from azure.mgmt.resource import ResourceManagementClient

resource_management_client = ResourceManagementClient(credentials=credentials, subscription_id=subscription)

group = resource_management_client.resource_groups.create_or_update(
        resource_group_name, {'location': 'northeurope'})

from azure.storage.file import FileService

storage_account_name = "<Insert Correct Value Here>"
storage_account_key = "<Insert Correct Value Here>"
fileshare = "data"

filesystem = FileService(storage_account_name, storage_account_key)

for f in ['Train-28x28_cntk_text.txt', 'Test-28x28_cntk_text.txt', 'ConvNet_MNIST.py']:
  filesystem.create_file_from_path(fileshare, "data", f, "z:/script/"+f)

## Create Cluster

cluster_name = 'shwarscluster'
relative_mount_point = 'azurefileshare'

parameters = models.ClusterCreateParameters(
    location='northeurope',
    vm_size='STANDARD_NC6',
    user_account_settings=models.UserAccountSettings(
         admin_user_name="shwars",
         admin_user_password="ShwarZ13!"),
    scale_settings=models.ScaleSettings(
         manual=models.ManualScaleSettings(target_node_count=1)
     ),
    node_setup=models.NodeSetup(
        # Mount shared volumes to the host
         mount_volumes=models.MountVolumes(
             azure_file_shares=[
                 models.AzureFileShareReference(
                     account_name=storage_account_name,
                     credentials=models.AzureStorageCredentialsInfo(
         account_key=storage_account_key),
         azure_file_url='https://{0}.file.core.windows.net/{1}'.format(
               storage_account_name, fileshare),
                  relative_mount_path = relative_mount_point)],
         ),
    ),
)

client.clusters.create(resource_group_name, cluster_name, parameters).result()

cluster = client.clusters.get(resource_group_name, cluster_name)
print('Cluster state: {0} Target: {1}; Allocated: {2}; Idle: {3}; '
      'Unusable: {4}; Running: {5}; Preparing: {6}; leaving: {7}'.format(
    cluster.allocation_state,
    cluster.scale_settings.manual.target_node_count,
    cluster.current_node_count,
    cluster.node_state_counts.idle_node_count,
    cluster.node_state_counts.unusable_node_count,
    cluster.node_state_counts.running_node_count,
    cluster.node_state_counts.preparing_node_count,
    cluster.node_state_counts.leaving_node_count))

## CREATE JOB

job_name = 'trainjob'

parameters = models.job_create_parameters.JobCreateParameters(
     location='northeurope',
     cluster=models.ResourceId(id=cluster.id),
     # The number of VMs in the cluster to use
     node_count=1,

     # Override the path where the std out and std err files will be written to.
     # In this case we will write these out to an Azure Files share
     std_out_err_path_prefix='$AZ_BATCHAI_MOUNT_ROOT/{0}'.format(relative_mount_point),

     input_directories=[models.InputDirectory(
         id='SAMPLE',
         path='$AZ_BATCHAI_MOUNT_ROOT/{0}/data'.format(relative_mount_point))],

     # Specify directories where files will get written to
     output_directories=[models.OutputDirectory(
        id='MODEL',
        path_prefix='$AZ_BATCHAI_MOUNT_ROOT/{0}'.format(relative_mount_point),
        path_suffix="Models")],

     # Container configuration
     container_settings=models.ContainerSettings(
         image_source_registry=models.ImageSourceRegistry(image='microsoft/cntk:2.1-gpu-python3.5-cuda8.0-cudnn6.0')),

     # Toolkit specific settings
     cntk_settings = models.CNTKsettings(
        python_script_file_path='$AZ_BATCHAI_INPUT_SAMPLE/ConvNet_MNIST.py',
        command_line_args='$AZ_BATCHAI_INPUT_SAMPLE $AZ_BATCHAI_OUTPUT_MODEL')
 )

# Create the job
client.jobs.create(resource_group_name, job_name, parameters).result()


## MONITOR JOB
job = client.jobs.get(resource_group_name, job_name)

print('Job state: {0} '.format(job.execution_state.name))


files = client.jobs.list_output_files(resource_group_name, job_name, models.JobsListOutputFilesOptions(outputdirectoryid="stdouterr"))

for file in list(files):
     print('file: {0}, download url: {1}'.format(file.name, file.download_url))

client.jobs.delete(resource_group_name, job_name)
client.clusters.delete(resource_group_name, cluster_name)