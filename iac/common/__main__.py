import os
import sys

# Determine the absolute path to the 'iac' directory
libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add this directory to sys.path,
# so that we can import the iac/lib module when we run pulumi from withing the iac/common directory.
sys.path.insert(0, libs_dir)

import pulumi
from deploy_common import deploy_common

from lib.std_pulumi import getconfig, getstackref, parse_realm_env_name_from_stack, load_dot_realm_env


def main():
    realm_name, environment_name, stack_name = parse_realm_env_name_from_stack()

    # Load environment variables.
    load_dot_realm_env(stack_name)

    # Get the config values
    location = getconfig("region", "gcp")

    # Get stack reference for environment
    env_reference = pulumi.StackReference(f"tabiya-tech/compass-environment/{stack_name}")
    project = getstackref(env_reference, "project_id")

    frontend_domain = getstackref(env_reference, "frontend_domain")
    frontend_url = getstackref(env_reference, "frontend_url")
    backend_url = getstackref(env_reference, "backend_url")

    # Get stack reference for dns
    dns_stack_ref = pulumi.StackReference(f"tabiya-tech/compass-dns/{stack_name}")
    dns_zone_name = getstackref(dns_stack_ref, "dns_zone_name")

    # Get stack reference for frontend
    frontend_stack_ref = pulumi.StackReference(f"tabiya-tech/compass-frontend/{stack_name}")
    frontend_bucket_name = getstackref(frontend_stack_ref, "bucket_name")
    frontend_bucket_name.apply(lambda name: print(f"Using frontend bucket name: {name}"))

    # Get the ESPv2 Cloud Run service from the backend stack.
    backend_stack_ref = pulumi.StackReference(f"tabiya-tech/compass-backend/{stack_name}")
    espv2_cloud_run_name = getstackref(backend_stack_ref, "espv2_cloud_run_name")
    espv2_cloud_run_name.apply(lambda name: print(f"Using ESPv2 Cloud Run service: {name}"))

    # Look up the ESPv2 Cloud Run service resource by name so we can pass it to deploy_common.
    import pulumi_gcp as gcp
    espv2_cloudrun_service = gcp.cloudrunv2.Service.get(
        "espv2-gateway-lookup",
        id=pulumi.Output.all(project=project, location=location, name=espv2_cloud_run_name).apply(
            lambda args: f"projects/{args['project']}/locations/{args['location']}/services/{args['name']}"
        ),
    )

    # Deploy common
    deploy_common(
        project=project,
        location=location,
        dns_zone_name=dns_zone_name,
        frontend_domain=frontend_domain,
        frontend_bucket_name=frontend_bucket_name,
        frontend_url=frontend_url,
        backend_url=backend_url,
        espv2_cloudrun_service=espv2_cloudrun_service)


if __name__ == "__main__":
    main()
