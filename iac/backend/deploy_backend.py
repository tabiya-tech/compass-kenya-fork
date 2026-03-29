import os
from dataclasses import dataclass
from typing import Optional

import pulumi
import pulumi_gcp as gcp

from pulumi import Output

from lib import ProjectBaseConfig, get_resource_name, get_project_base_config, Version
from scripts.formatters import construct_docker_tag
from backend.cv_bucket import _create_cv_upload_bucket, _grant_cloud_run_sa_access_to_cv_bucket
from espv2_image_builder import EspV2ImageBuilder
from _construct_espv2_cfg import construct_espv2_cfg


@dataclass(frozen=True)
class BackendServiceConfig:
    """
    Environment variables for the backend service
    See the backend service for more information on the environment variables.
    """
    taxonomy_mongodb_uri: str
    taxonomy_database_name: str
    taxonomy_model_id: str
    application_mongodb_uri: str
    application_database_name: str
    metrics_mongodb_uri: str
    metrics_database_name: str
    userdata_mongodb_uri: str
    userdata_database_name: str
    vertex_api_region: str
    embeddings_service_name: str
    embeddings_model_name: str
    target_environment_name: str
    target_environment_type: str | pulumi.Output[str]
    backend_url: str | pulumi.Output[str]
    frontend_url: str | pulumi.Output[str]
    sentry_dsn: str
    sentry_config: Optional[str]
    enable_sentry: str
    enable_metrics: str
    default_country_of_user: str
    gcp_oauth_client_id: str
    cloudrun_max_instance_request_concurrency: int
    cloudrun_min_instance_count: int
    cloudrun_max_instance_count: int
    cloudrun_request_timeout: str
    cloudrun_memory_limit: str
    cloudrun_cpu_limit: str
    features: Optional[str]
    experience_pipeline_config: Optional[str]
    cv_max_uploads_per_user: Optional[str]
    cv_rate_limit_per_minute: Optional[str]
    stream_chunk_size: Optional[str]
    stream_chunk_mode: Optional[str]
    stream_delta_delay_ms: Optional[str]
    enable_cv_upload: Optional[str]
    language_config: str
    global_product_name: Optional[str]
    matching_service_url: Optional[str]
    matching_service_api_key: Optional[str]
    inline_phase_transition: Optional[str]


"""
# Deploy an ESPv2 proxy on Cloud Run for the Compass backend.
#
# ESPv2 (Envoy-based) is a self-managed alternative to Apigee/API Gateway.
# Unlike API Gateway (which buffers full responses), ESPv2 passes bytes through
# without buffering — required for SSE/streaming endpoints.
#
# The proxy verifies Firebase JWTs and injects the X-Endpoint-API-UserInfo header
# (base64url-encoded JWT payload) so the backend can identify the authenticated user.
#
# Bootstrap challenge: ESPv2's image must have the Endpoints CONFIG_ID baked in.
# We solve this with a pulumi.dynamic.Resource (EspV2ImageBuilder in espv2_image_builder.py) that calls
# gcloud_build_image during create().
#
# Two-phase approach to break the circular dependency:
#   1. Deploy ESPv2 Cloud Run with a placeholder image → get stable URI/hostname.
#   2. Deploy gcp.endpoints.Service with the OpenAPI spec (using that hostname).
#   3. EspV2ImageBuilder runs gcloud_build_image → produces the real ESPv2 image URI.
#   4. The ESPv2 Cloud Run service is updated to use the real image.
"""


def _deploy_espv2_proxy(*,
                        basic_config: ProjectBaseConfig,
                        cloudrun: gcp.cloudrunv2.Service,
                        firebase_project_id: pulumi.Output[str],
                        deployable_version: Version,
                        min_instance_count: int,
                        max_instance_count: int,
                        dependencies: list[pulumi.Resource]) -> gcp.cloudrunv2.Service:
    """
    Deploy ESPv2 as a Cloud Run service proxying the backend Cloud Run service.

    Steps:
    1. Create espv2-proxy-sa in the env project.
    2. Grant espv2-proxy-sa roles/run.invoker on the backend Cloud Run service.
    3. Grant espv2-proxy-sa roles/servicemanagement.serviceController on the project.
    4. Deploy the Endpoints OpenAPI spec (gcp.endpoints.Service).
    5. Build the ESPv2 image with the config ID baked in (EspV2ImageBuilder).
    6. Deploy ESPv2 as a Cloud Run service using the built image.
    7. Allow unauthenticated invocations on the ESPv2 Cloud Run service (ESPv2 handles auth).
    8. Lock down the backend Cloud Run service to internal-only + espv2-proxy-sa invoker.
    """
    # 1. Dedicated service account for ESPv2.
    proxy_sa = gcp.serviceaccount.Account(
        resource_name=get_resource_name(resource="espv2-proxy", resource_type="sa"),
        account_id="espv2-proxy-sa",
        project=basic_config.project,
        display_name="ESPv2 Proxy Service Account — used by ESPv2 to call Cloud Run backend",
        create_ignore_already_exists=True,
        opts=pulumi.ResourceOptions(depends_on=dependencies, provider=basic_config.provider),
    )

    # 2. Allow ESPv2 SA to invoke the backend Cloud Run service.
    gcp.cloudrunv2.ServiceIamMember(
        resource_name=get_resource_name(resource="espv2-proxy-sa-run-invoker", resource_type="iam-member"),
        project=basic_config.project,
        location=basic_config.location,
        name=cloudrun.name,
        role="roles/run.invoker",
        member=proxy_sa.email.apply(lambda email: f"serviceAccount:{email}"),
        opts=pulumi.ResourceOptions(depends_on=[proxy_sa, cloudrun], provider=basic_config.provider),
    )

    # 3. Allow ESPv2 SA to report metrics/validate auth via Service Control API.
    gcp.projects.IAMMember(
        get_resource_name(resource="espv2-proxy-sa-service-controller", resource_type="iam-member"),
        project=basic_config.project,
        role="roles/servicemanagement.serviceController",
        member=proxy_sa.email.apply(lambda email: f"serviceAccount:{email}"),
        opts=pulumi.ResourceOptions(depends_on=[proxy_sa], provider=basic_config.provider),
    )

    # ESPv2 Cloud Run service name — set explicitly and stable across deploys.
    espv2_service_name = "espv2-gateway"

    # The Endpoints service name uses the canonical Cloud Endpoints DNS pattern:
    # {service}.endpoints.{project}.cloud.goog
    # This is deterministic (no GCP-generated hash), which avoids the circular dependency
    # where the Cloud Run URL is unknown until after first deploy.
    endpoints_service_name = pulumi.Output.concat(
        espv2_service_name, ".endpoints.", basic_config.project, ".cloud.goog"
    )

    # 4. Deploy the Endpoints OpenAPI spec.
    # The spec is built dynamically from the FastAPI /openapi.json, so public paths
    # (those without security: [...] in any operation) are injected with security: []
    # before the wildcard catch-all.  This allows unauthenticated access to routes
    # like /user-invitations/check-status without hardcoding them in the template.
    openapi_spec = pulumi.Output.all(
        backend_uri=cloudrun.uri,
        firebase_project_id=firebase_project_id,
        endpoints_service_name=endpoints_service_name,
    ).apply(lambda args: construct_espv2_cfg(
        cloud_run_url=args['backend_uri'],
        espv2_hostname=args['endpoints_service_name'],
        backend_uri=args['backend_uri'],
        firebase_project_id=args['firebase_project_id'],
        expected_version=deployable_version,
    ))

    endpoints_service = gcp.endpoints.Service(
        get_resource_name(resource="espv2-endpoints", resource_type="endpoints-service"),
        service_name=endpoints_service_name,
        openapi_config=openapi_spec,
        opts=pulumi.ResourceOptions(depends_on=[cloudrun] + dependencies, provider=basic_config.provider),
    )

    # 5. Build the ESPv2 image with the config ID baked in.
    image_builder = EspV2ImageBuilder(
        get_resource_name(resource="espv2-image-builder", resource_type="dynamic"),
        service_name=endpoints_service_name,
        config_id=endpoints_service.config_id,
        project_id=basic_config.project,
        opts=pulumi.ResourceOptions(depends_on=[endpoints_service]),
    )

    # 6. Deploy ESPv2 as a Cloud Run service.
    espv2_cloudrun = gcp.cloudrunv2.Service(
        get_resource_name(resource="espv2-gateway", resource_type="service"),
        name=espv2_service_name,
        project=basic_config.project,
        location=basic_config.location,
        ingress="INGRESS_TRAFFIC_ALL",
        template=gcp.cloudrunv2.ServiceTemplateArgs(
            scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
                min_instance_count=min_instance_count,
                max_instance_count=max_instance_count,
            ),
            service_account=proxy_sa.email,
            containers=[
                gcp.cloudrunv2.ServiceTemplateContainerArgs(
                    image=image_builder.image_uri,
                    args=[
                        "--listener_port=8080",
                        "--backend=grpc://localhost:9090",
                        pulumi.Output.concat("--service=", endpoints_service_name),
                        "--rollout_strategy=managed",
                        "--cors_preset=basic",
                    ],
                    envs=[
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="ESPv2_ARGS",
                            value="--http_request_timeout_s=3600",
                        ),
                    ],
                    ports=[gcp.cloudrunv2.ServiceTemplateContainerPortArgs(container_port=8080)],
                )
            ],
        ),
        opts=pulumi.ResourceOptions(
            depends_on=[image_builder, proxy_sa] + dependencies,
            provider=basic_config.provider,
        ),
    )

    # 7. Allow unauthenticated invocations on the ESPv2 Cloud Run service.
    # ESPv2 itself handles Firebase JWT validation.
    gcp.cloudrunv2.ServiceIamMember(
        resource_name=get_resource_name(resource="espv2-allusers-invoker", resource_type="iam-member"),
        project=basic_config.project,
        location=basic_config.location,
        name=espv2_cloudrun.name,
        role="roles/run.invoker",
        member="allUsers",
        opts=pulumi.ResourceOptions(depends_on=[espv2_cloudrun], provider=basic_config.provider),
    )

    # 8. Lock down the backend Cloud Run service: only allow ESPv2 SA to invoke it.
    # Change ingress to internal + load balancer only (blocks direct internet access).
    # NOTE: Pulumi will update the existing backend Cloud Run service in-place.
    gcp.cloudrunv2.ServiceIamMember(
        resource_name=get_resource_name(resource="backend-espv2-invoker", resource_type="iam-member"),
        project=basic_config.project,
        location=basic_config.location,
        name=cloudrun.name,
        role="roles/run.invoker",
        member=proxy_sa.email.apply(lambda email: f"serviceAccount:{email}"),
        opts=pulumi.ResourceOptions(depends_on=[proxy_sa, cloudrun, espv2_cloudrun], provider=basic_config.provider),
    )

    pulumi.export("espv2_cloud_run_name", espv2_cloudrun.name)

    return espv2_cloudrun


def _grant_docker_repository_access_to_project_service_account(
        basic_config: ProjectBaseConfig,
        project_number: pulumi.Output[str],
        docker_project_id: pulumi.Output[str],
        docker_repository_name: pulumi.Output[str],
) -> gcp.artifactregistry.RepositoryIamMember:
    # allow the current environment to read from the docker repository
    return gcp.artifactregistry.RepositoryIamMember(
        resource_name=get_resource_name(resource="project-sa-repository-reader", resource_type="iam-member"),
        project=docker_project_id,
        location=basic_config.location,
        repository=docker_repository_name,
        role="roles/artifactregistry.reader",
        member=project_number.apply(
            lambda _project_number:
            f"serviceAccount:service-{_project_number}@serverless-robot-prod.iam.gserviceaccount.com"),
        opts=pulumi.ResourceOptions(provider=basic_config.provider),
    )


def _get_fully_qualified_image_name(
        docker_repository: pulumi.Output[gcp.artifactregistry.Repository],
        tag: str
) -> pulumi.Output[str]:
    def _get_self_link(repository_info):
        # Get the latest docker image with this tag.
        # Given the actual tag may be assigned to another image, we need to get the latest image with this tag.
        # The `self_link` is the fully qualified image name. with the sha.
        # ref: https://www.pulumi.com/registry/packages/gcp/api-docs/artifactregistry/getdockerimage/#self_link_python
        repository_project_id = repository_info.get("project")
        repository_location = repository_info.get("location")
        repository_name = repository_info.get("name")

        image = gcp.artifactregistry.get_docker_image(
            image_name=f"backend:{tag}",
            location=repository_location,
            # The last part of the repository name to fetch from.
            # see: https://www.pulumi.com/registry/packages/gcp/api-docs/artifactregistry/getdockerimage/#repository_id_python
            # we are using the repository.get("name") to get the repository name because it is the one that returns the last part.
            # Using repository.get("id") would return the full name of the repository.
            repository_id=repository_name,
            project=repository_project_id
        )

        pulumi.info("Deploying image with the link: " + image.self_link)

        return image.self_link

    return docker_repository.apply(_get_self_link)


def _setup_nat_gateway(*,
                       basic_config: ProjectBaseConfig
                       ) -> tuple[gcp.compute.Network, gcp.compute.Subnetwork, list[pulumi.Resource]]:
    """
    Sets up a NAT Gateway in Google Cloud Platform.
    This is used so that all our cloud run instances route their requests through this NAT gateway with a static ip address.
    ref: https://docs.cloud.google.com/run/docs/configuring/static-outbound-ip
    """
    network = gcp.compute.Network(
        get_resource_name(resource="nat-gateway", resource_type="network"),
        auto_create_subnetworks=False,
        opts=pulumi.ResourceOptions(provider=basic_config.provider))

    sub_net = gcp.compute.Subnetwork(
        get_resource_name(resource="nat-gateway", resource_type="sub-network"),
        # Minimum /26 recommended for Cloud Run because ip addresses may change depending on the scaling of instances.
        # ref: https://docs.cloud.google.com/run/docs/configuring/vpc-direct-vpc#scale_up_and_scale_down
        ip_cidr_range="10.0.0.0/26",
        region=basic_config.location,
        network=network.id,
        opts=pulumi.ResourceOptions(provider=basic_config.provider, depends_on=[network]))

    static_ip = gcp.compute.Address(get_resource_name(resource="nat-gateway", resource_type="static-ip"),
                                    region=basic_config.location,
                                    opts=pulumi.ResourceOptions(provider=basic_config.provider))

    router = gcp.compute.Router(
        get_resource_name(resource="nat-gateway", resource_type="router"),
        network=network.id,
        region=basic_config.location,
        opts=pulumi.ResourceOptions(provider=basic_config.provider, depends_on=[network])
    )

    router_nat = gcp.compute.RouterNat(
        get_resource_name(resource="nat-gateway", resource_type="nat"),
        router=router.name,
        nat_ip_allocate_option="MANUAL_ONLY",
        nat_ips=[static_ip.id],
        region=basic_config.location,
        source_subnetwork_ip_ranges_to_nat="ALL_SUBNETWORKS_ALL_IP_RANGES",
        opts=pulumi.ResourceOptions(provider=basic_config.provider, depends_on=[router, sub_net, network]))

    # export the static IP since it might be used two whitelist the cloud run instances.
    pulumi.export("cloudrun_nat_gateway_egress_static_ip", static_ip.address)
    return network, sub_net, [router_nat]


# Deploy cloud run service
# See https://cloud.google.com/run/docs/overview/what-is-cloud-run for more information
def _deploy_cloud_run_service(
        *,
        basic_config: ProjectBaseConfig,
        fully_qualified_image_name: Output[str],
        backend_service_cfg: BackendServiceConfig,
        dependencies: list[pulumi.Resource],
        cv_bucket_name: Output[str],
):
    nat_network, nat_sub_network, nat_dependencies = _setup_nat_gateway(basic_config=basic_config)

    # See https://cloud.google.com/run/docs/securing/service-identity#per-service-identity for more information
    # Create a service account for the Cloud Run service
    service_account = gcp.serviceaccount.Account(
        get_resource_name(resource="backend", resource_type="sa"),

        account_id="backend-sa",
        display_name="The dedicated service account for the Compass backend service",
        create_ignore_already_exists=True,
        project=basic_config.project,
        opts=pulumi.ResourceOptions(depends_on=dependencies, provider=basic_config.provider),
    )

    # Assign the necessary roles to the service account for Vertex AI access.
    iam_member = gcp.projects.IAMMember(
        get_resource_name(resource="backend-sa", resource_type="ai-user-binding"),
        member=service_account.email.apply(lambda email: f"serviceAccount:{email}"),
        role="roles/aiplatform.user",
        project=basic_config.project,
        opts=pulumi.ResourceOptions(depends_on=dependencies + [service_account], provider=basic_config.provider),
    )

    # Deploy cloud run service
    service = gcp.cloudrunv2.Service(
        get_resource_name(resource="cloudrun", resource_type="service"),
        name="cloudrun-service",
        project=basic_config.project,
        location=basic_config.location,
        ingress="INGRESS_TRAFFIC_ALL",
        template=gcp.cloudrunv2.ServiceTemplateArgs(
            # Set max concurrency per instance
            max_instance_request_concurrency=backend_service_cfg.cloudrun_max_instance_request_concurrency,
            timeout=backend_service_cfg.cloudrun_request_timeout,
            execution_environment='EXECUTION_ENVIRONMENT_GEN2',  # Set the execution environment to second generation
            scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
                min_instance_count=backend_service_cfg.cloudrun_min_instance_count,
                max_instance_count=backend_service_cfg.cloudrun_max_instance_count,
            ),
            containers=[
                gcp.cloudrunv2.ServiceTemplateContainerArgs(
                    resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                        limits={
                            'memory': backend_service_cfg.cloudrun_memory_limit,
                            'cpu': backend_service_cfg.cloudrun_cpu_limit,
                        },
                    ),
                    image=fully_qualified_image_name,
                    envs=[
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="TAXONOMY_MONGODB_URI",
                            value=backend_service_cfg.taxonomy_mongodb_uri),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="TAXONOMY_DATABASE_NAME",
                            value=backend_service_cfg.taxonomy_database_name),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="TAXONOMY_MODEL_ID",
                            value=backend_service_cfg.taxonomy_model_id),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="APPLICATION_MONGODB_URI",
                            value=backend_service_cfg.application_mongodb_uri),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="APPLICATION_DATABASE_NAME",
                            value=backend_service_cfg.application_database_name),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="METRICS_MONGODB_URI",
                            value=backend_service_cfg.metrics_mongodb_uri),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="METRICS_DATABASE_NAME",
                            value=backend_service_cfg.metrics_database_name),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="USERDATA_MONGODB_URI",
                            value=backend_service_cfg.userdata_mongodb_uri),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="USERDATA_DATABASE_NAME",
                            value=backend_service_cfg.userdata_database_name),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="VERTEX_API_REGION",
                            value=backend_service_cfg.vertex_api_region),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="EMBEDDINGS_SERVICE_NAME",
                            value=backend_service_cfg.embeddings_service_name),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="EMBEDDINGS_MODEL_NAME",
                            value=backend_service_cfg.embeddings_model_name),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="TARGET_ENVIRONMENT_NAME",
                            value=backend_service_cfg.target_environment_name),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="TARGET_ENVIRONMENT_TYPE",
                            value=backend_service_cfg.target_environment_type),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_URL",
                            value=backend_service_cfg.backend_url),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="FRONTEND_URL",
                            value=backend_service_cfg.frontend_url),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_SENTRY_DSN",
                            value=backend_service_cfg.sentry_dsn),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_SENTRY_CONFIG",
                            value=backend_service_cfg.sentry_config),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_ENABLE_SENTRY",
                            value=backend_service_cfg.enable_sentry),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_ENABLE_METRICS",
                            value=backend_service_cfg.enable_metrics),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="DEFAULT_COUNTRY_OF_USER",
                            value=backend_service_cfg.default_country_of_user),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_FEATURES",
                            value=backend_service_cfg.features),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_EXPERIENCE_PIPELINE_CONFIG",
                            value=backend_service_cfg.experience_pipeline_config),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_CV_STORAGE_BUCKET",
                            value=cv_bucket_name,
                        ),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_CV_MAX_UPLOADS_PER_USER",
                            value=backend_service_cfg.cv_max_uploads_per_user),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_CV_RATE_LIMIT_PER_MINUTE",
                            value=backend_service_cfg.cv_rate_limit_per_minute),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_STREAM_CHUNK_SIZE",
                            value=backend_service_cfg.stream_chunk_size or "10"),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_STREAM_CHUNK_MODE",
                            value=backend_service_cfg.stream_chunk_mode or "chars"),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_STREAM_DELTA_DELAY_MS",
                            value=backend_service_cfg.stream_delta_delay_ms or "12"),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="BACKEND_LANGUAGE_CONFIG",
                            value=backend_service_cfg.language_config),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="GLOBAL_PRODUCT_NAME",
                            value=backend_service_cfg.global_product_name),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="GLOBAL_ENABLE_CV_UPLOAD",
                            value=backend_service_cfg.enable_cv_upload or "false"),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="MATCHING_SERVICE_URL",
                            value=backend_service_cfg.matching_service_url),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="MATCHING_SERVICE_API_KEY",
                            value=backend_service_cfg.matching_service_api_key),
                        gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                            name="COMPASS_INLINE_PHASE_TRANSITION",
                            value=backend_service_cfg.inline_phase_transition),
                        # Add more environment variables here
                    ],
                )
            ],
            service_account=service_account.email,
            vpc_access=gcp.cloudrunv2.ServiceTemplateVpcAccessArgs(
                network_interfaces=[
                    gcp.cloudrunv2.ServiceTemplateVpcAccessNetworkInterfaceArgs(
                        network=nat_network.id,
                        subnetwork=nat_sub_network.id,
                    )
                ],
                # All traffic in the system should pass through the network
                egress="ALL_TRAFFIC",
            )
        ),
        opts=pulumi.ResourceOptions(depends_on=dependencies + nat_dependencies + [iam_member], provider=basic_config.provider),
    )
    pulumi.export("cloud_run_url", service.uri)
    return service, service_account


# export a function build_and_push_image that will be used in the main pulumi program
def deploy_backend(
        *,
        location: str,
        project: str | Output[str],
        project_number: Output[str],
        backend_service_cfg: BackendServiceConfig,
        docker_repository: pulumi.Output[gcp.artifactregistry.Repository],
        deployable_version: Version,
        firebase_project_id: pulumi.Output[str],
):
    """
    Deploy the backend infrastructure
    """
    basic_config = get_project_base_config(project=project, location=location)
    docker_tag = construct_docker_tag(
        git_branch_name=deployable_version.git_branch_name,
        git_sha=deployable_version.git_sha
    )

    # grant the project service account access to the docker repository so that it can pull images
    membership = _grant_docker_repository_access_to_project_service_account(
        basic_config,
        project_number,
        docker_repository.apply(lambda repo: repo.get("project")),
        docker_repository.apply(lambda repo: repo.get("name"))
    )

    # get fully qualified image name
    fully_qualified_image_name = _get_fully_qualified_image_name(
        docker_repository=docker_repository,
        tag=docker_tag
    )

    # Create a private GCS bucket for CV uploads using helper
    cv_bucket = _create_cv_upload_bucket(basic_config=basic_config)

    # Deploy the image as a cloud run service
    cloud_run, cloud_run_sa = _deploy_cloud_run_service(
        basic_config=basic_config,
        fully_qualified_image_name=fully_qualified_image_name,
        backend_service_cfg=backend_service_cfg,
        dependencies=[membership, cv_bucket],
        cv_bucket_name=cv_bucket.name,
    )

    # Grant Cloud Run service account access to the bucket
    _grant_cloud_run_sa_access_to_cv_bucket(
        basic_config=basic_config,
        bucket=cv_bucket,
        service_account=cloud_run_sa,
    )

    _deploy_espv2_proxy(
        basic_config=basic_config,
        cloudrun=cloud_run,
        firebase_project_id=firebase_project_id,
        deployable_version=deployable_version,
        min_instance_count=backend_service_cfg.cloudrun_min_instance_count,
        max_instance_count=backend_service_cfg.cloudrun_max_instance_count,
        dependencies=[cloud_run],
    )
