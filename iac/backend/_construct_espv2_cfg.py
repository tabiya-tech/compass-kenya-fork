import os
import time

import pulumi
import yaml
import requests

import google
from google.oauth2 import id_token
from google.auth.transport.requests import Request

from lib import Version


def _get_open_api_config(cloud_run_url: str, _id_token: str, expected_version: Version) -> dict:
    """
    Fetch the OpenAPI 3 spec from the Cloud Run backend, retrying until the expected version is live.

    :param cloud_run_url: The URL of the Cloud Run service (internal, requires Bearer token).
    :param _id_token: ID token for authenticating to the private Cloud Run URL.
    :param expected_version: The deployed version; used to confirm the right image is serving.
    :return: Parsed OpenAPI 3 JSON as a dict.
    """
    attempts = 5

    open_api_3_json = None
    for attempt in range(attempts):
        response = requests.get(
            f"{cloud_run_url}/openapi.json",
            headers={"Authorization": f"Bearer {_id_token}"},
        )

        if response.status_code == 200:
            open_api_3_json = response.json()
        else:
            raise ValueError(f"Failed to fetch OpenAPI JSON from {cloud_run_url}: {response.status_code}")

        versions_match = (
            open_api_3_json.get("info", {}).get("version")
            == f"{expected_version.git_branch_name}-{expected_version.git_sha}"
        )

        # In preview/dry-run mode skip the version check to allow pulumi preview without a live service.
        if pulumi.runtime.is_dry_run() or versions_match:
            return open_api_3_json

        if attempt < attempts - 1:
            # Exponential backoff: 3, 6, 12, 24 seconds
            wait_time = 3 * (2 ** attempt)
            pulumi.info(f"Waiting {wait_time}s before retrying (attempt {attempt + 1}/{attempts})...")
            pulumi.info(
                f"Expected version: {expected_version.git_branch_name}-{expected_version.git_sha}, "
                f"got: {open_api_3_json.get('info', {}).get('version')}"
            )
            time.sleep(wait_time)

    return open_api_3_json


def _convert_params(parameters: list) -> list:
    """
    Convert OpenAPI 3 parameters to Swagger 2.0 format.
    Flattens schema.type up to the parameter level (required by Swagger 2.0).
    """
    result = []
    for param in parameters:
        p = {k: v for k, v in param.items() if k != "schema"}
        schema = param.get("schema", {})
        if "type" in schema:
            p["type"] = schema["type"]
        else:
            p["type"] = "string"  # default; ESPv2 needs a type
        result.append(p)
    return result


def _build_espv2_spec(
    openapi3: dict,
    template_str: str,
    espv2_hostname: str,
    backend_uri: str,
    firebase_project_id: str,
) -> str:
    """
    Build the ESPv2 Swagger 2.0 OpenAPI spec.

    Substitutes hostname/backend/firebase values into the template, then walks
    the FastAPI OpenAPI 3 spec to classify each path and inject explicit entries
    before the wildcard ``/{path=**}`` catch-all:

    - Public paths (no ``security`` on any operation) → ``security: []``
    - API-key paths (``gcp_api_key`` security on every operation) → ``security: [{api_key: []}]``
      ESPv2 validates the GCP API key from the ``x-api-key`` request header.
    - Firebase-protected paths → no injection; handled by the wildcard catch-all.

    :param openapi3: Parsed OpenAPI 3 spec from the FastAPI backend.
    :param template_str: Raw YAML text of espv2_openapi_template.yaml.
    :param espv2_hostname: Cloud Endpoints hostname (e.g. espv2-gateway.endpoints.<project>.cloud.goog).
    :param backend_uri: Cloud Run backend URI.
    :param firebase_project_id: Firebase / GCP project ID for JWT validation.
    :return: YAML string of the completed ESPv2 Swagger 2.0 spec.
    """
    spec = yaml.load(template_str, Loader=yaml.SafeLoader)

    # Fill in the placeholder values.
    spec["host"] = espv2_hostname
    spec["x-google-backend"]["address"] = backend_uri
    spec["x-google-backend"]["jwt_audience"] = backend_uri
    spec["securityDefinitions"]["firebase"]["x-google-issuer"] = (
        f"https://securetoken.google.com/{firebase_project_id}"
    )
    spec["securityDefinitions"]["firebase"]["x-google-audiences"] = firebase_project_id

    # The FastAPI scheme name used on API-key-protected routes (must match ApiKeyAuth's scheme_name).
    _FASTAPI_API_KEY_SCHEME = "gcp_api_key"
    # The ESPv2 security definition name for the x-api-key header (must match espv2_openapi_template.yaml).
    _ESPV2_API_KEY_SCHEME = "api_key"

    # Walk FastAPI's OpenAPI 3 paths and inject explicit entries into the ESPv2 Swagger 2.0 spec for:
    #   - Public paths: every operation has no ``security`` field → inject security: []
    #   - API-key paths: every operation declares gcp_api_key security → inject security: [{api_key: []}]
    # Firebase-protected paths are handled by the wildcard ``/{path=**}`` catch-all and need no injection.
    for path, path_item in openapi3.get("paths", {}).items():
        # path_item values can be non-dict (e.g. summary strings) — skip those.
        operations = {
            method: op
            for method, op in path_item.items()
            if isinstance(op, dict)
        }
        if not operations:
            continue

        is_public = all("security" not in op for op in operations.values())
        is_api_key_protected = all(
            op.get("security") == [{_FASTAPI_API_KEY_SCHEME: []}]
            for op in operations.values()
        )

        if not is_public and not is_api_key_protected:
            continue

        espv2_security: list = [] if is_public else [{_ESPV2_API_KEY_SCHEME: []}]

        for method, op_obj in operations.items():
            entry: dict = {
                "operationId": op_obj.get("operationId") or f"{method}_{path.replace('/', '_')}",
                "security": espv2_security,
                "responses": {"200": {"description": "OK"}},
            }
            if "parameters" in op_obj:
                entry["parameters"] = _convert_params(op_obj["parameters"])
            spec["paths"].setdefault(path, {})[method] = entry

        label = "public (no-auth)" if is_public else "api-key-protected (x-api-key header)"
        pulumi.info(f"ESPv2: injecting {label} path: {path}")

    yaml_bytes = yaml.dump(
        spec,
        None,
        encoding="utf-8",
        allow_unicode=True,
        indent=2,
    )
    return yaml_bytes.decode("utf-8")


def construct_espv2_cfg(
    *,
    cloud_run_url: str,
    espv2_hostname: str,
    backend_uri: str,
    firebase_project_id: str,
    expected_version: Version,
) -> str:
    """
    Entry point called from deploy_backend.py via pulumi.Output.apply().

    Fetches the FastAPI OpenAPI spec, extracts public paths, and returns the
    fully-rendered ESPv2 Swagger 2.0 YAML string.
    """
    pulumi.info("Constructing ESPv2 OpenAPI configuration...")
    pulumi.info(f"cloud_run_url: {cloud_run_url}")

    # Authenticate to the private Cloud Run URL using Application Default Credentials.
    _credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    request = Request()
    _credentials.refresh(request)
    _id_token = id_token.fetch_id_token(request, cloud_run_url)

    open_api_3_json = _get_open_api_config(cloud_run_url, _id_token, expected_version)

    template_path = os.path.join(os.path.dirname(__file__), "espv2_openapi_template.yaml")
    with open(template_path) as f:
        template_str = f.read()

    return _build_espv2_spec(
        openapi3=open_api_3_json,
        template_str=template_str,
        espv2_hostname=espv2_hostname,
        backend_uri=backend_uri,
        firebase_project_id=firebase_project_id,
    )
