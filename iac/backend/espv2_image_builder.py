"""
ESPv2 image build dynamic resource.

Kept free of `lib` and other iac-local imports: Pulumi runs dynamic providers in a
subprocess that does not execute __main__.py, so sys.path may not include the iac root.
"""

from typing import Any, Optional

import pulumi
import pulumi.dynamic

_ESPV2_DIAG_MAX_CHARS = 100_000


def _espv2_build_diagnostic_text(combined_stdout_stderr: str) -> str:
    s = combined_stdout_stderr.strip()
    if len(s) <= _ESPV2_DIAG_MAX_CHARS:
        return s
    omitted = len(s) - _ESPV2_DIAG_MAX_CHARS
    return f"[... {omitted} characters omitted ...]\n" + s[-_ESPV2_DIAG_MAX_CHARS:]


_GCLOUD_BUILD_IMAGE_NEW_IMAGE_LINE = (
    'NEW_IMAGE="${IMAGE_REPOSITORY}/endpoints-runtime-serverless:${ESP_FULL_VERSION}-${SERVICE}-${CONFIG_ID}"'
)
# Upstream tag is ${ESP_FULL_VERSION}-${SERVICE}-${CONFIG_ID}; SERVICE is the full *.endpoints.*.cloud.goog name
# and can exceed GCR/Cloud Build tag limits. Use a short deterministic tag (still unique per config).
_GCLOUD_BUILD_IMAGE_NEW_IMAGE_PATCHED = (
    'NEW_IMAGE="${IMAGE_REPOSITORY}/endpoints-runtime-serverless:espv2-'
    '$(printf \'%s\' "${ESP_FULL_VERSION}-${SERVICE}-${CONFIG_ID}" | sha256sum | cut -c1-32)"'
)


def _patch_gcloud_build_image_script(script_bytes: bytes) -> bytes:
    text = script_bytes.decode("utf-8")
    if _GCLOUD_BUILD_IMAGE_NEW_IMAGE_LINE not in text:
        raise RuntimeError(
            "ESPv2: upstream gcloud_build_image script changed; expected NEW_IMAGE line not found. "
            "Update _GCLOUD_BUILD_IMAGE_NEW_IMAGE_LINE in espv2_image_builder.py."
        )
    patched = text.replace(_GCLOUD_BUILD_IMAGE_NEW_IMAGE_LINE, _GCLOUD_BUILD_IMAGE_NEW_IMAGE_PATCHED, 1)
    return patched.encode("utf-8")


class _EspV2ImageBuilderProvider(pulumi.dynamic.ResourceProvider):
    """
    Runs `gcloud_build_image` to build an ESPv2 container image with the
    given Endpoints service name and config ID baked in.
    Uses the same GCP project as the rest of the Pulumi stack.
    """

    _GCLOUD_BUILD_IMAGE_URL = (
        "https://raw.githubusercontent.com/GoogleCloudPlatform/esp-v2/master"
        "/docker/serverless/gcloud_build_image"
    )

    def _run_build_image(self, service_name: str, config_id: str, project_id: str) -> str:
        import concurrent.futures
        import os
        import stat
        import subprocess
        import tempfile
        import time
        import urllib.request

        # CI often uses a service account whose *home* project differs from the env project. gcloud then
        # attributes API quota to that home project; if Cloud Build is only enabled on the env project,
        # you see: "cloudbuild.googleapis.com not enabled on project [<SA home number>]". Point quota at
        # the env project where this stack enables Cloud Build (see environment create_new_environment).
        subprocess.run(
            ["gcloud", "config", "set", "billing/quota_project", project_id],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        subprocess_env = os.environ.copy()
        subprocess_env["GOOGLE_CLOUD_QUOTA_PROJECT"] = project_id

        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False, mode="wb") as f:
            script_path = f.name
            with urllib.request.urlopen(self._GCLOUD_BUILD_IMAGE_URL) as resp:
                f.write(_patch_gcloud_build_image_script(resp.read()))
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)

        cmd = ["/bin/bash", script_path, "-s", service_name, "-c", config_id, "-p", project_id]
        pulumi.log.info("ESPv2 image build: started")
        start = time.monotonic()

        def _run_gcloud_build_image() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=subprocess_env,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_gcloud_build_image)
                while True:
                    try:
                        result = future.result(timeout=120)
                        break
                    except concurrent.futures.TimeoutError:
                        pulumi.log.info(
                            f"ESPv2 image build: still running ({int(time.monotonic() - start)}s elapsed)"
                        )
        finally:
            os.unlink(script_path)

        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode != 0:
            detail = _espv2_build_diagnostic_text(combined)
            raise RuntimeError(
                f"gcloud_build_image failed with exit code {result.returncode}:\n{detail}"
            )

        pulumi.log.info("ESPv2 image build: gcloud_build_image subprocess exited")

        for line in reversed(combined.splitlines()):
            for token in line.split():
                token = token.strip("'\"")
                if "endpoints-runtime-serverless" in token and ("gcr.io" in token or ".pkg.dev" in token):
                    return token
        detail = _espv2_build_diagnostic_text(combined)
        raise RuntimeError(
            "Could not extract ESPv2 image URI from gcloud_build_image output:\n" + detail
        )

    def create(self, props: dict[str, Any]) -> pulumi.dynamic.CreateResult:
        service_name: str = props["service_name"]
        config_id: str = props["config_id"]
        project_id: str = props["project_id"]
        image_uri = self._run_build_image(service_name, config_id, project_id)
        resource_id = f"{project_id}/{service_name}/{config_id}"
        return pulumi.dynamic.CreateResult(id_=resource_id, outs={**props, "image_uri": image_uri})

    def diff(self, id: str, olds: dict[str, Any], news: dict[str, Any]) -> pulumi.dynamic.DiffResult:
        changes = olds.get("config_id") != news.get("config_id")
        return pulumi.dynamic.DiffResult(changes=changes, replaces=[] if not changes else ["config_id"])

    def update(self, id: str, olds: dict[str, Any], news: dict[str, Any]) -> pulumi.dynamic.UpdateResult:
        result = self.create(news)
        return pulumi.dynamic.UpdateResult(outs=result.outs)

    def delete(self, id: str, props: dict[str, Any]) -> None:
        pass


class EspV2ImageBuilder(pulumi.dynamic.Resource):
    """
    Builds an ESPv2 container image with the Endpoints config_id baked in.
    Exposes `image_uri` as an Output.
    """
    image_uri: pulumi.Output[str]

    def __init__(self,
                 name: str,
                 service_name: pulumi.Input[str],
                 config_id: pulumi.Input[str],
                 project_id: pulumi.Input[str],
                 opts: Optional[pulumi.ResourceOptions] = None):
        super().__init__(
            _EspV2ImageBuilderProvider(),
            name,
            {
                "service_name": service_name,
                "config_id": config_id,
                "project_id": project_id,
                "image_uri": None,
            },
            opts,
        )
