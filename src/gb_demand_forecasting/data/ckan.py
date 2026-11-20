from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gb_demand_forecasting.utils.io import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadRecord:
    kind: str
    resource_id: str
    resource_name: str
    url: str
    local_path: str
    sha256: str
    size_bytes: int
    retrieved_utc: str


class CKANClient:
    def __init__(self, base_url: str, timeout: int = 60, retries: int = 4) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            backoff_factor=1.2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"User-Agent": "gb-demand-conformal-forecasting/1.0"})

    def package_show(self, dataset_id: str) -> dict[str, Any]:
        url = f"{self.base_url}/api/3/action/datapackage_show"
        response = self.session.get(url, params={"id": dataset_id}, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"CKAN package_show failed for {dataset_id}")
        return payload["result"]

    def select_resource(self, dataset_id: str, name_pattern: str) -> dict[str, Any]:
        package = self.package_show(dataset_id)
        pattern = re.compile(name_pattern, flags=re.IGNORECASE)
        candidates = [r for r in package.get("resources", []) if pattern.search(str(r.get("name", "")))]
        if not candidates:
            available = [str(r.get("name")) for r in package.get("resources", [])]
            raise LookupError(
                f"No resource matching /{name_pattern}/ in {dataset_id}. Available={available}"
            )
        csv_candidates = [r for r in candidates if str(r.get("format", "")).upper() == "CSV"]
        return (csv_candidates or candidates)[0]

    @staticmethod
    def resolve_resource_url(resource: dict[str, Any]) -> str:
        """Resolve a NESO resource download URL without weakening HTTPS checks.

        NESO's ``datapackage_show`` payload exposes the canonical download
        location in ``path``. Some CKAN-compatible payloads use ``url``
        instead, so ``url`` is retained as a compatibility fallback.
        """
        candidates = (resource.get("path"), resource.get("url"))
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.startswith("https://"):
                return candidate

        present = {
            key: resource.get(key)
            for key in ("path", "url")
            if resource.get(key) is not None
        }
        raise ValueError(
            f"Refusing resource without a valid HTTPS download path: {present or 'missing path/url'}"
        )

    def download(self, resource: dict[str, Any], out_path: Path, kind: str) -> DownloadRecord:
        url = self.resolve_resource_url(resource)
        ensure_dir(out_path.parent)
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        digest = hashlib.sha256()
        size = 0
        with self.session.get(url, timeout=self.timeout, stream=True) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        tmp_path.replace(out_path)
        return DownloadRecord(
            kind=kind,
            resource_id=str(resource.get("id", "")),
            resource_name=str(resource.get("name", out_path.name)),
            url=url,
            local_path=str(out_path),
            sha256=digest.hexdigest(),
            size_bytes=size,
            retrieved_utc=datetime.now(timezone.utc).isoformat(),
        )


def acquire_neso(cfg: dict[str, Any], root: Path) -> Path:
    source = cfg["source"]
    raw_dir = root / cfg["paths"]["raw_dir"]
    ensure_dir(raw_dir)
    client = CKANClient(
        base_url=source["base_url"],
        timeout=int(source.get("timeout_seconds", 60)),
        retries=int(source.get("retries", 4)),
    )
    records: list[DownloadRecord] = []
    years = [int(y) for y in source["years"]]

    demand_dataset = source["datasets"]["historic_demand"]
    for year in years:
        resource = client.select_resource(demand_dataset, rf"Historic[_\s-]+Demand[_\s-]+Data[_\s-]+{year}$")
        out = raw_dir / f"historic_demand_{year}.csv"
        LOGGER.info("Downloading %s", resource.get("name"))
        records.append(client.download(resource, out, kind="historic_demand"))

    embedded_dataset = source["datasets"]["embedded_forecasts"]
    for year in years:
        resource = client.select_resource(
            embedded_dataset, rf"Embedded[_\s-]+Solar[_\s-]+and[_\s-]+Wind[_\s-]+Forecast[_\s-]+Archive[_\s-]+{year}$"
        )
        out = raw_dir / f"embedded_forecast_archive_{year}.csv"
        LOGGER.info("Downloading %s", resource.get("name"))
        records.append(client.download(resource, out, kind="embedded_forecast"))

    performance_dataset = source["datasets"]["day_ahead_performance"]
    package = client.package_show(performance_dataset)
    resources = [
        r for r in package.get("resources", []) if str(r.get("format", "")).upper() == "CSV"
    ]
    if not resources:
        raise LookupError(f"No CSV resource found in {performance_dataset}")
    resource = resources[0]
    out = raw_dir / "day_ahead_half_hourly_performance.csv"
    LOGGER.info("Downloading %s", resource.get("name"))
    records.append(client.download(resource, out, kind="day_ahead_performance"))

    record_payloads: list[dict[str, Any]] = []
    root_resolved = root.resolve()
    for record in records:
        payload = asdict(record)
        path = Path(payload["local_path"]).resolve()
        try:
            payload["local_path"] = str(path.relative_to(root_resolved))
        except ValueError:
            payload["local_path"] = str(path)
        record_payloads.append(payload)

    manifest = {
        "source": "NESO Open Data",
        "license": "NESO Open Data Licence",
        "records": record_payloads,
    }
    return write_json(manifest, raw_dir / "source_manifest.json")
