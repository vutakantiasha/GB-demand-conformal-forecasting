from gb_demand_forecasting.data.ckan import CKANClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"success": True, "result": {"resources": []}}


class FakeSession:
    def __init__(self):
        self.called_url = None
        self.called_params = None

    def get(self, url, params=None, timeout=None, **kwargs):
        self.called_url = url
        self.called_params = params
        return FakeResponse()


def test_neso_uses_datapackage_show_action():
    client = CKANClient("https://api.neso.energy")
    fake = FakeSession()
    client.session = fake
    result = client.package_show("historic-demand-data")
    assert result == {"resources": []}
    assert fake.called_url.endswith("/api/3/action/datapackage_show")
    assert fake.called_params == {"id": "historic-demand-data"}


def test_neso_resource_url_prefers_path():
    resource = {
        "path": "https://api.neso.energy/dataset/x/resource/y/download/file.csv",
        "url": None,
    }
    assert CKANClient.resolve_resource_url(resource) == resource["path"]


def test_neso_resource_url_falls_back_to_url():
    resource = {"path": None, "url": "https://example.test/file.csv"}
    assert CKANClient.resolve_resource_url(resource) == resource["url"]


def test_neso_resource_url_rejects_non_https():
    resource = {"path": "http://example.test/file.csv", "url": None}
    try:
        CKANClient.resolve_resource_url(resource)
    except ValueError as exc:
        assert "valid HTTPS" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
