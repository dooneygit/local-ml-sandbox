import os
import tempfile

import pytest

os.environ["PLAYGROUND_DATA_DIR"] = tempfile.mkdtemp(prefix="playground-test-")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client
