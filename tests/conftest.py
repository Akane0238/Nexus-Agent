# tests/conftest.py
import os
from dotenv import load_dotenv
import pytest

# Loading environment variables before tests
@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()
