"""Тесты AuthService — проверка ролей пользователей."""

from src.services.auth_service import AuthService


def _make_auth() -> AuthService:
    return AuthService(
        admin_ids=frozenset({100, 200}),
        worker_ids=frozenset({300, 400}),
        beekeeper_id=500,
    )


class TestIsAdmin:
    def test_admin_in_set(self):
        auth = _make_auth()
        assert auth.is_admin(100) is True
        assert auth.is_admin(200) is True

    def test_non_admin(self):
        auth = _make_auth()
        assert auth.is_admin(300) is False
        assert auth.is_admin(999) is False

    def test_empty_admin_ids(self):
        auth = AuthService(admin_ids=frozenset(), worker_ids=frozenset())
        assert auth.is_admin(100) is False


class TestIsWorker:
    def test_worker_in_set(self):
        auth = _make_auth()
        assert auth.is_worker(300) is True

    def test_non_worker(self):
        auth = _make_auth()
        assert auth.is_worker(100) is False


class TestIsAdminOrWorker:
    def test_admin(self):
        auth = _make_auth()
        assert auth.is_admin_or_worker(100) is True

    def test_worker(self):
        auth = _make_auth()
        assert auth.is_admin_or_worker(300) is True

    def test_neither(self):
        auth = _make_auth()
        assert auth.is_admin_or_worker(999) is False


class TestIsAdminLegacy:
    def test_admin_id(self):
        auth = _make_auth()
        assert auth.is_admin_legacy(100) is True

    def test_beekeeper_id(self):
        auth = _make_auth()
        assert auth.is_admin_legacy(500) is True

    def test_non_admin_non_beekeeper(self):
        auth = _make_auth()
        assert auth.is_admin_legacy(999) is False
