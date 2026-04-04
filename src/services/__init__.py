"""Service Layer — единый источник бизнес-логики BEEBOT.

Сервисы не знают про Telegram, FastAPI или Redis.
Они работают с CRM, LLM, KB через инъекцию зависимостей.
"""

from src.services.auth_service import AuthService
from src.services.order_service import OrderService
from src.services.notification_service import NotificationService
from src.services.consult_service import ConsultService
from src.services.analytics_service import AnalyticsService
from src.services.worker_service import WorkerService
from src.services.delivery_service import DeliveryService
from src.services.dashboard_service import DashboardService

__all__ = [
    "AuthService",
    "OrderService",
    "NotificationService",
    "ConsultService",
    "AnalyticsService",
    "WorkerService",
    "DeliveryService",
    "DashboardService",
]
