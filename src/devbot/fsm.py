"""FSM состояния DEVBOT."""

from aiogram.fsm.state import State, StatesGroup


class DevTask(StatesGroup):
    """Состояния диалога разработки."""

    analyzing = State()    # Claude API анализирует задачу
    confirming = State()   # Ждём /approve или /edit от Александра
    executing = State()    # claude CLI запущен
    feedback = State()     # Ждём фидбека 10 мин после выполнения
