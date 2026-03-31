"""Оркестратор BEEBOT — маршрутизация между агентами через LangGraph.

Принимает сообщения от Telegram, классифицирует intent через Groq (~100 токенов),
направляет нужному агенту. Хранит состояние диалога по user_id с TTL 30 минут.

Intents:
    consult  → BeebotAgent  — вопросы о продуктах пчеловодства
    order    → LogistAgent  — купить / заказать
    edit     → (bot.py)     — изменить существующий заказ
    track    → (bot.py)     — где мой заказ / трек-номер / статус
    stats    → AnalystAgent — статистика (только для пчеловода)
    greeting → быстрый ответ без LLM
    inspect  → (bot.py)     — «Осмотр улья» диагностический диалог (InspectFSM)
"""

import json
import logging
import time
from collections import Counter
from typing import Literal

from groq import Groq
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from src.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, PROCESSED_DIR, MEMORY_DB_PATH
from src.agents.beebot import BeebotAgent
from src.agents.logist import LogistAgent
from src.agents.analyst import AnalystAgent
from src.memory import UserMemory, extract_fact
from src.ontology import OntologyCache
from src.shared_context import SharedContextStore
from src.anamnesis import AnamnesisCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent type
# ---------------------------------------------------------------------------

Intent = Literal["consult", "order", "edit", "track", "stats", "greeting", "inspect"]

# ---------------------------------------------------------------------------
# Dialog state (per user)
# ---------------------------------------------------------------------------

_DIALOG_TTL_SECONDS = 30 * 60  # 30 minutes
_MAX_HISTORY = 5  # максимум пар сообщений в истории

# FAQ: путь к файлу с накопленными запросами
_FAQ_PATH = PROCESSED_DIR / "faq_queries.json"
_FAQ_SAVE_EVERY = 50  # сохранять на диск каждые N новых запросов


class DialogState(TypedDict):
    """Состояние одного диалога."""
    user_id: int
    query: str
    intent: Intent
    response: str
    chunks: list[dict]
    updated_at: float


# ---------------------------------------------------------------------------
# Conversation history (per user)
# ---------------------------------------------------------------------------

class _ConversationMessage(TypedDict):
    role: str      # "user" или "assistant"
    content: str


# ---------------------------------------------------------------------------
# LangGraph state for a single routing run
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    user_id: int
    query: str
    intent: str          # classified intent
    response: str        # final response text
    chunks: list[dict]   # knowledge base chunks (from BEEBOT)
    history: list[dict]  # conversation history for LLM
    style: str | None    # «Голос Улья» — стиль ответа
    user_name: str | None  # имя пользователя из Telegram


# ---------------------------------------------------------------------------
# Fast intent detection (без LLM)
# ---------------------------------------------------------------------------

_GREETING_WORDS = {
    "привет", "здравствуйте", "здравствуй", "добрый день",
    "доброе утро", "добрый вечер", "хай", "hi", "hello",
    "приветствую", "здрасте",
}

_ORDER_WORDS = {
    "заказать", "купить", "оформить заказ", "хочу заказать",
    "хочу купить", "оформить", "сделать заказ",
}

_EDIT_WORDS = {
    "изменить заказ", "поменять адрес", "добавить товар",
    "убрать товар", "изменить количество", "поменять доставку",
    "редактировать заказ", "изменить адрес", "дозаказ",
    "добавить к заказу", "изменить заказ",
}

_TRACK_WORDS = {
    "где мой заказ", "где заказ", "трек-номер", "трек номер",
    "статус заказа", "отслеживание", "когда доставка",
    "когда приедет", "мой заказ", "что с заказом",
    "когда доставят",
}

_INSPECT_WORDS = {
    "осмотр улья", "осмотри улей", "осмотреть улей", "диагностика улья",
    "диагностику", "осмотри", "осмотр пчёл", "осмотр пчел",
    "проверить улей", "проверить пчёл", "проверить пчел",
}


def _fast_classify(query: str) -> Intent | None:
    """Быстрая классификация по ключевым словам (без LLM)."""
    q = query.lower().strip().rstrip("!?.")

    # Точное совпадение для приветствий
    if q in _GREETING_WORDS:
        return "greeting"

    # Фразовый поиск
    for phrase in _EDIT_WORDS:
        if phrase in q:
            return "edit"
    for phrase in _TRACK_WORDS:
        if phrase in q:
            return "track"
    for phrase in _INSPECT_WORDS:
        if phrase in q:
            return "inspect"
    for phrase in _ORDER_WORDS:
        if phrase in q:
            return "order"

    return None


# ---------------------------------------------------------------------------
# Intent classification prompt (short, ~100 tokens)
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = (
    "Ты классификатор намерений. Определи intent одним словом из списка:\n"
    "consult  — вопрос о продуктах пчеловодства, здоровье, рецептах\n"
    "order    — хочет купить, заказать, оформить новый заказ\n"
    "edit     — хочет изменить существующий заказ (адрес, товары, доставку)\n"
    "track    — спрашивает где заказ, трек-номер, статус доставки\n"
    "stats    — запрос статистики продаж или аналитики\n"
    "greeting — приветствие, здороваться\n"
    "inspect  — хочет осмотреть улей, диагностика пчёл, проверить пчёл\n"
    "Ответь ТОЛЬКО одним словом: consult, order, edit, track, stats, greeting или inspect."
)

_VALID_INTENTS = {"consult", "order", "edit", "track", "stats", "greeting", "inspect"}


def _classify_intent(client: Groq, model: str, query: str) -> Intent:
    """Классифицировать intent запроса через Groq (~100 токенов)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip().lower()
        if raw in _VALID_INTENTS:
            return raw  # type: ignore[return-value]
        logger.warning("Unexpected intent value '%s', falling back to consult", raw)
    except Exception as e:
        logger.error("Intent classification failed: %s — falling back to consult", e)
    return "consult"


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------

class Orchestrator:
    """Оркестратор: классифицирует intent и маршрутизирует к нужному агенту."""

    def __init__(self):
        kwargs: dict = {"api_key": GROQ_API_KEY}
        if GROQ_BASE_URL:
            kwargs["base_url"] = GROQ_BASE_URL
        self._groq = Groq(**kwargs)
        self._model = GROQ_MODEL

        self._beebot = BeebotAgent()
        self._logist = LogistAgent()
        self._analyst = AnalystAgent(groq_client=self._groq, groq_model=self._model)

        # Долгосрочная память пользователей (SQLite)
        self._memory = UserMemory(MEMORY_DB_PATH)

        # Онтология: Симптомы → Показания к применению (из Integram)
        self._ontology = OntologyCache()

        # SharedContext — рабочая память пользователей (Фаза 9.1)
        self._shared_ctx = SharedContextStore()

        # AnamnesisCache — эпизодическая память (Фаза 10.1)
        self._anamnesis = AnamnesisCache(self._memory)

        # CrmAgent — инжектируется из bot.py после создания IntegramClient (Фаза 9.2)
        self._crm_agent = None  # Optional[CrmAgent]

        # AgentSpecsCache — инжектируется из bot.py после загрузки (Фаза 9.5)
        self._agent_specs = None  # Optional[AgentSpecsCache]

        # In-memory dialog state per user_id (сохраняется для совместимости с тестами)
        self._dialog_states: dict[int, DialogState] = {}

        # FAQ: счётчик пользовательских consult-запросов
        self._query_counter: Counter = Counter()
        self._query_counter_since_save: int = 0
        self._load_faq_queries()

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_crm_agent(self, crm_agent) -> None:
        """Инжектировать CrmAgent после создания IntegramClient (вызывается из bot.py)."""
        self._crm_agent = crm_agent

    def set_agent_specs(self, agent_specs) -> None:
        """Инжектировать AgentSpecsCache после загрузки (вызывается из bot.py)."""
        self._agent_specs = agent_specs

    def load_kb(self):
        """Загрузить базу знаний BEEBOT (вызывается при старте бота)."""
        self._beebot.kb.load()

    async def load_ontology(self) -> None:
        """Загрузить онтологию Симптомы→Показания из Integram (при старте бота)."""
        await self._ontology.load()

    async def route(
        self,
        user_id: int,
        query: str,
        style: str | None = None,
        user_name: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Маршрутизировать запрос к нужному агенту.

        Returns:
            (response_text, chunks) — chunks пустой если агент не BEEBOT.
        """
        self._evict_stale_states()

        # Передать историю диалога из SharedContext
        history = self._shared_ctx.get(user_id).get_history()

        initial_state: OrchestratorState = {
            "user_id": user_id,
            "query": query,
            "intent": "",
            "response": "",
            "chunks": [],
            "history": history,
            "style": style,
            "user_name": user_name,
        }

        result = await self._graph.ainvoke(initial_state)

        # Persist dialog state
        self._dialog_states[user_id] = DialogState(
            user_id=user_id,
            query=query,
            intent=result["intent"],
            response=result["response"],
            chunks=result["chunks"],
            updated_at=time.monotonic(),
        )

        # Сохранить в SharedContext историю + FAQ-счётчик (только для consult)
        if result["intent"] == "consult" and result["response"]:
            self._shared_ctx.get(user_id).append_history(query, result["response"])
            self._track_query(query)

        return result["response"], result["chunks"]

    def get_intent(self, user_id: int) -> Intent | None:
        """Вернуть последний intent для пользователя (если не устарел)."""
        state = self._dialog_states.get(user_id)
        if state and self._is_fresh(state):
            return state["intent"]  # type: ignore[return-value]
        return None

    # ------------------------------------------------------------------
    # LangGraph graph definition
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(OrchestratorState)

        graph.add_node("classify", self._node_classify)
        graph.add_node("beebot", self._node_beebot)
        graph.add_node("logist", self._node_logist)
        graph.add_node("analyst", self._node_analyst)
        graph.add_node("greeting", self._node_greeting)
        graph.add_node("passthrough", self._node_passthrough)

        graph.set_entry_point("classify")
        graph.add_conditional_edges(
            "classify",
            self._route_by_intent,
            {
                "consult": "beebot",
                "order": "logist",
                "edit": "passthrough",
                "track": "passthrough",
                "inspect": "passthrough",
                "stats": "analyst",
                "greeting": "greeting",
            },
        )
        graph.add_edge("beebot", END)
        graph.add_edge("logist", END)
        graph.add_edge("analyst", END)
        graph.add_edge("greeting", END)
        graph.add_edge("passthrough", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _node_classify(self, state: OrchestratorState) -> OrchestratorState:
        """Классифицировать intent: сначала быстро, потом LLM."""
        fast = _fast_classify(state["query"])
        if fast:
            intent = fast
            logger.info(
                "Intent (fast) for user %d: '%s' → %s",
                state["user_id"], state["query"][:60], intent,
            )
        else:
            intent = _classify_intent(self._groq, self._model, state["query"])
            logger.info(
                "Intent (LLM) for user %d: '%s' → %s",
                state["user_id"], state["query"][:60], intent,
            )
        return {**state, "intent": intent}

    async def _node_beebot(self, state: OrchestratorState) -> OrchestratorState:
        """Маршрут: consult → BEEBOT-консультант."""
        user_id = state["user_id"]
        query = state["query"]

        # Загрузить онтологию при первом обращении (lazy, async)
        if not self._ontology.loaded:
            await self._ontology.load()

        # Долгосрочная память пользователя (SQLite)
        memory_facts: list[str] = list(self._memory.get_facts(user_id))

        # Онтологическая рекомендация по симптому (из Integram)
        onto_hint = self._ontology.match(query)
        if onto_hint:
            memory_facts.insert(0, onto_hint)

        # Персонализация «Вы уже брали» (Фаза 10.1/10.3) — через AnamnesisCache
        anamnesis = await self._anamnesis.get(user_id, self._crm_agent)
        anamnesis_hint = self._anamnesis.format_for_llm(anamnesis)
        if anamnesis_hint:
            memory_facts.append(anamnesis_hint)

        advice_text = self._ontology.get_advice_prompt() or None

        # system_prompt из AGENT_SPECS (если таблица создана и промпт задан)
        system_prompt_override = None
        if self._agent_specs:
            system_prompt_override = self._agent_specs.get_system_prompt("beebot")

        response, chunks = self._beebot.answer(
            query,
            history=state.get("history"),
            style=state.get("style"),
            memory_facts=memory_facts or None,
            advice_text=advice_text,
            user_name=state.get("user_name"),
            system_prompt_override=system_prompt_override,
        )

        # Авто-сохранить факт если пользователь упомянул здоровье/интерес
        # (отрицания автоматически отфильтровываются в extract_fact — Фаза 10.2)
        fact_result = extract_fact(query)
        if fact_result:
            fact_text, category = fact_result
            added = self._memory.add_fact(user_id, fact_text, category=category, source="auto")
            # Дублировать health-факт в CrmAgent → Integram (Фаза 9.2)
            if added and category == "health":
                if self._crm_agent:
                    await self._crm_agent.add_health_fact(user_id, fact_text)
                else:
                    # Fallback: прямой вызов если CrmAgent не инжектирован
                    try:
                        from src.integram_client import IntegramClient
                        async with IntegramClient() as crm:
                            await crm.add_health_profile(user_id, fact_text)
                    except Exception as _e:
                        logger.debug("Профиль здоровья: не удалось записать в Integram: %s", _e)

        return {**state, "response": response, "chunks": chunks}

    async def _node_logist(self, state: OrchestratorState) -> OrchestratorState:
        """Маршрут: order → переход к FSM (обрабатывается в bot.py)."""
        # Реальный FSM запускается в bot.py после проверки intent
        return {**state, "response": "", "chunks": []}

    async def _node_analyst(self, state: OrchestratorState) -> OrchestratorState:
        """Маршрут: stats → Аналитик."""
        try:
            response = await self._analyst.handle_query(state["query"])
        except Exception as e:
            logger.error("AnalystAgent ошибка: %s", e)
            response = "Не удалось получить статистику. Попробуйте позже."
        return {**state, "response": response, "chunks": []}

    def _node_greeting(self, state: OrchestratorState) -> OrchestratorState:
        """Маршрут: greeting → быстрый ответ."""
        name = state.get("user_name")
        greeting = f"Привет, {name}!" if name else "Привет!"
        return {
            **state,
            "response": (
                f"{greeting} Я бот-помощник Александра Дмитрова.\n\n"
                "Задайте вопрос о продуктах пчеловодства, "
                "или напишите /order чтобы оформить заказ."
            ),
            "chunks": [],
        }

    def _node_passthrough(self, state: OrchestratorState) -> OrchestratorState:
        """Маршрут: edit/track → обрабатывается в bot.py."""
        return {**state, "response": "", "chunks": []}

    # ------------------------------------------------------------------
    # Routing & state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _route_by_intent(state: OrchestratorState) -> str:
        intent = state["intent"]
        if intent in _VALID_INTENTS:
            return intent
        return "consult"

    def _is_fresh(self, state: DialogState) -> bool:
        return (time.monotonic() - state["updated_at"]) < _DIALOG_TTL_SECONDS

    def _evict_stale_states(self):
        """Удалить устаревшие состояния диалога (TTL 30 мин)."""
        stale = [uid for uid, s in self._dialog_states.items() if not self._is_fresh(s)]
        for uid in stale:
            del self._dialog_states[uid]
        # Вытеснить просроченные SharedContext
        self._shared_ctx.evict_stale()

    # ------------------------------------------------------------------
    # FAQ: сбор частых запросов
    # ------------------------------------------------------------------

    def _load_faq_queries(self) -> None:
        """Загрузить накопленные запросы с диска при старте."""
        try:
            if _FAQ_PATH.exists():
                with open(_FAQ_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                self._query_counter = Counter(data)
                logger.info("FAQ: загружено %d уникальных запросов", len(self._query_counter))
        except Exception as e:
            logger.warning("FAQ: не удалось загрузить запросы: %s", e)

    def _save_faq_queries(self) -> None:
        """Сохранить счётчик запросов на диск."""
        try:
            _FAQ_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_FAQ_PATH, "w", encoding="utf-8") as f:
                json.dump(dict(self._query_counter), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("FAQ: не удалось сохранить запросы: %s", e)

    def _track_query(self, query: str) -> None:
        """Зарегистрировать запрос в FAQ-счётчике."""
        # Нормализуем: нижний регистр, обрезаем пробелы
        key = query.lower().strip()
        if len(key) < 5 or len(key) > 200:
            return
        self._query_counter[key] += 1
        self._query_counter_since_save += 1
        if self._query_counter_since_save >= _FAQ_SAVE_EVERY:
            self._save_faq_queries()
            self._query_counter_since_save = 0

    def get_top_queries(self, n: int = 20) -> list[tuple[str, int]]:
        """Вернуть топ-N самых частых запросов."""
        return self._query_counter.most_common(n)

    def flush_faq(self) -> None:
        """Принудительно сохранить FAQ на диск."""
        self._save_faq_queries()
        self._query_counter_since_save = 0
