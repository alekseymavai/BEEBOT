"""Плагины BEEBOT — компоненты микроядерной архитектуры.

Каждый плагин — автономный модуль (Plugin-подкласс) со своим lifecycle.
Ядро (BeeBotApp) инициализирует их в порядке зависимостей.

Зависимости:
    crm       → (нет)
    knowledge → crm
    agents    → crm, knowledge
    orders    → crm, agents
    analytics → crm, agents
    delivery  → (нет)
    workers   → crm
    gift      → crm, agents
    monitoring → crm
    web       → crm, orders, analytics, delivery, workers
"""
