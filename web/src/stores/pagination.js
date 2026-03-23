/**
 * Pinia-стор для кэширования страниц пагинации.
 * Позволяет сохранять позицию и данные при навигации между страницами.
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'

const CACHE_TTL = 2 * 60 * 1000 // 2 минуты

function createPageCache() {
  return {
    items: [],
    total: 0,
    page: 1,
    filters: {},
    fetchedAt: 0,
  }
}

export const useOrdersPageStore = defineStore('ordersPage', () => {
  const cache = ref(createPageCache())

  function isFresh(filters = {}) {
    if (Date.now() - cache.value.fetchedAt > CACHE_TTL) return false
    // Кэш невалиден если фильтры изменились
    const cached = JSON.stringify(cache.value.filters)
    const current = JSON.stringify(filters)
    return cached === current
  }

  function save({ items, total, page, filters = {} }) {
    cache.value = { items, total, page, filters, fetchedAt: Date.now() }
  }

  function invalidate() {
    cache.value.fetchedAt = 0
  }

  return { cache, isFresh, save, invalidate }
})

export const useClientsPageStore = defineStore('clientsPage', () => {
  const cache = ref(createPageCache())

  function isFresh(filters = {}) {
    if (Date.now() - cache.value.fetchedAt > CACHE_TTL) return false
    const cached = JSON.stringify(cache.value.filters)
    const current = JSON.stringify(filters)
    return cached === current
  }

  function save({ items, total, page, filters = {} }) {
    cache.value = { items, total, page, filters, fetchedAt: Date.now() }
  }

  function invalidate() {
    cache.value.fetchedAt = 0
  }

  return { cache, isFresh, save, invalidate }
})
