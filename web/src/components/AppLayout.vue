<template>
  <div class="flex h-screen bg-gray-50">
    <!-- Боковое меню -->
    <aside class="w-64 flex-shrink-0 bg-white border-r border-gray-200 flex flex-col shadow-sm">
      <!-- Логотип -->
      <div class="px-6 py-5 border-b border-gray-100">
        <div class="flex items-center gap-3">
          <span class="text-3xl">🐝</span>
          <div>
            <h1 class="text-base font-bold text-gray-800 leading-tight">Усадьба Дмитровых</h1>
            <p class="text-xs text-gray-400">Панель управления</p>
          </div>
        </div>
      </div>

      <!-- Навигация -->
      <nav class="flex-1 px-3 py-4 space-y-1">
        <RouterLink
          v-for="item in visibleNavItems"
          :key="item.to"
          :to="item.to"
          class="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors"
          :class="isActive(item.to)
            ? 'bg-amber-50 text-amber-700'
            : 'text-gray-600 hover:bg-gray-50 hover:text-gray-800'"
        >
          <i :class="['pi', item.icon, 'text-base']" />
          <span>{{ item.label }}</span>
          <span
            v-if="item.badge"
            class="ml-auto bg-amber-100 text-amber-700 text-xs font-semibold px-2 py-0.5 rounded-full"
          >{{ item.badge }}</span>
        </RouterLink>
      </nav>

      <!-- Пользователь + Выход -->
      <div class="px-3 py-4 border-t border-gray-100 space-y-2">
        <div class="px-3 py-1">
          <div class="text-sm font-medium text-gray-700 truncate">{{ auth.displayName || auth.role }}</div>
          <div class="text-xs text-gray-400">{{ roleName }}</div>
        </div>
        <button
          @click="handleLogout"
          class="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-sm font-medium text-gray-500 hover:bg-gray-50 hover:text-gray-700 transition-colors"
        >
          <i class="pi pi-sign-out text-base" />
          <span>Выйти</span>
        </button>
      </div>
    </aside>

    <!-- Основной контент -->
    <main class="flex-1 overflow-auto">
      <div class="p-6">
        <RouterView />
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter, RouterLink, RouterView } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import { useAuthStore } from '../stores/auth.js'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const toast = useToast()

let _sse = null

onMounted(() => {
  if (!auth.token) return
  _sse = new EventSource(`/api/events?token=${encodeURIComponent(auth.token)}`)
  _sse.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.type === 'order_status') {
        toast.add({
          severity: 'info',
          summary: 'Статус заказа изменён',
          detail: `Заказ #${data.order_number || data.order_id} → ${data.status}`,
          life: 6000,
        })
      } else if (data.type === 'order_tracking') {
        toast.add({
          severity: 'info',
          summary: 'Трек-номер добавлен',
          detail: `Заказ #${data.order_number || data.order_id}: ${data.tracking_number}`,
          life: 6000,
        })
      }
    } catch {
      // Игнорировать невалидные события
    }
  }
  _sse.onerror = () => {
    // Браузер автоматически переподключается; ошибки не показываем
  }
})

onUnmounted(() => {
  if (_sse) {
    _sse.close()
    _sse = null
  }
})

const allNavItems = [
  { to: '/dashboard', icon: 'pi-chart-bar', label: 'Дашборд', roles: ['admin'] },
  { to: '/journal', icon: 'pi-calendar', label: 'Журнал по месяцам', roles: ['admin'] },
  { to: '/orders', icon: 'pi-shopping-cart', label: 'Заказы', roles: ['admin'] },
  { to: '/clients', icon: 'pi-users', label: 'Клиенты', roles: ['admin'] },
  { to: '/products', icon: 'pi-box', label: 'Товары', roles: ['admin'] },
  { to: '/orders/new', icon: 'pi-plus-circle', label: 'Новый заказ', roles: ['admin'] },
  { to: '/batches', icon: 'pi-send', label: 'Партии отправки', roles: ['admin'] },
  { to: '/packing', icon: 'pi-box', label: 'Сборка', roles: ['admin', 'warehouse'] },
  { to: '/stock', icon: 'pi-warehouse', label: 'Склад', roles: ['admin', 'warehouse'] },
  { to: '/users', icon: 'pi-cog', label: 'Пользователи', roles: ['admin'] },
  { to: '/architecture', icon: 'pi-sitemap', label: 'Архитектура', roles: ['admin'] },
]

const visibleNavItems = computed(() =>
  allNavItems.filter(item => item.roles.includes(auth.role))
)

const roleName = computed(() => {
  const names = { admin: 'Администратор', warehouse: 'Склад' }
  return names[auth.role] || auth.role
})

function isActive(path) {
  if (path === '/dashboard' || path === '/journal' || path === '/batches') return route.path === path
  return route.path.startsWith(path)
}

function handleLogout() {
  auth.logout()
  router.push('/login')
}
</script>
