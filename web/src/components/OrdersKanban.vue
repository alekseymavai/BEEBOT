<template>
  <div class="flex gap-4 overflow-x-auto pb-4">
    <div
      v-for="col in columns"
      :key="col.status"
      class="flex-shrink-0 w-64"
    >
      <!-- Заголовок колонки -->
      <div class="flex items-center justify-between mb-3 px-1">
        <div class="flex items-center gap-2">
          <span :class="['w-2.5 h-2.5 rounded-full', col.dot]" />
          <span class="text-sm font-semibold text-gray-700">{{ col.status }}</span>
        </div>
        <span class="text-xs bg-gray-100 text-gray-500 font-medium px-2 py-0.5 rounded-full">
          {{ ordersByStatus[col.status]?.length || 0 }}
        </span>
      </div>

      <!-- Карточки -->
      <div
        class="min-h-32 space-y-2 rounded-xl p-2"
        :class="dragOver === col.status ? 'bg-amber-50 ring-2 ring-amber-200' : 'bg-gray-100'"
        @dragover.prevent="dragOver = col.status"
        @dragleave="dragOver = null"
        @drop.prevent="onDrop(col.status)"
      >
        <div
          v-for="order in ordersByStatus[col.status]"
          :key="order.id"
          draggable="true"
          class="bg-white rounded-lg p-3 shadow-sm border border-gray-100 cursor-grab active:cursor-grabbing hover:shadow-md transition-shadow"
          :class="draggingId === order.id ? 'opacity-40' : ''"
          @dragstart="onDragStart(order)"
          @dragend="onDragEnd"
        >
          <div class="flex items-start justify-between mb-1.5">
            <span class="text-xs font-mono font-semibold text-amber-700">{{ order.number }}</span>
            <span class="text-xs text-gray-400">{{ formatDate(order.date) }}</span>
          </div>
          <div class="text-sm text-gray-700 font-medium truncate mb-1">
            {{ order.client_name || `Клиент #${order.client_id}` }}
          </div>
          <div class="flex items-center justify-between mt-2">
            <span class="text-xs text-gray-500 flex items-center gap-1">
              <i :class="deliveryIcon(order.delivery_method)" class="text-xs" />
              {{ order.delivery_method || '—' }}
            </span>
            <span class="text-sm font-semibold text-gray-800">{{ formatMoney(order.total) }}</span>
          </div>
        </div>

        <div v-if="!ordersByStatus[col.status]?.length" class="text-center py-6 text-xs text-gray-400">
          Нет заказов
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { formatDate, formatMoney } from '../utils.js'

const props = defineProps({
  orders: { type: Array, default: () => [] },
})

const emit = defineEmits(['status-change'])

const columns = [
  { status: 'Новый',       dot: 'bg-blue-500' },
  { status: 'Подтверждён', dot: 'bg-amber-500' },
  { status: 'В сборке',    dot: 'bg-purple-500' },
  { status: 'Отправлен',   dot: 'bg-indigo-500' },
  { status: 'Доставлен',   dot: 'bg-green-500' },
]

const ordersByStatus = computed(() => {
  const map = {}
  for (const col of columns) map[col.status] = []
  for (const o of props.orders) {
    if (map[o.status]) map[o.status].push(o)
    // Отменённые не показываем в канбане
  }
  return map
})

const draggingOrder = ref(null)
const draggingId = ref(null)
const dragOver = ref(null)

function onDragStart(order) {
  draggingOrder.value = order
  draggingId.value = order.id
}

function onDragEnd() {
  draggingId.value = null
  dragOver.value = null
}

function onDrop(targetStatus) {
  dragOver.value = null
  if (!draggingOrder.value) return
  if (draggingOrder.value.status === targetStatus) return
  emit('status-change', { order: draggingOrder.value, newStatus: targetStatus })
  draggingOrder.value = null
}

function deliveryIcon(method) {
  if (!method) return 'pi pi-question'
  if (method.includes('СДЭК')) return 'pi pi-truck'
  if (method.includes('Почта') || method.includes('почта')) return 'pi pi-envelope'
  return 'pi pi-map-marker'
}
</script>
