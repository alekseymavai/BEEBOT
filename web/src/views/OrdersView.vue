<template>
  <div>
    <Toast />
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-2xl font-bold text-gray-800">Заказы</h2>
      <div class="flex items-center gap-2">
        <!-- Переключатель вида -->
        <div class="flex rounded-lg border border-gray-200 overflow-hidden text-sm">
          <button
            :class="['px-3 py-1.5 transition-colors', viewMode === 'list' ? 'bg-amber-500 text-white' : 'bg-white text-gray-600 hover:bg-gray-50']"
            @click="viewMode = 'list'"
          ><i class="pi pi-list" /></button>
          <button
            :class="['px-3 py-1.5 transition-colors', viewMode === 'kanban' ? 'bg-amber-500 text-white' : 'bg-white text-gray-600 hover:bg-gray-50']"
            @click="viewMode = 'kanban'"
          ><i class="pi pi-th-large" /></button>
        </div>
        <RouterLink to="/orders/new">
          <Button label="Новый заказ" icon="pi pi-plus" size="small" />
        </RouterLink>
      </div>
    </div>

    <!-- Фильтры -->
    <div class="bg-white rounded-xl border border-gray-100 shadow-sm p-4 mb-4 flex gap-3 flex-wrap">
      <InputText
        v-model="filterSearch"
        placeholder="Поиск: номер, клиент, трек..."
        class="w-56"
        @input="onSearchInput"
      />
      <DatePicker
        v-model="filterDateRange"
        selection-mode="range"
        :manual-input="false"
        date-format="dd.mm.yy"
        placeholder="Период"
        class="w-48"
        show-button-bar
        @update:model-value="onDateRangeChange"
      />
      <Select
        v-model="filterStatus"
        :options="[{ label: 'Все статусы', value: '' }, ...statusOptions]"
        option-label="label"
        option-value="value"
        placeholder="Статус"
        class="w-44"
        @change="loadOrders"
      />
      <Select
        v-model="filterSource"
        :options="[{ label: 'Все источники', value: '' }, ...sourceOptions]"
        option-label="label"
        option-value="value"
        placeholder="Источник"
        class="w-44"
        @change="loadOrders"
      />
      <span class="flex-1" />
      <Button
        label="Сбросить"
        icon="pi pi-filter-slash"
        severity="secondary"
        size="small"
        @click="resetFilters"
      />
      <Button
        label="CSV"
        icon="pi pi-download"
        severity="secondary"
        size="small"
        :loading="exporting"
        @click="doExport"
      />
    </div>

    <!-- Канбан-вид -->
    <div v-if="viewMode === 'kanban'" class="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
      <OrdersKanban :orders="orders" @status-change="onKanbanStatusChange" />
    </div>

    <!-- Таблица заказов -->
    <div v-else class="bg-white rounded-xl border border-gray-100 shadow-sm">
      <!-- Панель групповых действий -->
      <Transition name="slide-up">
        <div
          v-if="selectedOrders.length"
          class="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-gray-900 text-white px-5 py-3 rounded-2xl shadow-xl"
        >
          <span class="text-sm font-medium">Выбрано: {{ selectedOrders.length }}</span>
          <Select
            v-model="batchStatus"
            :options="statusOptions"
            option-label="label"
            option-value="value"
            placeholder="Сменить статус"
            class="w-44 text-sm"
          />
          <Button
            label="Применить"
            size="small"
            :disabled="!batchStatus"
            :loading="batchLoading"
            @click="applyBatchStatus"
          />
          <Button
            icon="pi pi-times"
            severity="secondary"
            size="small"
            text
            @click="selectedOrders = []"
          />
        </div>
      </Transition>

      <DataTable
        :value="orders"
        :loading="loading"
        dataKey="id"
        lazy
        paginator
        :rows="50"
        :total-records="totalRecords"
        v-model:expandedRows="expandedRows"
        v-model:selection="selectedOrders"
        :row-class="rowStatusClass"
        row-hover
        sort-mode="single"
        :sort-field="sortField"
        :sort-order="sortOrderNum"
        class="text-sm"
        @page="onPage"
        @sort="onSort"
        @row-click="(e) => $router.push(`/orders/${e.data.id}`)"
      >
        <template #empty>
          <div class="text-center py-8 text-gray-400">Заказов нет</div>
        </template>
        <Column selection-mode="multiple" style="width:3rem" @click.stop />
        <Column expander style="width:3rem" />
        <Column field="number" header="Номер" sortable style="width:120px" />
        <Column field="client_name" header="Клиент" sortable>
          <template #body="{ data }">{{ data.client_name || `Клиент #${data.client_id}` }}</template>
        </Column>
        <Column field="date" header="Дата" sortable>
          <template #body="{ data }">{{ formatDate(data.date) }}</template>
        </Column>
        <Column field="source" header="Источник" sortable>
          <template #body="{ data }">
            <span v-if="data.source" class="text-xs px-2 py-0.5 rounded-full"
              :class="{
                'bg-blue-50 text-blue-700': data.source === 'Telegram',
                'bg-purple-50 text-purple-700': data.source === 'UDS',
                'bg-green-50 text-green-700': data.source === 'WhatsApp',
                'bg-gray-50 text-gray-600': !['Telegram','UDS','WhatsApp'].includes(data.source),
              }">{{ data.source }}</span>
            <span v-else class="text-gray-300">—</span>
          </template>
        </Column>
        <Column field="status" header="Статус" sortable style="width:150px">
          <template #body="{ data }">
            <Select
              :model-value="data.status"
              :options="statusOptions"
              option-label="label"
              option-value="value"
              class="text-xs w-full"
              :pt="{ root: { class: 'inline-status-select' } }"
              @update:model-value="(val) => changeStatus(data, val)"
              @click.stop
            />
          </template>
        </Column>
        <Column field="delivery_method" header="Доставка" />
        <Column field="total" header="Сумма" sortable>
          <template #body="{ data }">{{ formatMoney(data.total) }}</template>
        </Column>
        <Column field="tracking_number" header="Трек-номер">
          <template #body="{ data }">
            <span v-if="data.tracking_number" class="font-mono text-xs text-blue-600">
              {{ data.tracking_number }}
            </span>
            <span v-else class="text-gray-300">—</span>
          </template>
        </Column>
        <template #expansion="{ data }">
          <div class="px-4 py-2">
            <div v-if="!data.items" class="text-sm text-gray-400">Загрузка состава...</div>
            <div v-else-if="data.items.length === 0" class="text-sm text-gray-400">Позиции не найдены в CRM</div>
            <table v-else class="text-sm w-full">
              <thead>
                <tr class="text-gray-500 text-xs">
                  <th class="text-left pb-1 font-medium">Товар</th>
                  <th class="text-right pb-1 font-medium w-16">Кол-во</th>
                  <th class="text-right pb-1 font-medium w-24">Цена</th>
                  <th class="text-right pb-1 font-medium w-24">Сумма</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in data.items" :key="item.id" class="border-t border-gray-100">
                  <td class="py-1 text-gray-700">{{ item.product_name || `Товар #${item.product_id}` }}</td>
                  <td class="py-1 text-right text-gray-600">{{ item.quantity }} шт</td>
                  <td class="py-1 text-right text-gray-600">{{ formatMoney(item.unit_price) }}</td>
                  <td class="py-1 text-right font-medium">{{ formatMoney(item.total) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>
      </DataTable>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRouter, RouterLink } from 'vue-router'
import OrdersKanban from '../components/OrdersKanban.vue'
import DataTable from 'primevue/datatable'
import Column from 'primevue/column'
import Button from 'primevue/button'
import Select from 'primevue/select'
import InputText from 'primevue/inputtext'
import DatePicker from 'primevue/datepicker'
import Toast from 'primevue/toast'
import { useToast } from 'primevue/usetoast'
import { getOrders, getReference, exportOrders, getOrderItems, updateOrderStatus, batchUpdateStatus } from '../api.js'
import StatusBadge from '../components/StatusBadge.vue'
import { formatDate, formatMoney } from '../utils.js'

const toast = useToast()
const router = useRouter()
const viewMode = ref(localStorage.getItem('orders-view') || 'list')
watch(viewMode, v => localStorage.setItem('orders-view', v))

const STATUS_ROW_CLASS = {
  'Новый':       'row-status-new',
  'Подтверждён': 'row-status-confirmed',
  'В сборке':    'row-status-packing',
  'Отправлен':   'row-status-shipped',
  'Доставлен':   'row-status-delivered',
  'Отменён':     'row-status-cancelled',
}
function rowStatusClass(data) {
  return STATUS_ROW_CLASS[data.status] || ''
}

const loading = ref(true)
const exporting = ref(false)
const orders = ref([])
const totalRecords = ref(0)
const currentPage = ref(1)
const filterSearch = ref('')
const filterStatus = ref('')
const filterSource = ref('')
const filterDateRange = ref(null)
const sortField = ref(null)
const sortOrderNum = ref(null)  // 1 = asc, -1 = desc (PrimeVue convention)
let searchTimer = null

function onSearchInput() {
  clearTimeout(searchTimer)
  searchTimer = setTimeout(() => loadOrders(1), 400)
}

function onDateRangeChange(val) {
  // Загружаем только когда выбраны обе даты или сброс (null)
  if (!val || (Array.isArray(val) && val[0] && val[1])) {
    loadOrders(1)
  }
}

function toISODate(d) {
  if (!d) return null
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}
const statusOptions = ref([])
const sourceOptions = ref([])
const expandedRows = ref({})
const selectedOrders = ref([])
const batchStatus = ref('')
const batchLoading = ref(false)

watch(expandedRows, async (val) => {
  for (const orderId of Object.keys(val)) {
    const order = orders.value.find(o => String(o.id) === orderId)
    if (order && order.items === undefined) {
      order.items = null
      try {
        const items = await getOrderItems(Number(orderId))
        order.items = Array.isArray(items) ? items : (items.items ?? [])
      } catch {
        order.items = []
      }
    }
  }
})

onMounted(async () => {
  const ref_ = await getReference()
  statusOptions.value = ref_.order_statuses.map((s) => ({ label: s, value: s }))
  sourceOptions.value = (ref_.order_sources || []).map((s) => ({ label: s, value: s }))
  await loadOrders()
})

async function loadOrders(page = 1) {
  loading.value = true
  currentPage.value = page
  try {
    const params = { page }
    if (filterSearch.value) params.search = filterSearch.value
    if (filterStatus.value) params.status = filterStatus.value
    if (filterSource.value) params.source = filterSource.value
    if (filterDateRange.value?.[0]) params.date_from = toISODate(filterDateRange.value[0])
    if (filterDateRange.value?.[1]) params.date_to = toISODate(filterDateRange.value[1])
    if (sortField.value) {
      params.sort_by = sortField.value
      params.sort_order = sortOrderNum.value === -1 ? 'desc' : 'asc'
    }
    const result = await getOrders(params)
    orders.value = result.items ?? result
    totalRecords.value = result.total ?? orders.value.length
  } finally {
    loading.value = false
  }
}

function onPage(event) {
  loadOrders(event.page + 1)
}

function onSort(event) {
  sortField.value = event.sortField
  sortOrderNum.value = event.sortOrder
  loadOrders(1)
}

function resetFilters() {
  filterSearch.value = ''
  filterStatus.value = ''
  filterSource.value = ''
  filterDateRange.value = null
  sortField.value = null
  sortOrderNum.value = null
  loadOrders(1)
}

async function applyBatchStatus() {
  if (!batchStatus.value || !selectedOrders.value.length) return
  batchLoading.value = true
  try {
    const ids = selectedOrders.value.map(o => o.id)
    const result = await batchUpdateStatus(ids, batchStatus.value)
    toast.add({ severity: 'success', summary: 'Готово', detail: `Обновлено: ${result.updated.length}`, life: 3000 })
    selectedOrders.value = []
    batchStatus.value = ''
    await loadOrders(currentPage.value)
  } catch {
    toast.add({ severity: 'error', summary: 'Ошибка', detail: 'Не удалось обновить статусы', life: 3000 })
  } finally {
    batchLoading.value = false
  }
}

async function onKanbanStatusChange({ order, newStatus }) {
  await changeStatus(order, newStatus)
}

async function changeStatus(order, newStatus) {
  if (order.status === newStatus) return
  const prevStatus = order.status
  order.status = newStatus  // Оптимистичное обновление
  try {
    await updateOrderStatus(order.id, newStatus)
    toast.add({ severity: 'success', summary: 'Статус обновлён', detail: `${order.number}: ${newStatus}`, life: 3000 })
  } catch {
    order.status = prevStatus  // Откат
    toast.add({ severity: 'error', summary: 'Ошибка', detail: 'Не удалось сменить статус', life: 3000 })
  }
}

async function doExport() {
  exporting.value = true
  try {
    const params = {}
    if (filterStatus.value) params.status = filterStatus.value
    await exportOrders(params)
  } finally {
    exporting.value = false
  }
}
</script>

<style scoped>
:deep(.row-status-new td:first-child)       { border-left: 3px solid #3b82f6; }
:deep(.row-status-confirmed td:first-child) { border-left: 3px solid #f59e0b; }
:deep(.row-status-packing td:first-child)   { border-left: 3px solid #8b5cf6; }
:deep(.row-status-shipped td:first-child)   { border-left: 3px solid #6366f1; }
:deep(.row-status-delivered td:first-child) { border-left: 3px solid #10b981; }
:deep(.row-status-cancelled td:first-child) { border-left: 3px solid #ef4444; }

/* Компактный Select для инлайн-смены статуса */
:deep(.inline-status-select .p-select-label) { font-size: 0.75rem; padding: 0.25rem 0.5rem; }
:deep(.inline-status-select .p-select-dropdown) { width: 1.5rem; }
:deep(.inline-status-select) { border-radius: 9999px; }

/* Анимация панели групповых действий */
.slide-up-enter-active, .slide-up-leave-active { transition: all 0.25s ease; }
.slide-up-enter-from, .slide-up-leave-to { opacity: 0; transform: translate(-50%, 1rem); }
</style>
