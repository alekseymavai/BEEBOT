<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-2xl font-bold text-gray-800">Заказы</h2>
      <RouterLink to="/orders/new">
        <Button label="Новый заказ" icon="pi pi-plus" size="small" />
      </RouterLink>
    </div>

    <!-- Фильтры -->
    <div class="bg-white rounded-xl border border-gray-100 shadow-sm p-4 mb-4 flex gap-3 flex-wrap">
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

    <!-- Таблица заказов -->
    <div class="bg-white rounded-xl border border-gray-100 shadow-sm">
      <DataTable
        :value="orders"
        :loading="loading"
        dataKey="id"
        lazy
        paginator
        :rows="50"
        :total-records="totalRecords"
        v-model:expandedRows="expandedRows"
        row-hover
        class="text-sm"
        @page="onPage"
        @row-click="(e) => $router.push(`/orders/${e.data.id}`)"
      >
        <template #empty>
          <div class="text-center py-8 text-gray-400">Заказов нет</div>
        </template>
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
        <Column field="status" header="Статус" sortable>
          <template #body="{ data }">
            <StatusBadge :status="data.status" />
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
import { RouterLink } from 'vue-router'
import DataTable from 'primevue/datatable'
import Column from 'primevue/column'
import Button from 'primevue/button'
import Select from 'primevue/select'
import { getOrders, getReference, exportOrders, getOrderItems } from '../api.js'
import StatusBadge from '../components/StatusBadge.vue'
import { formatDate, formatMoney } from '../utils.js'

const loading = ref(true)
const exporting = ref(false)
const orders = ref([])
const totalRecords = ref(0)
const currentPage = ref(1)
const filterStatus = ref('')
const filterSource = ref('')
const statusOptions = ref([])
const sourceOptions = ref([])
const expandedRows = ref({})

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
    if (filterStatus.value) params.status = filterStatus.value
    if (filterSource.value) params.source = filterSource.value
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

function resetFilters() {
  filterStatus.value = ''
  filterSource.value = ''
  loadOrders(1)
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
