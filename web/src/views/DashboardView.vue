<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-2xl font-bold text-gray-800">Дашборд</h2>
      <div class="flex gap-2 items-center">
        <!-- Переключатель периода -->
        <div class="flex rounded-lg border border-gray-200 overflow-hidden text-sm">
          <button
            v-for="p in periodOptions"
            :key="p.value"
            :class="[
              'px-3 py-1.5 transition-colors',
              activePeriod === p.value
                ? 'bg-amber-500 text-white font-medium'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            ]"
            @click="setPeriod(p.value)"
          >{{ p.label }}</button>
        </div>
        <!-- PDF-отчёты -->
        <Button
          v-for="p in reportPeriods"
          :key="p.value"
          :label="p.label"
          icon="pi pi-file-pdf"
          size="small"
          severity="secondary"
          :loading="reportLoading === p.value"
          @click="downloadReport(p.value)"
        />
      </div>
    </div>

    <!-- Блок «Требуют внимания» -->
    <div v-if="alerts.stale_new || alerts.stale_confirmed || alerts.low_stock"
         class="flex gap-3 flex-wrap mb-6">
      <RouterLink
        v-if="alerts.stale_new"
        to="/orders?status=Новый"
        class="flex items-center gap-2 px-4 py-2.5 bg-red-50 border border-red-200 rounded-xl text-sm font-medium text-red-700 hover:bg-red-100 transition-colors"
      >
        <i class="pi pi-exclamation-circle" />
        {{ alerts.stale_new }} новых без ответа &gt;24ч
      </RouterLink>
      <RouterLink
        v-if="alerts.stale_confirmed"
        to="/orders?status=Подтверждён"
        class="flex items-center gap-2 px-4 py-2.5 bg-amber-50 border border-amber-200 rounded-xl text-sm font-medium text-amber-700 hover:bg-amber-100 transition-colors"
      >
        <i class="pi pi-clock" />
        {{ alerts.stale_confirmed }} подтверждённых &gt;3 дней без отправки
      </RouterLink>
      <RouterLink
        v-if="alerts.low_stock"
        to="/stock"
        class="flex items-center gap-2 px-4 py-2.5 bg-orange-50 border border-orange-200 rounded-xl text-sm font-medium text-orange-700 hover:bg-orange-100 transition-colors"
      >
        <i class="pi pi-box" />
        {{ alerts.low_stock }} товаров на исходе
      </RouterLink>
    </div>

    <!-- Карточки статистики -->
    <div v-if="loading" class="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
      <Skeleton v-for="i in 6" :key="i" height="100px" class="rounded-xl" />
    </div>

    <div v-else class="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
      <StatCard
        icon="pi-list"
        icon-color="text-amber-600"
        bg-color="bg-amber-50"
        label="Всего заказов"
        :value="stats.total_orders"
      />
      <StatCard
        icon="pi-users"
        icon-color="text-blue-600"
        bg-color="bg-blue-50"
        label="Клиентов"
        :value="stats.total_clients"
      />
      <StatCard
        icon="pi-wallet"
        icon-color="text-green-600"
        bg-color="bg-green-50"
        label="Общая выручка"
        :value="formatMoney(stats.total_revenue)"
      />
      <StatCard
        icon="pi-chart-line"
        icon-color="text-purple-600"
        bg-color="bg-purple-50"
        label="Средний чек"
        :value="formatMoney(stats.avg_order)"
      />
      <StatCard
        icon="pi-shopping-cart"
        icon-color="text-orange-600"
        bg-color="bg-orange-50"
        label="Новых заказов"
        :value="stats.new_orders"
      />
      <StatCard
        icon="pi-truck"
        icon-color="text-teal-600"
        bg-color="bg-teal-50"
        label="Доставленных"
        :value="stats.delivered_orders"
      />
    </div>

    <!-- Графики -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      <!-- Выручка по месяцам -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
        <h3 class="font-semibold text-gray-700 mb-4">Выручка по месяцам</h3>
        <Chart v-if="revenueChartData" type="bar" :data="revenueChartData" :options="revenueOptions" class="h-64" />
        <Skeleton v-else height="256px" class="rounded-lg" />
      </div>

      <!-- Воронка статусов -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
        <h3 class="font-semibold text-gray-700 mb-4">Заказы по статусам</h3>
        <Chart v-if="funnelChartData" type="bar" :data="funnelChartData" :options="funnelOptions" class="h-64" />
        <Skeleton v-else height="256px" class="rounded-lg" />
      </div>

      <!-- Заказы по месяцам (линия) -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
        <h3 class="font-semibold text-gray-700 mb-4">Количество заказов</h3>
        <Chart v-if="countChartData" type="line" :data="countChartData" :options="countOptions" class="h-64" />
        <Skeleton v-else height="256px" class="rounded-lg" />
      </div>

      <!-- Способы доставки (пончик) -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
        <h3 class="font-semibold text-gray-700 mb-4">Способы доставки</h3>
        <div class="flex justify-center">
          <Chart v-if="deliveryChartData" type="doughnut" :data="deliveryChartData" :options="doughnutOptions" style="max-height: 256px; max-width: 320px;" />
          <Skeleton v-else height="256px" width="320px" class="rounded-lg" />
        </div>
      </div>

      <!-- Топ-5 товаров -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5 lg:col-span-2">
        <h3 class="font-semibold text-gray-700 mb-4">Топ товаров за период</h3>
        <Skeleton v-if="!topProducts" height="160px" class="rounded-lg" />
        <table v-else class="w-full text-sm">
          <thead>
            <tr class="text-xs text-gray-500 border-b border-gray-100">
              <th class="text-left pb-2 font-medium">Товар</th>
              <th class="text-right pb-2 font-medium w-20">Шт</th>
              <th class="text-right pb-2 font-medium w-28">Выручка</th>
              <th class="text-right pb-2 font-medium w-16">Доля</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="p in topProducts" :key="p.name" class="border-b border-gray-50 last:border-0">
              <td class="py-2 text-gray-700">{{ p.name }}</td>
              <td class="py-2 text-right text-gray-600">{{ p.qty }}</td>
              <td class="py-2 text-right font-medium">{{ formatMoney(p.revenue) }}</td>
              <td class="py-2 text-right">
                <span class="text-xs px-1.5 py-0.5 bg-amber-50 text-amber-700 rounded">{{ p.share }}%</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Последние заказы -->
    <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
      <div class="flex items-center justify-between mb-4">
        <h3 class="font-semibold text-gray-700">Последние заказы</h3>
        <RouterLink to="/orders" class="text-sm text-amber-600 hover:text-amber-700 font-medium">
          Все заказы →
        </RouterLink>
      </div>

      <DataTable
        :value="recentOrders"
        :loading="ordersLoading"
        v-model:expandedRows="expandedRows"
        :row-class="rowStatusClass"
        row-hover
        class="text-sm"
        @row-click="(e) => $router.push(`/orders/${e.data.id}`)"
      >
        <Column expander style="width:3rem" />
        <Column field="number" header="Номер" style="width:110px" />
        <Column field="client_name" header="Клиент">
          <template #body="{ data }">{{ data.client_name || `Клиент #${data.client_id}` }}</template>
        </Column>
        <Column field="date" header="Дата" style="width:110px">
          <template #body="{ data }">{{ formatDate(data.date) }}</template>
        </Column>
        <Column field="status" header="Статус" style="width:130px">
          <template #body="{ data }">
            <StatusBadge :status="data.status" />
          </template>
        </Column>
        <Column field="total" header="Сумма" style="width:110px">
          <template #body="{ data }">{{ formatMoney(data.total) }}</template>
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
import Skeleton from 'primevue/skeleton'
import Chart from 'primevue/chart'
import Button from 'primevue/button'
import { getDashboard, getDashboardCharts, getDashboardAlerts, getOrders, getOrderItems, downloadSalesReport } from '../api.js'
import StatCard from '../components/StatCard.vue'
import StatusBadge from '../components/StatusBadge.vue'
import { formatDate, formatMoney } from '../utils.js'

const periodOptions = [
  { label: 'Сегодня', value: 'today' },
  { label: '7 дней', value: '7d' },
  { label: '30 дней', value: '30d' },
  { label: 'Квартал', value: '90d' },
  { label: 'Всё время', value: 'all' },
]
const activePeriod = ref('all')

async function setPeriod(period) {
  activePeriod.value = period
  await loadDashboard()
}

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
const ordersLoading = ref(true)
const alerts = ref({ stale_new: 0, stale_confirmed: 0, low_stock: 0 })
const reportLoading = ref(null)
const reportPeriods = [
  { label: 'PDF 30 дней', value: '30d' },
  { label: 'PDF квартал', value: '90d' },
  { label: 'PDF год', value: '365d' },
]

async function downloadReport(period) {
  reportLoading.value = period
  try {
    await downloadSalesReport(period)
  } finally {
    reportLoading.value = null
  }
}
const expandedRows = ref({})

// При раскрытии строки — загрузить позиции заказа если ещё не загружены
watch(expandedRows, async (val) => {
  for (const orderId of Object.keys(val)) {
    const order = recentOrders.value.find(o => String(o.id) === orderId)
    if (order && order.items === undefined) {
      order.items = null  // индикатор загрузки
      try {
        const items = await getOrderItems(Number(orderId))
        order.items = Array.isArray(items) ? items : (items.items ?? [])
      } catch {
        order.items = []
      }
    }
  }
})
const stats = ref({
  total_orders: 0,
  total_clients: 0,
  total_revenue: 0,
  avg_order: 0,
  new_orders: 0,
  delivered_orders: 0
})
const recentOrders = ref([])

// Chart data
const revenueChartData = ref(null)
const countChartData = ref(null)
const funnelChartData = ref(null)
const deliveryChartData = ref(null)
const topProducts = ref(null)

// Chart options
const revenueOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      callbacks: {
        label: (ctx) => `${ctx.parsed.y.toLocaleString('ru-RU')} \u20BD`
      }
    }
  },
  scales: {
    y: {
      beginAtZero: true,
      ticks: {
        callback: (v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v
      }
    }
  }
}

const countOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    y: { beginAtZero: true, ticks: { stepSize: 1 } }
  }
}

const funnelOptions = {
  indexAxis: 'y',
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { beginAtZero: true, ticks: { stepSize: 1 } }
  }
}

const doughnutOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { position: 'bottom', labels: { padding: 16 } }
  }
}

const STATUS_COLORS = {
  'Новый': '#3b82f6',
  'Подтверждён': '#f59e0b',
  'В сборке': '#8b5cf6',
  'Отправлен': '#06b6d4',
  'Доставлен': '#10b981',
  'Отменён': '#ef4444',
}

const DELIVERY_COLORS = ['#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ef4444']

async function loadDashboard() {
  loading.value = true
  revenueChartData.value = null
  countChartData.value = null
  funnelChartData.value = null
  deliveryChartData.value = null
  topProducts.value = null

  const period = activePeriod.value
  try {
    const [dashData, ordersResult, alertsData] = await Promise.all([
      getDashboard({ period }),
      getOrders({ per_page: 10 }),
      getDashboardAlerts(),
    ])
    alerts.value = alertsData
    stats.value = dashData
    recentOrders.value = ordersResult.items ?? ordersResult
  } finally {
    loading.value = false
    ordersLoading.value = false
  }

  // Load charts data (non-blocking)
  try {
    const charts = await getDashboardCharts({ period })

    revenueChartData.value = {
      labels: charts.monthly.labels,
      datasets: [{
        label: 'Выручка',
        data: charts.monthly.revenue,
        backgroundColor: '#f59e0b',
        borderRadius: 6,
        borderSkipped: false,
      }]
    }

    countChartData.value = {
      labels: charts.monthly.labels,
      datasets: [{
        label: 'Заказов',
        data: charts.monthly.count,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 4,
        pointBackgroundColor: '#3b82f6',
      }]
    }

    funnelChartData.value = {
      labels: charts.funnel.labels,
      datasets: [{
        data: charts.funnel.data,
        backgroundColor: charts.funnel.labels.map(s => STATUS_COLORS[s] || '#9ca3af'),
        borderRadius: 6,
        borderSkipped: false,
      }]
    }

    deliveryChartData.value = {
      labels: charts.delivery.labels,
      datasets: [{
        data: charts.delivery.data,
        backgroundColor: DELIVERY_COLORS.slice(0, charts.delivery.labels.length),
        hoverOffset: 8,
      }]
    }

    topProducts.value = charts.top_products || []
  } catch (e) {
    console.error('Charts load error:', e)
  }
}

onMounted(loadDashboard)
</script>

<style scoped>
:deep(.row-status-new td:first-child)       { border-left: 3px solid #3b82f6; }
:deep(.row-status-confirmed td:first-child) { border-left: 3px solid #f59e0b; }
:deep(.row-status-packing td:first-child)   { border-left: 3px solid #8b5cf6; }
:deep(.row-status-shipped td:first-child)   { border-left: 3px solid #6366f1; }
:deep(.row-status-delivered td:first-child) { border-left: 3px solid #10b981; }
:deep(.row-status-cancelled td:first-child) { border-left: 3px solid #ef4444; }
</style>
