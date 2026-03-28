<template>
  <div>
    <!-- Заголовок -->
    <div class="flex items-center gap-3 mb-6">
      <Button icon="pi pi-arrow-left" severity="secondary" text @click="$router.back()" />
      <h2 class="text-2xl font-bold text-gray-800">
        Заказ {{ order?.number || '#' + orderId }}
      </h2>
      <StatusBadge v-if="order" :status="order.status" class="ml-1" />
      <span v-if="order && order.editable" class="text-xs text-green-600 bg-green-50 px-2 py-1 rounded-full">
        можно редактировать
      </span>
    </div>

    <div v-if="loading" class="space-y-4">
      <Skeleton height="200px" class="rounded-xl" />
      <Skeleton height="150px" class="rounded-xl" />
    </div>

    <div v-else-if="order" class="space-y-4">
      <!-- Основная информация -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <OrderDetailsCard :order="order" @update-field="handleUpdateField" />
        <OrderDeliveryCard
          :order="order"
          :delivery-methods="deliveryMethods"
          :saving-tracking="savingTracking"
          @update-field="handleUpdateField"
          @save-tracking="handleSaveTracking"
        />
      </div>

      <OrderStatusBar
        :current-status="order.status"
        :status-options="statusOptions"
        @change-status="handleChangeStatus"
      />

      <!-- Партия отправки -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
        <h3 class="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
          <i class="pi pi-send text-gray-400" />
          Партия отправки
        </h3>
        <div class="flex items-center gap-3">
          <Select
            v-model="selectedBatchId"
            :options="batches"
            option-label="label"
            option-value="value"
            placeholder="Не назначена"
            class="w-64"
            show-clear
            @change="handleBatchChange"
          />
          <span v-if="batchSaving" class="text-sm text-gray-400">Сохранение...</span>
          <RouterLink
            v-if="selectedBatchId"
            to="/batches"
            class="text-sm text-amber-600 hover:text-amber-700"
          >Все партии →</RouterLink>
        </div>
      </div>

      <!-- Чеклист подготовки к отправке -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
        <h3 class="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
          <i class="pi pi-check-square text-gray-400" />
          Подготовка к отправке
        </h3>
        <div class="flex flex-wrap gap-4">
          <label class="flex items-center gap-2 cursor-pointer text-sm text-gray-700">
            <input type="checkbox" class="rounded"
              :checked="order.stock_checked"
              @change="handleChecklist('stock_checked', $event.target.checked)" />
            Наличие проверено
          </label>
          <label class="flex items-center gap-2 cursor-pointer text-sm text-gray-700">
            <input type="checkbox" class="rounded"
              :checked="order.cdek_confirmed"
              @change="handleChecklist('cdek_confirmed', $event.target.checked)" />
            Адрес СДЭК уточнён
          </label>
          <label class="flex items-center gap-2 cursor-pointer text-sm text-gray-700">
            <input type="checkbox" class="rounded"
              :checked="order.client_notified"
              @change="handleChecklist('client_notified', $event.target.checked)" />
            Клиент оповещён
          </label>
        </div>
      </div>

      <OrderItemsTable
        :items="order.items"
        :products="products"
        :editable="order.editable"
        :deleting-item="deletingItem"
        :order-total="order.total"
        @save-item="handleSaveItem"
        @remove-item="handleRemoveItem"
        @add-item="handleAddItem"
      />

      <!-- История статусов -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
        <h3 class="text-sm font-semibold text-gray-700 mb-4 flex items-center gap-2">
          <i class="pi pi-history text-gray-400" />
          История статусов
        </h3>
        <div v-if="historyLoading" class="text-sm text-gray-400">Загрузка...</div>
        <div v-else-if="!history.length" class="text-sm text-gray-400">Нет записей</div>
        <ol v-else class="relative border-l border-gray-200 ml-3">
          <li v-for="item in history" :key="item.id" class="mb-4 ml-4">
            <div class="absolute -left-1.5 mt-1.5 h-3 w-3 rounded-full border border-white bg-gray-300" />
            <p class="text-xs text-gray-400">{{ item.date }}</p>
            <p class="text-sm font-medium text-gray-700">
              <span class="text-gray-400">{{ item.from_status }}</span>
              <i class="pi pi-arrow-right text-xs mx-1 text-gray-400" />
              <span class="text-gray-800">{{ item.to_status }}</span>
            </p>
            <p v-if="item.comment" class="text-xs text-gray-500 mt-0.5">{{ item.comment }}</p>
          </li>
        </ol>
      </div>
    </div>

    <div v-else class="text-center py-16 text-gray-400">
      <i class="pi pi-exclamation-triangle text-4xl mb-3 block" />
      Заказ не найден
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useToast } from 'primevue/usetoast'
import Button from 'primevue/button'
import Select from 'primevue/select'
import Skeleton from 'primevue/skeleton'
import { RouterLink } from 'vue-router'
import {
  getOrder, updateOrderStatus, updateOrderTracking, updateOrder,
  addOrderItem, updateOrderItem, deleteOrderItem,
  getReference, getProducts, getOrderHistory, updateOrderChecklist,
  getBatches, assignOrderBatch
} from '../api.js'
import StatusBadge from '../components/StatusBadge.vue'
import OrderDetailsCard from '../components/OrderDetailsCard.vue'
import OrderDeliveryCard from '../components/OrderDeliveryCard.vue'
import OrderStatusBar from '../components/OrderStatusBar.vue'
import OrderItemsTable from '../components/OrderItemsTable.vue'

const route = useRoute()
const toast = useToast()
const orderId = route.params.id

const loading = ref(true)
const order = ref(null)
const statusOptions = ref([])
const deliveryMethods = ref([])
const products = ref([])
const savingTracking = ref(false)
const deletingItem = ref(null)
const history = ref([])
const historyLoading = ref(false)
const batches = ref([])
const selectedBatchId = ref(null)
const batchSaving = ref(false)

onMounted(async () => {
  const [orderData, refData, productsData, batchesData] = await Promise.all([
    getOrder(orderId),
    getReference(),
    getProducts(),
    getBatches()
  ])
  order.value = orderData
  statusOptions.value = refData.order_statuses
  deliveryMethods.value = refData.delivery_methods
  products.value = productsData
  const batchList = batchesData.items ?? batchesData
  batches.value = batchList.map(b => ({
    value: b.id,
    label: `${b.date ? b.date.slice(0, 10) : '?'} — ${b.delivery_method || 'без метода'}`
  }))
  selectedBatchId.value = orderData.batch_id ?? null
  loading.value = false

  historyLoading.value = true
  try {
    history.value = await getOrderHistory(orderId)
  } catch (_) {
    // история не критична
  } finally {
    historyLoading.value = false
  }
})

async function reloadOrder() {
  order.value = await getOrder(orderId)
}

async function handleBatchChange() {
  batchSaving.value = true
  try {
    await assignOrderBatch(orderId, selectedBatchId.value || null)
    toast.add({ severity: 'success', summary: 'Партия обновлена', life: 2000 })
  } catch {
    toast.add({ severity: 'error', summary: 'Ошибка', detail: 'Не удалось обновить партию', life: 3000 })
  } finally {
    batchSaving.value = false
  }
}

function showSuccess(msg) {
  toast.add({ severity: 'success', summary: 'Готово', detail: msg, life: 3000 })
}

function showError(e, fallback = 'Не удалось сохранить') {
  toast.add({ severity: 'error', summary: 'Ошибка', detail: e?.response?.data?.detail || fallback, life: 3000 })
}

async function handleChangeStatus(newStatus) {
  try {
    await updateOrderStatus(orderId, newStatus)
    order.value.status = newStatus
    order.value.editable = ['Новый', 'Подтверждён', 'В сборке'].includes(newStatus)
    showSuccess(`Статус изменён на "${newStatus}"`)
    // Обновить историю после смены статуса
    try { history.value = await getOrderHistory(orderId) } catch (_) {}
  } catch (e) {
    showError(e, 'Не удалось изменить статус')
  }
}

async function handleSaveTracking(trackingNumber) {
  savingTracking.value = true
  try {
    await updateOrderTracking(orderId, trackingNumber)
    order.value.tracking_number = trackingNumber
    showSuccess('Трек-номер обновлён')
  } catch (e) {
    showError(e)
  } finally {
    savingTracking.value = false
  }
}

async function handleUpdateField(fields) {
  try {
    await updateOrder(orderId, fields)
    await reloadOrder()
    const fieldName = Object.keys(fields)[0]
    const labels = {
      delivery_cost: 'Стоимость доставки обновлена',
      delivery_address: 'Адрес обновлён',
      delivery_method: 'Способ доставки обновлён',
      comment: 'Комментарий обновлён',
    }
    showSuccess(labels[fieldName] || 'Сохранено')
  } catch (e) {
    showError(e)
  }
}

async function handleSaveItem({ itemId, quantity, unit_price }) {
  try {
    await updateOrderItem(orderId, itemId, { quantity, unit_price })
    await reloadOrder()
    showSuccess('Позиция обновлена')
  } catch (e) {
    showError(e)
  }
}

async function handleRemoveItem(item) {
  if (!confirm(`Удалить "${item.product_name || 'позицию'}"?`)) return
  deletingItem.value = item.id
  try {
    await deleteOrderItem(orderId, item.id)
    await reloadOrder()
    showSuccess('Позиция удалена')
  } catch (e) {
    showError(e, 'Не удалось удалить')
  } finally {
    deletingItem.value = null
  }
}

async function handleChecklist(field, value) {
  try {
    await updateOrderChecklist(orderId, { [field]: value })
    order.value[field] = value
  } catch (e) {
    showError(e, 'Не удалось сохранить')
  }
}

async function handleAddItem(itemData) {
  try {
    await addOrderItem(orderId, itemData)
    await reloadOrder()
    showSuccess('Товар добавлен')
  } catch (e) {
    showError(e, 'Не удалось добавить')
  }
}
</script>
