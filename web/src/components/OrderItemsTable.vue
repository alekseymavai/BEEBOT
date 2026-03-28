<template>
  <div class="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
    <div class="flex justify-between items-center mb-4">
      <h3 class="font-semibold text-gray-700">Позиции заказа</h3>
      <Button
        v-if="editable"
        icon="pi pi-plus"
        label="Добавить"
        size="small"
        severity="success"
        outlined
        @click="showAddItem = true"
      />
    </div>

    <DataTable :value="items" class="text-sm">
      <template #empty>
        <div class="py-4 text-center text-sm">
          <div class="text-gray-400 mb-1">Позиции не сохранены в CRM</div>
          <div v-if="orderTotal" class="text-gray-600">
            Сумма заказа: <strong>{{ formatMoney(orderTotal) }}</strong>
          </div>
        </div>
      </template>
      <Column field="product_name" header="Товар">
        <template #body="{ data }">
          {{ data.product_name || products.find(p => p.id === data.product_id)?.name || `Товар #${data.product_id}` }}
        </template>
      </Column>
      <Column field="quantity" header="Кол-во" style="width:100px">
        <template #body="{ data }">
          <span v-if="editingItem !== data.id">{{ data.quantity }}</span>
          <InputNumber v-else v-model="editItemQty" size="small" class="w-16" :min="1" />
        </template>
      </Column>
      <Column field="unit_price" header="Цена" style="width:120px">
        <template #body="{ data }">
          <span v-if="editingItem !== data.id">{{ formatMoney(data.unit_price) }}</span>
          <InputNumber v-else v-model="editItemPrice" size="small" class="w-24" suffix=" ₽" :min="0" />
        </template>
      </Column>
      <Column field="total" header="Сумма" style="width:100px">
        <template #body="{ data }">{{ formatMoney(data.total) }}</template>
      </Column>
      <Column v-if="editable" header="" style="width:80px">
        <template #body="{ data }">
          <div class="flex gap-1">
            <template v-if="editingItem !== data.id">
              <Button icon="pi pi-pencil" size="small" text severity="secondary" @click="startEditItem(data)" />
              <Button icon="pi pi-trash" size="small" text severity="danger" @click="$emit('remove-item', data)" :loading="deletingItem === data.id" />
            </template>
            <template v-else>
              <Button icon="pi pi-check" size="small" text severity="success" @click="saveItem(data)" :loading="savingItem" />
              <Button icon="pi pi-times" size="small" text severity="secondary" @click="editingItem = null" />
            </template>
          </div>
        </template>
      </Column>
    </DataTable>

    <!-- Диалог добавления позиции -->
    <Dialog v-model:visible="showAddItem" header="Добавить товар" modal class="w-96">
      <div class="space-y-3">
        <div>
          <label class="text-sm text-gray-500 mb-1 block">Товар</label>
          <Select
            v-model="newItem.product"
            :options="products"
            optionLabel="name"
            placeholder="Выберите товар"
            filter
            class="w-full"
          />
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="text-sm text-gray-500 mb-1 block">Количество</label>
            <InputNumber v-model="newItem.quantity" :min="1" class="w-full" />
          </div>
          <div>
            <label class="text-sm text-gray-500 mb-1 block">Цена</label>
            <InputNumber v-model="newItem.price" suffix=" ₽" :min="0" class="w-full" />
          </div>
        </div>
      </div>
      <template #footer>
        <Button label="Отмена" severity="secondary" text @click="showAddItem = false" />
        <Button label="Добавить" @click="addItem" :loading="addingItem" :disabled="!newItem.product" />
      </template>
    </Dialog>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import DataTable from 'primevue/datatable'
import Column from 'primevue/column'
import Button from 'primevue/button'
import InputNumber from 'primevue/inputnumber'
import Select from 'primevue/select'
import Dialog from 'primevue/dialog'
import { formatMoney } from '../utils.js'

defineProps({
  items: { type: Array, default: () => [] },
  products: { type: Array, default: () => [] },
  editable: { type: Boolean, default: false },
  deletingItem: { type: [Number, null], default: null },
  orderTotal: { type: Number, default: null },
})

const emit = defineEmits(['save-item', 'remove-item', 'add-item'])

const editingItem = ref(null)
const editItemQty = ref(1)
const editItemPrice = ref(0)
const savingItem = ref(false)

const showAddItem = ref(false)
const addingItem = ref(false)
const newItem = ref({ product: null, quantity: 1, price: 0 })

watch(() => newItem.value.product, (p) => {
  if (p && p.price) newItem.value.price = p.price
})

function startEditItem(item) {
  editingItem.value = item.id
  editItemQty.value = item.quantity
  editItemPrice.value = item.unit_price
}

async function saveItem(item) {
  savingItem.value = true
  try {
    await emit('save-item', {
      itemId: item.id,
      quantity: editItemQty.value,
      unit_price: editItemPrice.value
    })
    editingItem.value = null
  } finally {
    savingItem.value = false
  }
}

async function addItem() {
  if (!newItem.value.product) return
  addingItem.value = true
  try {
    await emit('add-item', {
      product_id: newItem.value.product.id,
      quantity: newItem.value.quantity,
      unit_price: newItem.value.price
    })
    showAddItem.value = false
    newItem.value = { product: null, quantity: 1, price: 0 }
  } finally {
    addingItem.value = false
  }
}
</script>
