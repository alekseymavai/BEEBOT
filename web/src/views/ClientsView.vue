<template>
  <div>
    <h2 class="text-2xl font-bold text-gray-800 mb-6">Клиенты</h2>

    <div class="bg-white rounded-xl border border-gray-100 shadow-sm">
      <DataTable
        :value="clients"
        :loading="loading"
        lazy
        paginator
        :rows="25"
        :total-records="total"
        row-hover
        class="text-sm"
        @page="onPage"
        @row-click="(e) => $router.push(`/clients/${e.data.id}`)"
      >
        <template #header>
          <div class="flex justify-between items-center p-1">
            <span class="text-base font-semibold text-gray-700">{{ total }} клиентов</span>
            <div class="flex gap-2 items-center">
              <IconField>
                <InputIcon class="pi pi-search" />
                <InputText
                  v-model="searchQuery"
                  placeholder="Поиск по имени, телефону, городу"
                  class="w-64"
                  @input="onSearch"
                />
              </IconField>
              <Button
                label="CSV"
                icon="pi pi-download"
                severity="secondary"
                size="small"
                :loading="exporting"
                @click="doExport"
              />
            </div>
          </div>
        </template>
        <template #empty>
          <div class="text-center py-8 text-gray-400">Клиенты не найдены</div>
        </template>
        <Column field="name" header="ФИО" sortable />
        <Column field="phone" header="Телефон">
          <template #body="{ data }">{{ data.phone || '—' }}</template>
        </Column>
        <Column field="city" header="Город">
          <template #body="{ data }">{{ data.city || '—' }}</template>
        </Column>
        <Column field="telegram_username" header="Telegram">
          <template #body="{ data }">
            <span v-if="data.telegram_username" class="text-blue-500">
              @{{ data.telegram_username }}
            </span>
            <span v-else class="text-gray-300">—</span>
          </template>
        </Column>
        <Column field="source" header="Источник">
          <template #body="{ data }">
            <Tag :value="data.source || 'Неизвестно'" severity="secondary" />
          </template>
        </Column>
      </DataTable>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import DataTable from 'primevue/datatable'
import Column from 'primevue/column'
import InputText from 'primevue/inputtext'
import IconField from 'primevue/iconfield'
import InputIcon from 'primevue/inputicon'
import Tag from 'primevue/tag'
import { getClientsPaged, exportClients } from '../api.js'

const loading = ref(true)
const exporting = ref(false)
const clients = ref([])
const total = ref(0)
const searchQuery = ref('')
let searchTimer = null

onMounted(() => loadClients())

async function loadClients(page = 1) {
  loading.value = true
  try {
    const params = { page }
    if (searchQuery.value) params.search = searchQuery.value
    const result = await getClientsPaged(params)
    clients.value = result.items ?? result
    total.value = result.total ?? clients.value.length
  } finally {
    loading.value = false
  }
}

function onPage(event) {
  loadClients(event.page + 1)
}

function onSearch() {
  clearTimeout(searchTimer)
  searchTimer = setTimeout(() => loadClients(1), 400)
}

async function doExport() {
  exporting.value = true
  try {
    await exportClients()
  } finally {
    exporting.value = false
  }
}
</script>
