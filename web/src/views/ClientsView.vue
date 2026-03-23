<template>
  <div>
    <h2 class="text-2xl font-bold text-gray-800 mb-6">Клиенты</h2>

    <div class="bg-white rounded-xl border border-gray-100 shadow-sm">
      <DataTable
        :value="clients"
        :loading="loading"
        paginator
        :rows="25"
        row-hover
        filter-display="row"
        :global-filter-fields="['name', 'phone', 'city']"
        v-model:filters="filters"
        class="text-sm"
        @row-click="(e) => $router.push(`/clients/${e.data.id}`)"
      >
        <template #header>
          <div class="flex justify-between items-center p-1">
            <span class="text-base font-semibold text-gray-700">{{ clients.length }} клиентов</span>
            <div class="flex gap-2 items-center">
              <IconField>
                <InputIcon class="pi pi-search" />
                <InputText
                  v-model="searchQuery"
                  placeholder="Поиск по имени, телефону, городу"
                  class="w-64"
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
import { ref, computed, onMounted } from 'vue'
import DataTable from 'primevue/datatable'
import Column from 'primevue/column'
import InputText from 'primevue/inputtext'
import IconField from 'primevue/iconfield'
import InputIcon from 'primevue/inputicon'
import Tag from 'primevue/tag'
import { getClients, exportClients } from '../api.js'

const loading = ref(true)
const exporting = ref(false)
const allClients = ref([])
const searchQuery = ref('')
const filters = ref({})

const clients = computed(() => {
  const q = searchQuery.value.toLowerCase()
  if (!q) return allClients.value
  return allClients.value.filter(
    (c) =>
      c.name?.toLowerCase().includes(q) ||
      c.phone?.includes(q) ||
      c.city?.toLowerCase().includes(q)
  )
})

onMounted(async () => {
  try {
    allClients.value = await getClients()
  } finally {
    loading.value = false
  }
})

async function doExport() {
  exporting.value = true
  try {
    await exportClients()
  } finally {
    exporting.value = false
  }
}
</script>
