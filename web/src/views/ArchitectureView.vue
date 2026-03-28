<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <div>
        <h1 class="text-xl font-bold text-gray-800">Архитектура системы</h1>
        <p class="text-sm text-gray-500 mt-0.5">Диаграммы и описание компонентов BEEBOT</p>
      </div>
    </div>

    <!-- Загрузка -->
    <div v-if="loading" class="flex items-center justify-center py-24 text-gray-400 gap-3">
      <i class="pi pi-spin pi-spinner text-2xl" />
      <span>Загрузка документа...</span>
    </div>

    <!-- Ошибка -->
    <div v-else-if="error" class="bg-red-50 border border-red-200 rounded-xl p-6 text-red-700">
      <i class="pi pi-exclamation-circle mr-2" />{{ error }}
    </div>

    <!-- Контент -->
    <div
      v-else
      ref="contentRef"
      class="architecture-content bg-white rounded-xl border border-gray-200 p-8 prose prose-gray max-w-none"
      v-html="htmlContent"
    />
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { marked } from 'marked'
import mermaid from 'mermaid'
import api from '../api.js'

const loading = ref(true)
const error = ref(null)
const htmlContent = ref('')
const contentRef = ref(null)

mermaid.initialize({
  startOnLoad: false,
  theme: 'default',
  fontFamily: 'inherit',
  flowchart: { useMaxWidth: true, htmlLabels: true },
  sequence: { useMaxWidth: true },
})

onMounted(async () => {
  try {
    const { data } = await api.get('/api/docs/architecture')
    htmlContent.value = marked(data)

    await nextTick()

    // Преобразовать <pre><code class="language-mermaid">...</code></pre>
    // в <div class="mermaid">...</div> для рендера Mermaid
    const codeBlocks = contentRef.value?.querySelectorAll('code.language-mermaid') ?? []
    codeBlocks.forEach((code) => {
      const div = document.createElement('div')
      div.className = 'mermaid'
      div.textContent = code.textContent
      code.parentElement.replaceWith(div)
    })

    if (codeBlocks.length > 0) {
      await mermaid.run({ nodes: contentRef.value.querySelectorAll('.mermaid') })
    }
  } catch (e) {
    error.value = e.response?.data?.detail ?? e.message ?? 'Не удалось загрузить документ'
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.architecture-content :deep(h1) {
  font-size: 1.5rem;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #f59e0b;
}
.architecture-content :deep(h2) {
  font-size: 1.2rem;
  font-weight: 700;
  color: #374151;
  margin-top: 2rem;
  margin-bottom: 0.75rem;
}
.architecture-content :deep(h3) {
  font-size: 1rem;
  font-weight: 600;
  color: #4b5563;
  margin-top: 1.5rem;
  margin-bottom: 0.5rem;
}
.architecture-content :deep(p) {
  color: #4b5563;
  line-height: 1.6;
  margin-bottom: 0.75rem;
}
.architecture-content :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-size: 0.875rem;
}
.architecture-content :deep(th) {
  background: #fef3c7;
  color: #92400e;
  font-weight: 600;
  padding: 0.5rem 0.75rem;
  text-align: left;
  border: 1px solid #fde68a;
}
.architecture-content :deep(td) {
  padding: 0.5rem 0.75rem;
  border: 1px solid #e5e7eb;
  color: #374151;
}
.architecture-content :deep(tr:nth-child(even) td) {
  background: #f9fafb;
}
.architecture-content :deep(pre) {
  background: #1f2937;
  color: #f3f4f6;
  border-radius: 0.5rem;
  padding: 1rem;
  overflow-x: auto;
  font-size: 0.8rem;
  margin: 1rem 0;
}
.architecture-content :deep(code:not(.mermaid *)) {
  background: #f3f4f6;
  color: #b45309;
  padding: 0.1rem 0.3rem;
  border-radius: 0.25rem;
  font-size: 0.85em;
}
.architecture-content :deep(pre code) {
  background: none;
  color: inherit;
  padding: 0;
}
.architecture-content :deep(.mermaid) {
  background: #fafafa;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 1.5rem;
  margin: 1rem 0;
  text-align: center;
  overflow-x: auto;
}
.architecture-content :deep(blockquote) {
  border-left: 3px solid #f59e0b;
  padding-left: 1rem;
  color: #6b7280;
  font-style: italic;
  margin: 1rem 0;
}
.architecture-content :deep(hr) {
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 2rem 0;
}
.architecture-content :deep(a) {
  color: #d97706;
  text-decoration: underline;
}
.architecture-content :deep(ul), .architecture-content :deep(ol) {
  padding-left: 1.5rem;
  margin-bottom: 0.75rem;
}
.architecture-content :deep(li) {
  color: #4b5563;
  margin-bottom: 0.25rem;
}
</style>
