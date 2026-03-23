/**
 * API-клиент для взаимодействия с FastAPI-бэкендом.
 */
import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 30000
})

// Автоматически добавляем JWT-токен в заголовки
http.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// При 401 — перенаправляем на страницу входа
http.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export async function login(username, password) {
  const form = new URLSearchParams()
  form.append('username', username)
  form.append('password', password)
  const { data } = await http.post('/auth/token', form, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
  })
  return data
}

export async function getMe() {
  const { data } = await http.get('/auth/me')
  return data
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export async function getDashboard() {
  const { data } = await http.get('/dashboard')
  return data
}

export async function getDashboardCharts() {
  const { data } = await http.get('/dashboard/charts')
  return data
}

// ---------------------------------------------------------------------------
// Reference
// ---------------------------------------------------------------------------

export async function getReference() {
  const { data } = await http.get('/reference')
  return data
}

// ---------------------------------------------------------------------------
// Orders
// ---------------------------------------------------------------------------

export async function getOrders(params = {}) {
  const { data } = await http.get('/orders', { params: { per_page: 1000, ...params } })
  return data.items ?? data
}

// Paginated version for OrdersView — returns { items, total, page, per_page }
export async function getOrdersPaged(params = {}) {
  const { data } = await http.get('/orders', { params: { per_page: 20, ...params } })
  return data
}

export async function getOrder(id) {
  const { data } = await http.get(`/orders/${id}`)
  return data
}

export async function createOrder(body) {
  const { data } = await http.post('/orders', body)
  return data
}

export async function updateOrderStatus(id, status) {
  const { data } = await http.patch(`/orders/${id}/status`, { status })
  return data
}

export async function updateOrderTracking(id, trackingNumber) {
  const { data } = await http.patch(`/orders/${id}/tracking`, {
    tracking_number: trackingNumber
  })
  return data
}

export async function updateOrder(id, body) {
  const { data } = await http.patch(`/orders/${id}`, body)
  return data
}

// ---------------------------------------------------------------------------
// Order Items
// ---------------------------------------------------------------------------

export async function getOrderItems(orderId) {
  const { data } = await http.get(`/orders/${orderId}/items`)
  return data
}

export async function addOrderItem(orderId, body) {
  const { data } = await http.post(`/orders/${orderId}/items`, body)
  return data
}

export async function updateOrderItem(orderId, itemId, body) {
  const { data } = await http.patch(`/orders/${orderId}/items/${itemId}`, body)
  return data
}

export async function deleteOrderItem(orderId, itemId) {
  const { data } = await http.delete(`/orders/${orderId}/items/${itemId}`)
  return data
}

// ---------------------------------------------------------------------------
// Clients
// ---------------------------------------------------------------------------

export async function getClients() {
  const { data } = await http.get('/clients', { params: { per_page: 1000 } })
  return data.items ?? data
}

// Paginated version for ClientsView — returns { items, total, page, per_page }
export async function getClientsPaged(params = {}) {
  const { data } = await http.get('/clients', { params: { per_page: 25, ...params } })
  return data
}

export async function getClient(id) {
  const { data } = await http.get(`/clients/${id}`)
  return data
}

// ---------------------------------------------------------------------------
// Products
// ---------------------------------------------------------------------------

export async function getProducts(inStockOnly = false) {
  const { data } = await http.get('/products', {
    params: { per_page: 1000, ...(inStockOnly ? { in_stock_only: true } : {}) }
  })
return data.items ?? data
}

export async function createProduct(body) {
  const { data } = await http.post('/products', body)
  return data
}

export async function updateProduct(id, body) {
  const { data } = await http.patch(`/products/${id}`, body)
  return data
}

export async function deleteProduct(id) {
  const { data } = await http.delete(`/products/${id}`)
  return data
}

// ---------------------------------------------------------------------------
// Stock (Склад)
// ---------------------------------------------------------------------------

export async function updateStock(productId, stock) {
  const { data } = await http.patch(`/products/${productId}/stock`, { stock })
  return data
}

// ---------------------------------------------------------------------------
// Export (CSV)
// ---------------------------------------------------------------------------

function _downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export async function exportOrders(params = {}) {
  const response = await http.get('/export/orders', { params, responseType: 'blob' })
  _downloadBlob(response.data, 'orders.csv')
}

export async function exportClients() {
  const response = await http.get('/export/clients', { responseType: 'blob' })
  _downloadBlob(response.data, 'clients.csv')
}

export async function exportProducts() {
  const response = await http.get('/export/products', { responseType: 'blob' })
  _downloadBlob(response.data, 'products.csv')
}

// ---------------------------------------------------------------------------
// Users (Пользователи)
// ---------------------------------------------------------------------------

export async function getUsers() {
  const { data } = await http.get('/users')
  return data
}

export async function createUser(body) {
  const { data } = await http.post('/users', body)
  return data
}

export async function updateUser(id, body) {
  const { data } = await http.patch(`/users/${id}`, body)
  return data
}

export async function deleteUser(id) {
  const { data } = await http.delete(`/users/${id}`)
  return data
}
