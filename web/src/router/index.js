import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth.js'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('../views/LoginView.vue'),
    meta: { public: true }
  },
  {
    path: '/',
    component: () => import('../components/AppLayout.vue'),
    meta: { requiresAuth: true },
    children: [
      {
        path: '',
        redirect: '/dashboard'
      },
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('../views/DashboardView.vue'),
        meta: { roles: ['admin'] }
      },
      {
        path: 'orders/new',
        name: 'NewOrder',
        component: () => import('../views/NewOrderView.vue'),
        meta: { roles: ['admin'] }
      },
      {
        path: 'orders/:id',
        name: 'OrderDetail',
        component: () => import('../views/OrderDetailView.vue'),
        meta: { roles: ['admin'] }
      },
      {
        path: 'orders',
        name: 'Orders',
        component: () => import('../views/OrdersView.vue'),
        meta: { roles: ['admin'] }
      },
      {
        path: 'clients/:id',
        name: 'ClientDetail',
        component: () => import('../views/ClientDetailView.vue'),
        meta: { roles: ['admin'] }
      },
      {
        path: 'clients',
        name: 'Clients',
        component: () => import('../views/ClientsView.vue'),
        meta: { roles: ['admin'] }
      },
      {
        path: 'products',
        name: 'Products',
        component: () => import('../views/ProductsView.vue'),
        meta: { roles: ['admin'] }
      },
      {
        path: 'journal',
        name: 'Journal',
        component: () => import('../views/MonthlyOrdersView.vue'),
        meta: { roles: ['admin'] }
      },
      {
        path: 'batches',
        name: 'Batches',
        component: () => import('../views/BatchesView.vue'),
        meta: { roles: ['admin'] }
      },
      {
        path: 'packing',
        name: 'Packing',
        component: () => import('../views/PackingView.vue'),
        meta: { roles: ['admin', 'warehouse'] }
      },
      {
        path: 'stock',
        name: 'Stock',
        component: () => import('../views/StockView.vue'),
        meta: { roles: ['admin', 'warehouse'] }
      },
      {
        path: 'users',
        name: 'Users',
        component: () => import('../views/UsersView.vue'),
        meta: { roles: ['admin'] }
      },
      {
        path: 'architecture',
        name: 'Architecture',
        component: () => import('../views/ArchitectureView.vue'),
        meta: { roles: ['admin'] }
      }
    ]
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/'
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// Защита роутов — редирект на /login если не авторизован,
// проверка ролей для доступа к страницам
router.beforeEach((to) => {
  const auth = useAuthStore()

  if (to.meta.requiresAuth && !auth.isAuthenticated) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }

  // Проверка ролей
  const allowedRoles = to.meta.roles
  if (allowedRoles && auth.role && !allowedRoles.includes(auth.role)) {
    // Перенаправить на домашнюю страницу роли
    const home = auth.isWarehouse ? '/packing' : '/dashboard'
    if (to.path !== home) return { path: home }
  }

  // Редирект авторизованного с /login
  if (to.path === '/login' && auth.isAuthenticated) {
    return { path: auth.isWarehouse ? '/packing' : '/dashboard' }
  }

  // Редирект / на домашнюю страницу роли
  if (to.path === '/' && auth.isAuthenticated && auth.isWarehouse) {
    return { path: '/packing' }
  }
})

export default router
