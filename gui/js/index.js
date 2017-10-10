import Vue from 'vue'
import store from './store/store.js'
import router from './router.js'
import App from './components/App.vue'

new Vue({
    el: '#app',
    router: router,
    store: store,
    render: h => h(App)
})
