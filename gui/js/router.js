import Vue from 'vue'
import Router from 'vue-router'
import TopologyPage from './components/page/topology/Page.vue'

Vue.use(Router)

const router = new Router({
    routes: [
        { path: '/', name: 'TopologyPage', component: TopologyPage },
    ]
})

export default router