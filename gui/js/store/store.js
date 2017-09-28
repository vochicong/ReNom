import Vue from 'vue'
import Vuex from 'vuex'
import TopologyModule from './modules/TopologyModule.js'

Vue.use(Vuex)

const store = new Vuex.Store({
    modules: {
        topology: TopologyModule,
    }
})

export default store
