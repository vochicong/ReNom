import TopologyState from './state/TopologyState.js'
import TopologyMutation from './mutation/TopologyMutation.js'
import TopologyAction from './action/TopologyAction.js'

const TopologyModule = {
    state: TopologyState,
    mutations: TopologyMutation,
    actions: TopologyAction
}

export default TopologyModule;