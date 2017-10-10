<template>
  <div id="nodedataviewer">
    <nodedata v-for="(data, index) in node_data" :data="data" :index="index" :key="index"></nodedata>
  </div>
</template>

<script>
import NodeData from './NodeData.vue'

export default {
    name: "TopologyNodeDataViewer",
    components: {
        'nodedata': NodeData
    },
    computed: {
        node_data() {
            return this.$store.state.topology.node_data;
        }
    }
}
</script>

<style lang="scss">
#nodedataviewer {
    height: 80%;
    overflow: scroll;
}
</style>