<template>
  <div id="center" class="col-lg-8">
    <svg id="topology_canvas">
      <g v-if="edges" class="links">
        <edge v-if="edges" v-for="(edge, index) in edges" :edge="edge" :key="index"></edge>
      </g>

      <g v-if="nodes" class="nodes">
        <node v-if="nodes" v-for="(node, index) in nodes" :node="node" :index="index" :key="index"></node>
      </g>
    </svg>
  </div>
</template>

<script>
import Node from './Node.vue'
import Edge from './Edge.vue'

export default {
    name: "TopologyCenter",
    components: {
        'node': Node,
        'edge': Edge
    },
    computed: {
        nodes(){
            return this.$store.state.topology.nodes
        },
        edges() {
            return this.$store.state.topology.edges
        }
    }
}
</script>

<style lang="scss">
#center {
    margin: 0;
    padding: 0;
}
#topology_canvas {
    width: 100%;
    height: 100%;
    .nodes {
        width: 100%;
        height: 100%;
    }
}
</style>