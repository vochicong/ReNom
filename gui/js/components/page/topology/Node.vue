<template>
  <circle :id="'circle_'+index" :r="size*10" :fill="color" :cx="cx" :cy="cy" stroke="rgba(153,153,153,0)" stroke-width=3 @mousedown="clicknode"></circle>
</template>

<script>
export default {
    name: "Node",
    props: ["node", "index"],
    data: function() {
        return {
            cx: 0,
            cy: 0,

        }
    },
    computed: {
        width() {
            let e = document.getElementById("center");
            return e.clientWidth;
        },
        height() {
            let e = document.getElementById("center");
            return e.clientHeight;
        },
        size(){
            return this.$store.state.topology.node_sizes[this.index]
        },
        color() {
            return this.$store.state.topology.colors[this.index]
        }
    },
    mounted: function() {
        this.cx = this.node[0]*this.width,
        this.cy = this.node[1]*this.height
    },
    methods: {
        clicknode: function() {
            this.$store.dispatch('click_node', {
                'click_node_index': this.index
            });
        }
    }
}
</script>

<style lang="scss">
</style>