<template>
  <line stroke-width="1" :stroke="color" :x1="x1" :y1="y1" :x2="x2" :y2="y2"></line>
</template>

<script>
export default {
    name: "Edge",
    props: ["edge"],
    data: function() {
        return {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0

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
        n1() {
            return this.$store.state.topology.nodes[this.edge[0]]
        },
        n2() {
            return this.$store.state.topology.nodes[this.edge[1]]
        },
        color() {
            return this.$store.state.topology.colors[this.edge[0]]
        }
    },
    mounted: function() {
        this.x1 = this.n1[0]*this.width,
        this.y1 = this.n1[1]*this.height,
        this.x2 = this.n2[0]*this.width,
        this.y2 = this.n2[1]*this.height
    }
}
</script>

<style lang="scss">
</style>