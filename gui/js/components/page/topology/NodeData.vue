<template>
  <div id="nodedata">
    <p>{{ categorical_data[show_category] }}</p>
    <p>{{ labels[color_index] }}: {{ data[color_index] }}</p>
  </div>
</template>

<script>
export default {
    name: "TopologyNodeData",
    props: ["data", "index"],
    computed: {
        categorical_data() {
            return this.$store.state.topology.node_categorical_data[this.index]
        },
        show_category() {
            return this.$store.state.topology.show_category
        },
        color_index() {
            return this.$store.state.topology.color_index
        },
        labels() {
            return this.$store.state.topology.labels
        }
    }
}
</script>

<style lang="scss">
#nodedata {
    margin: 10px 10%;
    border-bottom: 1px solid #ccc;
}
</style>