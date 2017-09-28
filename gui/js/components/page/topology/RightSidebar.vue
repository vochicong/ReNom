<template>
  <div id="rightsidebar" class="col-lg-2">
    <searchbox></searchbox>
    <nodedataviewer></nodedataviewer>
  </div>
</template>

<script>
import SearchBox from './SearchBox.vue'
import NodeDataViewer from './NodeDataViewer.vue'

export default {
    name: "TopologyRightSideBar",
    components: {
        'searchbox': SearchBox,
        'nodedataviewer': NodeDataViewer
    }
}
</script>

<style lang="scss">
#rightsidebar {
    height: 100%;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
}
</style>