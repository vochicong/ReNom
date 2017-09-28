<template>
  <div id="leftsidebar" class="col-lg-2">
    <fileloader></fileloader>
    <paramselector></paramselector>
  </div>
</template>

<script>
import FileLoader from './FileLoader.vue'
import ParamSelector from './ParamSelector.vue'

export default {
    name: "TopologyLeftSideBar",
    components: {
        'fileloader': FileLoader,
        'paramselector': ParamSelector
    }
}
</script>

<style lang="scss">
#leftsidebar {
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
}
</style>