<template>
  <div id="page" class="row">
    <leftsidebar></leftsidebar>
    <center></center>
    <rightsidebar></rightsidebar>
  </div>
</template>

<script>
import LeftSideBar from './LeftSideBar.vue'
import Center from './Center.vue'
import RightSideBar from './RightSideBar.vue'

export default {
    name: "TopologyPage",
    components: {
        'leftsidebar': LeftSideBar,
        'center': Center,
        'rightsidebar': RightSideBar
    },
    mounted: function() {
        let header_height = 50;
        let window_height = window.innerHeight;
        this.$el.style.height = window_height - header_height + "px";
    }
}
</script>

<style lang="scss">
</style>