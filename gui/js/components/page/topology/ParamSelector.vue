<template>
  <div id="paramselector" class="col-lg-11" v-if="labels">
    <div class="input-group vertical">
      <label for="filename">resolution: {{resolution}}</label>
      <input type="range" min="5" max="50" step="5" class="slider" v-model="resolution">
    </div>

    <div class="input-group vertical">
      <label for="filename">overlap: {{overlap}}</label>
      <input type="range" min="0.1" max="1" step="0.1" class="slider" v-model="overlap">
    </div>

    <div class="input-group vertical">
      <label for="filename">colored by:</label>
      <select v-model="color_index">
        <option v-for="(item, index) in labels" :value="index">{{item}}</option>
      </select>
    </div>

    <button class="run_button" v-on:click="run"><span class="run_icon"><i class="fa fa-play" aria-hidden="true"></i></span>RUN</button>
  </div>
</template>

<script>
export default {
    name: "TopologyParamSelector",
    data: function() {
        return {
            resolution: 25,
            overlap: 0.5,
            color_index: 0
        }
    },
    computed: {
        labels(){
            return this.$store.state.topology.labels
        }
    },
    methods: {
        run: function() {
            if (this.check_algorithm_range) {
                return
            }
            this.$store.dispatch('create_topology', {
                'resolution': this.resolution,
                'overlap': this.overlap,
                'color_index': this.color_index
            });
        }
    }
}
</script>

<style lang="scss">
#paramselector {
  margin: 0 auto;
  .input-group {
    margin-bottom: 12px;
  }
  .run_button {
    width: 100%;
    margin: 0;
    padding: 8px;
    .run_icon {
      margin-right: 8px;
    }
  }
}
</style>