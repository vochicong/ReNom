<template>
  <div id="fileloader" class="col-lg-11">
    <div class="input-group vertical">
      <label for="filename">File Name</label>
      <input type="text" id="filename" placeholder="input file name" v-model="filename">
    </div>

    <div class="input-group vertical">
      <label for="filename">Algorithm</label>
      <select v-model="algorithm">
        <option v-for="(a, index) in algorithms" :value="index">{{a}}</option>
      </select>
    </div>

    <button class="load_file_button" v-on:click="load_file">Load file</button>
  </div>
</template>

<script>
export default {
    name: "TopologyFileLoader",
    data: function() {
        return {
            filename: "",
            algorithm: 0
        }
    },
    computed: {
        algorithms(){
            return this.$store.state.topology.algorithms
        }
    },
    methods: {
        load_file: function() {
            this.$store.dispatch('load_file', {
                'filename': this.filename,
                'algorithm_index': this.algorithm
            });
        }
    }
}
</script>

<style lang="scss">
#fileloader {
  margin: 0 auto;
  margin-bottom: 20px;
  .input-group {
    margin-bottom: 12px;
  }
  .load_file_button {
    width: 100%;
    margin: 0;
    padding: 8px;
  }
}
</style>