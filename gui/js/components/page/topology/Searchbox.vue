<template>
  <div id="searchbox" class="col-lg-11">
    <div class="input-group vertical">
      <input id="searchtxtbox" placeholder="search" v-model="search_value" @keyup.enter="search" @keyup.delete="deletetext">
    </div>
  </div>
</template>

<script>
export default {
    name: "TopologySearchBox",
    data: function() {
        return {
            search_value: ""
        }
    },
    methods: {
        search: function() {
            // 検索文字をbase64に変換
            let text_encoder = new TextEncoder();
            const search_txt = text_encoder.encode(this.search_value);
            this.$store.dispatch('search_topology', {
                'search_txt': search_txt
            });
        },
        deletetext: function() {
            if (this.search_value == "") {
                this.search();
            }
        }
    }
}
</script>

<style lang="scss">
#searchbox {
    margin: 0 auto;
    margin-bottom: 20px;
}
</style>