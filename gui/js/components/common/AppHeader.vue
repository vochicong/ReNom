<template>
  <header>
    <div class="header_title">ReNom TDA</div>
  </header>
</template>

<script>
export default {
    name: "AppHeader"
}
</script>

<style lang="scss">
header {
  height: 50px;
  margin: 0;
  padding: 0;
  background: -moz-linear-gradient(#223347, #1c4254);
  background: -webkit-gradient(linear, left top, left bottom, from(#223347), to(#1c4254));
  font-family:"Arial Black";
  .header_title {
    padding-left: 20px;
    height: 100%;
    color: #eeeeee;
    font-size: 18px;
    line-height: 50px;
  }
}
</style>