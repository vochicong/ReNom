<template>
  <div class="loading">
    <div class="animation">
      <div class="bar bar1"></div>
      <div class="bar bar2"></div>
      <div class="bar bar3"></div>
      <div class="bar bar4"></div>
      <div class="bar bar5"></div>
    </div>
  </div>
</template>

<script>
export default {
    name: "LoadingModal"
}
</script>

<style lang="scss">
/* loading animation */
.loading {
  position: absolute;
  top: 0;
  height: 100%;
  width: 100%;
  z-index: 999;
  background-color: #000000;
  opacity: 0.3;
}
.animation {
  position: absolute;
  top: 45%;
  left: 45%;
}
.animation div {
  height:30px;
  width:10px;
  background-color: #ffffff;
  display:inline-block;
  margin-right:10px;
  -webkit-animation: animation-bar 1s infinite;
  animation: animation-bar 1s infinite;
}

.animation .bar1 {
  -webkit-animation-delay: 0.5s;
  animation-delay: 0.5s;
}
.animation .bar2 {
  -webkit-animation-delay: 0.6s;
  animation-delay: 0.6s;
}
.animation .bar3 {
  -webkit-animation-delay: 0.7s;
  animation-delay: 0.7s;
}
.animation .bar4 {
  -webkit-animation-delay: 0.8s;
  animation-delay: 0.8s;
}
.animation .bar5 {
  -webkit-animation-delay: 0.9s;
  animation-delay: 0.9s;
}

@keyframes animation-bar{
  30%
  {
      transform: scaleY(2);
  }
}
</style>