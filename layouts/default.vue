<template>
  <div class="apml-default">
    <div class="teal-bar" />
    <div class="content">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.apml-default {
  position: relative;
  height: 100%;
  width: 100%;
  background: #ffffff;
  display: flex;
  flex-direction: row;
  overflow: hidden;
}

.teal-bar {
  flex-shrink: 0;
  width: 10px;
  height: 100%;
  background: #00CCCC;
}

.content {
  flex: 1;
  min-width: 0;
  padding: 38px 56px 28px 68px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
</style>
