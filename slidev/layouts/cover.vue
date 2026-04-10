<template>
  <div class="slidev-layout cover apml-cover">
    <div class="cover-content">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.apml-cover {
  height: 100%;
  width: 100%;
  background: #ffffff;
  display: flex !important;
  justify-content: flex-end !important;
  align-items: flex-end !important;
  overflow: hidden;
}

.cover-content {
  text-align: right;
  padding: 48px 60px;
  max-width: 72%;
}
</style>
