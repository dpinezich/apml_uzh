<template>
  <div class="slidev-layout end apml-end">
    <slot />
  </div>
</template>

<style scoped>
/*
 * End slide layout — matches GVZ template exactly:
 *   Top 40%  → white  (large teal heading sits here)
 *   Mid 57%  → teal   (subtitle + black bar sit here)
 *   Last 3%  → black  (thin footer stripe)
 */
.apml-end {
  position: relative;
  height: 100%;
  width: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: linear-gradient(
    to bottom,
    #ffffff  0%,
    #ffffff 40%,
    #00CCCC 40%,
    #00CCCC 97%,
    #000000 97%,
    #000000 100%
  );
}
</style>
