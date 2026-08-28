<template>
  <div class="app">
    <nav class="tabs">
      <button v-for="t in tabs" :key="t" :class="{ active: tab===t }" @click="tab=t">{{ t }}</button>
    </nav>

    <section v-if="tab==='Merge'" class="panel">
      <h2>Merge</h2>
      <p>Methods: linear, slerp, ties, dare, della, passthrough, darwin, frankenmerge</p>
      <input v-model="mergeOutput" placeholder="output dir" />
      <button @click="doMerge">Merge</button>
      <pre>{{ log }}</pre>
    </section>

    <section v-if="tab==='Quantize'" class="panel">
      <h2>Quantize</h2>
      <p>jang, dynamic3, apex, btl4, mixed — GGUF + JangQ MLX</p>
    </section>

    <section v-if="tab==='Eval'" class="panel">
      <h2>Eval</h2>
      <p>Benchmarks: hella, mmlu, arc, gsm8k, gpqa — Evals: ace, swe, terminal, gaia, hle</p>
    </section>

    <section v-if="tab==='Models'" class="panel">
      <h2>Models</h2>
      <p>Search & download via HuggingFace Hub</p>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { invoke } from '@tauri-apps/api/core'
const tabs = ['Merge','Quantize','Eval','Models']
const tab = ref('Merge')
const mergeOutput = ref('./merged')
const log = ref('')
async function doMerge() {
  try { log.value = await invoke('merge_models', { req: { models: [], method: 'linear', output: mergeOutput.value } }) }
  catch(e){ log.value = String(e) }
}
</script>

<style>
.app { font-family: system-ui; max-width: 960px; margin: 2rem auto; }
.tabs { display: flex; gap: .5rem; margin-bottom: 1rem; }
.tabs button.active { background: #222; color: #fff; }
.panel { border: 1px solid #ddd; padding: 1rem; border-radius: 8px; }
</style>
