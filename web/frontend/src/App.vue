<script setup>
import { ref } from "vue";
import MetricsBoard from "./components/MetricsBoard.vue";

// 批量示例：含转折句式（检验模型对「但是/可惜/却」的理解）和普通正负句
const EXAMPLES = [
  "房间很干净，但是服务态度太差，不会再来了。",
  "位置有点偏，但性价比高，前台也很热情。",
  "早餐丰富，房间宽敞明亮，下次还住这家。",
  "服务很热情，可惜隔音太差，一晚上没睡好。",
  "离地铁站很近，房间却小得转不开身。",
  "酒店就在海边，推开窗就是海景，服务也没得挑。",
  "卫生一塌糊涂，床单上还有头发，慎重入住。",
];

const HISTORY_KEY = "history";
const HISTORY_LIMIT = 50;

const text = ref("");
const loading = ref(false);
const error = ref("");
const result = ref(null);
const history = ref(loadHistory());

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.value));
}

async function analyze() {
  const content = text.value.trim();
  if (!content || loading.value) return;
  loading.value = true;
  error.value = "";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: content }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "请求失败");
    result.value = data;
    history.value.unshift({
      text: content,
      label: data.label,
      confidence: data.confidence,
      time: new Date().toLocaleString(),
    });
    history.value = history.value.slice(0, HISTORY_LIMIT);
    saveHistory();
  } catch (err) {
    error.value = err.message;
  } finally {
    loading.value = false;
  }
}

function clearHistory() {
  history.value = [];
  saveHistory();
}
</script>

<template>
  <div class="container">
    <header>
      <h1>酒店评论情感分析系统</h1>
      <p class="subtitle">输入一条酒店评论，模型判断它是「好评」还是「差评」，并给出置信度</p>
    </header>

    <section class="card">
      <h2>评论分析</h2>
      <textarea
        v-model="text"
        placeholder="例如：房间很干净，但是服务态度太差，不会再来了。"
        @keydown.enter.meta="analyze"
      ></textarea>
      <button class="analyze-btn" :disabled="loading || !text.trim()" @click="analyze">
        {{ loading ? "分析中…" : "分 析" }}
      </button>

      <div class="examples">
        <button v-for="example in EXAMPLES" :key="example" @click="text = example">
          {{ example }}
        </button>
      </div>

      <div v-if="result" class="result-card">
        <div class="result-label" :class="result.label === '好评' ? 'positive' : 'negative'">
          {{ result.label }}
        </div>
        <div class="result-confidence">置信度 {{ (result.confidence * 100).toFixed(1) }}%</div>
      </div>
      <p v-if="error" class="error-msg">{{ error }}</p>
    </section>

    <section class="card">
      <h2>
        历史记录
        <button v-if="history.length" class="clear-btn" @click="clearHistory">清空</button>
      </h2>
      <p v-if="!history.length" class="empty-tip">暂无记录，分析一条评论后这里会保留最近 50 条</p>
      <ul v-else class="history-list">
        <li v-for="(item, index) in history" :key="index">
          <span class="history-text">{{ item.text }}</span>
          <span class="history-tag" :class="item.label === '好评' ? 'positive' : 'negative'">
            {{ item.label }} {{ (item.confidence * 100).toFixed(0) }}%
          </span>
          <span class="history-time">{{ item.time }}</span>
        </li>
      </ul>
    </section>

    <MetricsBoard />
  </div>
</template>
