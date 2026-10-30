<script setup>
import { ref, onMounted } from "vue";

// 数据全部来自后端 /api/metrics（读模型侧评估产物），前端不算任何数字
const metrics = ref(null);
const error = ref("");

onMounted(async () => {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) throw new Error((await res.json()).detail || "指标加载失败");
    metrics.value = await res.json();
  } catch (err) {
    error.value = err.message;
  }
});

function percent(value) {
  return value == null ? "--" : (value * 100).toFixed(2) + "%";
}
</script>

<template>
  <section class="card">
    <h2>模型成绩单</h2>
    <p class="section-tip">ChnSentiCorp 酒店评论子集测试集（1,553 条，训练全程不接触）｜两个模型使用同一切分与评估口径</p>

    <p v-if="error" class="error-msg">{{ error }}</p>

    <template v-if="metrics">
      <div class="score-row">
        <div v-for="(item, key) in metrics" :key="key" class="score-item">
          <div class="name">{{ item.model }}</div>
          <div class="score" :class="{ win: key === 'bert' }">{{ percent(item.test_accuracy) }}</div>
        </div>
      </div>

      <div class="matrices">
        <img v-for="(item, key) in metrics" :key="key" :src="`/api/matrix/${key}`" :alt="`${key} 混淆矩阵`" />
      </div>
    </template>
  </section>
</template>
