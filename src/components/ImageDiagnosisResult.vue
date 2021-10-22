<template>
  <div class="image-diagnosis-box">
    <div class="waiting" v-if="isDiagnosing">
      <span>딥러닝 알고리즘이 진단 중입니다.</span>
    </div>
    <div class="result-box" v-if="!isDiagnosing">
      <div class="result-title">
        <span>진단 결과</span>
      </div>
      <span v-if="diseaseArr.length === 0">진단 없음</span>
      <ul>
        <li class="result-item" v-for="disease in diseaseArr" v-bind:key="disease">
          <span>{{ disease }}</span>
          <img v-bind:src="'http://localhost:3000/result/' + inputImageNum + '/' + disease + '.png'" alt="" width="250">
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ImageDiagnosisResult',
  components: {

  },
  props: {
    diseaseArr: Array,
    isDiagnosing: Boolean,
    inputImageNum: String
  },
  data: () => ({

  }),
  methods: {

  }
}
</script>

<style scoped>
ul {
  display: flex;
  justify-content: center;
}

.result-box {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.result-title {

}

.result-item {
  display: flex;
  flex-direction: column;
  padding: 20px;
}
</style>