<template>
  <div class="container">
    <div class="upload-wrapper" v-if="isUploading">
      <ImageUploader @uploadImage="uploadImage" v-bind:inputImageNum="inputImageNum"/>
      <div class="predict-box" v-if="inputImageNum !== ''">
        <button class="predict-btn" @click="predict">진단</button>
      </div>
    </div>
    <ImageDiagnosisResult v-if="!isUploading"
                          v-bind:diseaseArr="diseaseArr"
                          v-bind:isDiagnosing="isDiagnosing"
                          v-bind:inputImageNum="inputImageNum"/>
  </div>
</template>

<script>
import ImageUploader from './components/ImageUploader'
import ImageDiagnosisResult from './components/ImageDiagnosisResult'
import axios from 'axios'

export default {
  name: 'App',
  components: {
    ImageUploader,
    ImageDiagnosisResult
  },
  data: () => ({
    inputImageNum: '',
    diseaseArr: [],
    isUploading: true,
    isDiagnosing: false
  }),
  methods: {
    uploadImage: function (image) {
      const form = new FormData()

      this.inputImageName = image.name

      form.append('image', image)

      axios.post('http://localhost:3000/uploads', form, {
        header: { 'Content-Type': 'multipart/form-data' }
      }).then(({ data }) => {
        this.inputImageNum = data
      }).catch(err => console.log(err))
    },
    predict: function () {
      this.isUploading = false
      this.isDiagnosing = true

      axios.get('http://localhost:3000/predict').then(({ data }) => {
        const LABEL_LIST = ['Atelectasis', 'Effusion', 'Pneumonia', 'Pneumothorax', 'Mass_or_Nodule']
        const dataArr = data.split('\r\n')
        const resultRegex = /\[\d{1}, \d{1}, \d{1}, \d{1}, \d{1}]/

        for (let i = 0; i < dataArr.length; i++) {
          if (resultRegex.test(dataArr[i])) {
            const resultArr = JSON.parse(dataArr[i])

            for (let j = 0; j < resultArr.length; j++) {
              if (resultArr[j] === 1) {
                this.diseaseArr.push(LABEL_LIST[j])
              }
            }

            this.isDiagnosing = false
          }
        }
      }).catch(err => console.log(err))
    }
  }
}
</script>

<style>
div {
  box-sizing: border-box;
}

ul {
  padding: 0;
  margin: 0;
}

li {
  list-style: none;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2C3E50;
  margin-top: 60px;
}

.container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.predict-box {
  margin-top: 20px;
}

.predict-btn {
  padding: 5px  15px;
  background-color: coral;
  color: white;
  border: 0;
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
}
</style>
