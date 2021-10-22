<template>
  <div class="image-uploader-box">
    <div class="title">
      <span>흉부 X-ray 이미지 첨부</span>
    </div>
    <div class="upload-box">
      <div class="image-name-box">
        <span>{{ inputImageName }}</span>
      </div>
      <label class="input-file-btn" for="image">
      </label>
      <input id="image" ref="image" name="image" type="file" @change="uploadImage()">
    </div>
    <div class="img-box" v-if="inputImageNum !== ''">
      <img v-bind:src="'http://localhost:3000/uploads/' + inputImageNum + '.png'" alt="" width="458">
    </div>
  </div>
</template>

<script>
export default {
  name: 'ImageUploader',
  props: {
    inputImageNum: String
  },
  data: () => ({
    inputImageName: ''
  }),
  methods: {
    uploadImage: function () {
      const image = this.$refs['image'].files[0]

      this.inputImageName = image.name

      this.$emit('uploadImage', image)
    },
  }
}
</script>

<style scoped>
.image-uploader-box {
  display: flex;
  flex-direction: column;
}

.title {
  display: flex;
  justify-content: left;
  padding: 10px;
  margin-bottom: 10px
}

.upload-box {
  display: flex;
  justify-content: space-between;
  width: 500px;
  padding: 10px 25px;
  border: 1px solid rgba(0, 0, 0, 0.25);
  border-radius: 3px;
  background-color: #EEEEEE;
}

.image-name-box {
  display: flex;
  align-items: center;
}

.input-file-btn {
  width: 25px;
  height: 25px;
  background-image: url("/icon/attachment-icon.png");
  background-position: center;
  background-size: cover;
  border-radius: 4px;
  color: white;
  cursor: pointer;
}

input[type="file"] {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}

.img-box {
  width: 500px;
  background-color: black;
  padding: 20px;
  border: 1px solid rgba(0, 0, 0, 0.25);
  border-radius: 5px;
  margin-top: 20px;
}
</style>