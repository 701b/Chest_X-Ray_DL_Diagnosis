const multer = require('multer')
const spawn = require('child_process').spawn
const fs = require('fs')
const app_1 = require('./app')
const app = app_1.app

const port = 3000
let imageNum = 1

if (!fs.existsSync('public/uploads')) {
    fs.mkdirSync('public/uploads')
}

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'public/uploads/')
    },
    filename: function (req, file, cb) {
        cb(null, imageNum + '.png')
        imageNum++
    }
})
const upload = multer({ storage: storage })

app.get('/', (req, res) => {

})

app.post('/uploads', upload.single('image'), (req, res) => {
    app.locals.fileNum = imageNum - 1
    res.send((imageNum - 1).toString())
})

app.get('/predict', (req, res) => {
    const net = spawn('python', ['model/predict.py', app.locals.fileNum])
    
    net.stdout.on('data', function (data) {
        console.log(data.toString())
        
        res.send(data.toString())
    })
    
    net.stderr.on('data', function (data) {
        console.log(data.toString())
    })
})

app.listen(port, () => {
    console.log(`server is listening at localhost:${ port }`)
})