const express = require('express')
const cors = require('cors')

class App {
    
    constructor() {
        this.app = express()
        
        this.setViewEngine()
        this.setMiddleWare()
        this.setStatic()
        this.setLocals()
        this.setErrorHandler()
    }
    
    setMiddleWare() {
        this.app.use(cors())
    }
    
    setViewEngine() {
    
    }
    
    setStatic() {
        this.app.use('/css', express.static('css'))
        this.app.use('/js', express.static('js'))
        this.app.use('/image', express.static('image'))
        this.app.use('/uploads', express.static('public/uploads'))
        this.app.use('/result', express.static('model/result'))
    }
    
    setLocals() {
        this.app.use((req, res, next) => {
            this.app.locals.fileNum
            next()
        })
    }
    
    setErrorHandler() {
        this.app.use((err, req, res, _) => {
        })
    }
    
}

exports.App = App;
exports.app = new App().app