import os
from mask_detector import detect_masks
from flask import Flask, request, render_template, url_for

# creates a Flask instance
app = Flask(__name__)

#connection between the URL / and a function that returns a response
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/detect', methods= ['POST', 'GET'])
def detect():
    if request.method == 'POST':
        image = request.files['image']
        
        if not os.path.exists(os.path.join('static', 'images')):
            os.mkdir(os.path.join('static', 'images'))
        
        image_path = os.path.join('static', 'images', image.filename)
        image.save(image_path)
        
        response = detect_masks()        
        return render_template('detection.html',predictions=response, image=url_for('static', filename=os.path.join('images', image.filename)))
    

if __name__ == "__main__":
    app.run(debug=True)
