 <h1 align="center"> 😷 End-to-End-Face-Mask-Detector</h1>


____________________________________________________________________________________________________________________________________________________________

<div align= "center" style="width:90%; margin:30px;font-family:fantasy;"><b>
  A robust solution that detects if the person is wearing mask correctly, wearing mask incorrectly or not wearing at all in static images as well as in real-time video streams. The approach to achieve this purpose is to train a deep learning classification model using Machine Learning packages like TensorFlow, Keras, OpenCV and expose the model over a HTTPS endpoint where the model will be available for inference.</b>
</div>

 

<div align= "center" style="margin:20px;"><img width=300 src="./demo/model-demo.gif"></div>
 
____________________________________________________________________________________________________________________________________________________________



## 📁 Dataset
* Artificially created dataset of people wearing masks correctly and incorrectly. 
* Take faces without masks and use OpenCV and dlib to overlay masks on them.

<img style="display:inline;margin:50px;" width=200 height=350 src="./demo/artificial-dataset.png"/>
     <img  style="display:inline;margin:50px;" width=200 height=350 src="./demo/artificial-dataset1.png"/>

______________________________________________________________________________________________________________________________________________________


## 🏋️‍♂️ Training


* Uploaded local Dataset to Azure blob storage. 
* Created cluster of Azure virtual machines (GPUs) with 4 nodes as training environment . 
* Trained the machine learning model on the remote cluster using a training script written in TensorFlow and Keras. 

<br>
______________________________________________________________________________________________________________________________________________________


## ☁️ Deployment
* Deployed the model as a web service on Azure using Azure Container Instances. 
* Exposed the web service's HTTP endpoint via a Web App built using Flask and Azure App Service. 

<div align= "center" style="margin:30px;"><img src="./demo/webapp-demo.gif"></div>
