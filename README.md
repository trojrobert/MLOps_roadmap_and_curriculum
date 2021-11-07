# A guide and roadmap from Zero to Hero for Machine Learning in Production / MLOps 

The question many ask is "How do I start". I am documenting my journey and creating a roadmap/curriculum that can be used as a guide to learn MLOps. Since I learn more by doing, this journey is project-based.

![Machine Learning Workflow](https://ml-ops.org/img/ml-engineering.jpg)
*Machine Learning Engineer Workflow. source - https://ml-ops.org/img/ml-engineering.jpg* 

Building a machine learning model with high accuracy and performance is good but we need to move beyond that. To add more value to our machine learning model, we need to deploy the machine learning or deep learning model to production to solve real-life problems. Automating the process of deploying a model to production is sometimes known as MLOps. MLOps is an emerging and complex field because it requires a lot of components and skills. This repo makes the learning of MLOps easy.

## Lesson 1
### Introduction to Machine Learning Operations(MLOps)

- Meaning and Importances of MLOps
- Difference between MLOps and other Ops like DevOps, DataOps, DevSecOps?
- Building user interface and deploying Machine Model with Gradio and Streamlit 

### Keywords
MLOps, Gradio, Streamlit 

#### Resources

- [Machine Learning Operations](https://ml-ops.org/)

- [Why I Started ML in Production](https://mlinproduction.com/why-i-started-mlinproduction/)

- [MLOps SIG Roadmap](https://github.com/tdcox/mlops-roadmap/blob/master/MLOpsRoadmap2020.md)

- [Awesome MLOps](https://github.com/visenger/awesome-mlops)

- [Differences: DevOps, ITOps, MLOps, DataOps, ModelOps, AIOps, SecOps, DevSecOps](https://medium.com/vitrox-publication/differences-devops-itops-mlops-dataops-modelops-aiops-secops-devsecops-part-1-3-8b238cf72942)


![Example of Image classifier with Streamlit](https://res.cloudinary.com/dbzzslryr/image/upload/v1631955454/mlops/streamlit_classifier.png)

*Image classifier with Streamlit*


### Projects to build
Deploy an image classifier with Gradio and Streamlit - [Link to the deployed image classifier](https://github.com/trojrobert/deploying_image_classification)

Write an article to compare Gradio and Streamlit - [A guide for creating UI and deploying machine learning model with Streamlit vs Gradio](https://trojrobert.medium.com/a-guide-for-deploying-machine-learning-model-with-streamlit-vs-gradio-563a0b2dc1bd)

#### Project Resources 
**Gradio Resources**
- [Getting Started with Gradio](https://gradio.app/getting_started)

- [Gradio documentation](https://gradio.app/docs)

- [Building NLP Web Apps With Gradio And Hugging Face Transformers](https://towardsdatascience.com/building-nlp-web-apps-with-gradio-and-hugging-face-transformers-59ce8ab4a319)

![Example of Image classifier with Gradio](https://res.cloudinary.com/dbzzslryr/image/upload/v1631955456/mlops/gradio_clasifier.png)
*Image classifier with Gradio*

**Streamlit resources**
- [Examples of projects with Streamlit](https://streamlit.io/gallery)

- [Getting started with Streamlit](https://docs.streamlit.io/en/stable/)
 
### Other resources
[How to Implement Customer Churn Prediction [Machine Learning Guide for Programmers]](https://neptune.ai/blog/how-to-implement-customer-churn-prediction)

[Why data scientists shouldn’t need to know Kubernetes](https://huyenchip.com/2021/09/13/data-science-infrastructure.html)


## Lesson 2 - Deploying machine learning model with python web frameworks

- Introduction to Python Web Frameworks
- Differences between Flask, Django, and FastAPI
- Introduction to Docker
- Deploying computer vision model with Flask and uWSGI on Digital Ocean 

### Keywords
Flask, uWGSGI, Django, FastAPI,Docker, Digital Ocean  
#### Docker Resources
[Docker Tutorial for Beginners - A Full DevOps Course on How to Run Applications in Containers](https://youtu.be/fqMOX6JJhGo)

[Docker Tutorial for Beginners [FULL COURSE in 3 Hours]](https://youtu.be/3c-iBn73dDE)

####  Fast API Resources

[Using FastAPI to Build Python Web APIs](https://realpython.com/fastapi-python-web-apis/)

[Fast API docs](https://fastapi.tiangolo.com/)

#### Flask Resources 

[Introduction to Web development using Flask](https://www.geeksforgeeks.org/python-introduction-to-web-development-using-flask/)

[Flask docs](https://flask.palletsprojects.com/en/2.0.x/)

[Deploying a Python Flask Example Application Using Heroku](https://realpython.com/flask-by-example-part-1-project-setup/)

[FastAPI vs. Django vs. Flask](https://youtu.be/9YBAOYQOzWs)



### Project to build
- Deploy ML model to generate artistic portrait drawing (APDDrawingGAN) with Flask + uWSGI on Digital Ocean

   - Front-end - Bootstrap

   - Web Framework - Flask + uWSGI

   - ML model - Pytorch + Fastai(APDrawingGAN)

   - Deployment - Docker

   - Cloud service - Digital ocean

- Build a web app with FastAPI, integrate a machine learning model for time series and deploy on Heroku (Optional)

#### Projects Resources 
[How to Deploy Machine Learning Models using Flask with Code!](https://www.analyticsvidhya.com/blog/2020/04/how-to-deploy-machine-learning-model-flask/)

[ML Model Deployment With Flask On Heroku](https://youtu.be/pMIwu5FwJ78)

## Others 
https://colab.research.google.com/github/fastai/fastbook/blob/master/02_production.ipynb#scrollTo=RuQMz_jqihsU


## Lesson 3- Machine Learning Model tracking with MLflow and deployment optimization

- Running multiple containers with docker compose
- Introduction to Github Action
- Introduction to mlflow

### Project 
- Build a forecasting model with Fast API and track the model with mlflow
