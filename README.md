# A guide and roadmap from beginner to advance for Machine Learning in Production / MLOps 

The question many ask is "How do I start". I am documenting my journey and creating a roadmap/curriculum that can be used as a guide to learn MLOps. Since I learn more by doing, this journey is project-based.

![Machine Learning Workflow](https://ml-ops.org/img/ml-engineering.jpg)
*Machine Learning Engineer Workflow. source - https://ml-ops.org/img/ml-engineering.jpg* 

Building a machine learning model with high accuracy and performance is good but we need to move beyond that. To add more value to our machine learning model, we need to deploy the machine learning or deep learning model to production to solve real-life problems. Automating the process of deploying a model to production is sometimes known as MLOps. MLOps is an emerging and complex field because it requires a lot of components and skills. This repo makes the learning of MLOps easy.

## Lesson 1
### Introduction to Machine Learning Operations(MLOps)

- Meaning and Importances of MLOps
- Difference between MLOps and other Ops like DevOps, DataOps, DevSecOps?
- Building user interface and deploying Machine Model with Gradio and Streamlit 

#### Resources

- [Machine Learning Operations](https://ml-ops.org/)

- [Why I Started ML in Production](https://mlinproduction.com/why-i-started-mlinproduction/)

- [MLOps SIG Roadmap](https://github.com/tdcox/mlops-roadmap/blob/master/MLOpsRoadmap2020.md)

- [Awesome MLOps](https://github.com/visenger/awesome-mlops)


![Example of Image classifier with Streamlit](https://res.cloudinary.com/dbzzslryr/image/upload/v1631955454/mlops/streamlit_classifier.png)

*Image classifier with Streamlit*


### Projects to build
Deploy an image classifier with Gradio and Streamlit - [Link to the deployed image classifier](https://github.com/trojrobert/deploying_image_classification)

Write an article to compare Gradio and Stremlit - *Article in progress* 

#### Resources used in deploying the projects 
**Gradio Resources**
- [Getting Started with Gradio](https://gradio.app/getting_started)

- [Gradio documentation](https://gradio.app/docs)

- [Building NLP Web Apps With Gradio And Hugging Face Transformers](https://towardsdatascience.com/building-nlp-web-apps-with-gradio-and-hugging-face-transformers-59ce8ab4a319)

![Example of Image classifier with Gradio](https://res.cloudinary.com/dbzzslryr/image/upload/v1631955456/mlops/gradio_clasifier.png)
*Image classifier with Gradio*

**Streamlit resources**
- [Examples of projects with Streamlit](https://streamlit.io/gallery)

- [Getting started with Streamlit](https://docs.streamlit.io/en/stable/)
 


#### Extra Reading
[Why data scientists shouldn’t need to know Kubernetes](https://huyenchip.com/2021/09/13/data-science-infrastructure.html)


## Lesson 2 - Deploying machine learning model with python web frameworks

- Introduction to Python Web Frameworks
- Differences between Flask, Django, and FastAPI
- Deploying ML model with Flask 
- From ML Model to Production with FastAPI
- Accessing your model in real-time with Heroku

####  Fast API Resources

[Using FastAPI to Build Python Web APIs](https://realpython.com/fastapi-python-web-apis/)

[Fast API docs](https://fastapi.tiangolo.com/)

#### Flask Resources 

[Introduction to Web development using Flask](https://www.geeksforgeeks.org/python-introduction-to-web-development-using-flask/)

[Flask docs](https://flask.palletsprojects.com/en/2.0.x/)

[Deploying a Python Flask Example Application Using Heroku](https://realpython.com/flask-by-example-part-1-project-setup/)

[FastAPI vs. Django vs. Flask](https://youtu.be/9YBAOYQOzWs)

### Project to build
- Deploy ML model to generate artistic portrait drawing (APDDrawingGAN) with flask + uwsgi

- Build a web app and integrate a machine learning model with FastAPI

#### Projects Resources 
[How to Deploy Machine Learning Models using Flask with Code!](https://www.analyticsvidhya.com/blog/2020/04/how-to-deploy-machine-learning-model-flask/)

[ML Model Deployment With Flask On Heroku](https://youtu.be/pMIwu5FwJ78)

