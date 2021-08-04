# Arxiv Neural Search
![Arxiv Neural Search Demo](https://github.com/karndeb/Arxiv-Neural-Search/blob/master/demo/animation_demo.gif)

This project is a sample demo of Arxiv search related to AI/ML Papers built using Streamlit, sentence-transformers and Faiss. 


## Some of the features of the project

- The search has more than 100,000 AI/ML related papers indexed 
- The index covers all the five important categories of arxiv AI/ML research which are 
  - Computation and Language 
  - Information Retrieval 
  - Machine Learning 
  - Human-Computer Interaction
  - Computer Vision and Pattern Recognition
- The indexing has been done using Faiss index so its super fast
- Distillbert model is powering the generation of query and abstract embeddings 
- The Frontend has been built using streamlit which can be used to define the number of search results, filter out the results at the category level
- The Frontend provides the direct pdf links to the papers incase you want to download the papers 

## Requirements
- Create two empty directories called data and models
- Run the colab notebook in the notebooks folder to generate the data and the serialized index and save it in the two respective folders

## Installation & Usage

```bash
$ git clone https://github.com/karndeb/Arxiv-Neural-Search.git
# install packages
$ pip install -r requirements.txt
# start the streamlit app server
$ streamlit run app.py
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE.md](https://github.com/karndeb/Arxiv-Neural-Search/blob/master/LICENSE.md) file for details
