import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from vector_engine.utils import vector_search


@st.cache
def read_data(data="data/Arxiv_AIML_processed.csv"):
    """Read the data from local."""
    return pd.read_csv(data)


@st.cache(allow_output_mutation=True)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="models/faiss_index_aiml.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)


def main():
    # Load data and models
    data = read_data()
    model = load_bert_model()
    faiss_index = load_faiss_index()

    st.title("Arxiv AI/ML Vector search with Sentence Transformers and Faiss")

    # User search
    user_input = st.text_area("Search box")

    # Filters
    st.markdown("**Filters**")
    num_results = st.slider("Number of search results", 10, 50, 10)
    arxiv_map = {"Computation and Language": "cs.CL", "Information Retrieval": "cs.IR", "Machine Learning": "cs.LG",
                 "Human-Computer Interaction": "cs.HC", "Computer Vision and Pattern Recognition": "cs.CV"}
    # Create a list of possible values and multiselect menu with them in it.
    cat = ["Computation and Language", "Information Retrieval", "Machine Learning", "Human-Computer Interaction",
           "Computer Vision and Pattern Recognition"]
    cat_selected = st.multiselect('Select the AI/ML Categories', cat)
    lst = [arxiv_map[x] for x in cat_selected]
    arxiv_cat = data[data.categories.str.match('|'.join(lst))]
    arxiv_cs = arxiv_cat.loc[arxiv_cat["arxiv_id"].str.startswith('cs', na=False)]
    keys = list(arxiv_cs.columns.values)
    i1 = arxiv_cat.set_index(keys).index
    i2 = arxiv_cs.set_index(keys).index
    arxiv_notcs = arxiv_cat[~i1.isin(i2)]
    assert len(arxiv_cat) == len(arxiv_cs) + len(arxiv_notcs)
    arxiv_notcs['arxiv_id'] = arxiv_notcs['arxiv_id'].astype(float).round(decimals=5)
    frames = [arxiv_cs, arxiv_notcs]
    rounded_frame = pd.concat(frames)
    # Create the pdf links
    link_lst = []
    for i in rounded_frame['arxiv_id']:
        link_lst.append(f"https://arxiv.org/pdf/{i}.pdf")
    rounded_frame['pdf_links'] = link_lst

    # Fetch results
    if cat_selected:
        # Get paper IDs
        D, I = vector_search([user_input], model, faiss_index, num_results)
        # Get individual results
        for id_ in I.flatten().tolist():
            if id_ in set(rounded_frame.id):
                f = rounded_frame[(rounded_frame.id == id_)]
            else:
                continue

            st.write(
                f"""**{f.iloc[0].title}**  
            **Abstract**
            {f.iloc[0].abstract}
            """
            )
            st.write(f"**Link** [link to PDF]({f.iloc[0].pdf_links})")


if __name__ == "__main__":
    main()

