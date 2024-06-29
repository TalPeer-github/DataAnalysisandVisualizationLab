import textwrap
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, Dataset
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import cohere
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def initialize_client_connection():
    with open("pinecone_api_key.txt") as f:
        PINECONE_API_KEY = f.read().strip()
    with open("cohere_api_key.txt") as f:
        COHERE_API_KEY = f.read().strip()
    return PINECONE_API_KEY, COHERE_API_KEY


def embedding_model():
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def format_data(df: pd.DataFrame):
    def format_row(row):
        formatted_text = ""
        for column_name, value in row.items():
            formatted_text += f"{column_name}:{value}\n"
        return formatted_text.strip()

    formatted_texts = df.apply(format_row, axis=1)
    formatted_df = pd.DataFrame({'text': formatted_texts})
    return formatted_df


def load_and_embedd_dataset(
        dataset_name: str,
        split: str = 'train',
        model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2'),
        text_field: str = 'text',
        rec_num: int = 400) -> tuple:
    """
    Load a dataset and embedd the text field using a sentence-transformer model
    :param dataset_name: The name of the dataset to load
    :param split: The split of the dataset to load
    :param model: The model to use for embedding
    :param text_field: The field in the dataset that contains the text
    :param rec_num: The number of records to load and embedd
    :return: tuple: A tuple containing the dataset and the embeddings
    """
    print(f"Loading and embedding dataset: {dataset_name}")
    df = pd.read_csv('incident_ds.csv')
    format_df = format_data(df)
    dataset = Dataset.from_pandas(format_df, split=split)
    embeddings = model.encode(dataset[text_field][:rec_num])
    print("Done!")
    return dataset, embeddings


def create_pinecone_index(
        pc_api_key: str,
        index_name: str,
        dimension: int,
        metric: str = 'cosine', ):
    """
    In Pinecone, an index is the highest-level organizational unit of data, where you define the dimension of vectors
    to be stored and the similarity metric to be used when querying them.
        - It is crucial that the metric you will use in your VectorDB will also be a metric your embedding model
    :param pc_api_key: Pinecone API Key
    :param index_name: The name of the index
    :param dimension: The dimension of the index
    :param metric: The metric to use for the index
    :return:
        Pinecone: A pinecone object which can later be used for upserting vectors and connecting to VectorDBs
    """

    print("Creating a Pinecone index...")
    pc = Pinecone(api_key=pc_api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(name=index_name, dimension=dimension, metric=metric,
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
    print("Done!")
    return pc


def upsert_vectors(index: Pinecone,
                   embeddings: np.ndarray,
                   dataset: dict,
                   text_field: str = 'text',
                   batch_size: int = 24):
    """
    Upsert vectors to a pinecone index.
    Within an index, vectors are stored in namespaces, and all update & insert queries, and other data operations
    always target one namespace.
        :param index: The pinecone index object
        :param embeddings: The embeddings to upsert
        :param dataset: The dataset containing the metadata
        :param text_field: The field in the dataset that contains the text
        :param batch_size: The batch size to use for upserting
    :return:
        An updated pinecone index
    """
    print("Upserting the embeddings to the Pinecone index...")
    shape = embeddings.shape

    ids = [str(i) for i in range(shape[0])]
    meta = [{text_field: text} for text in dataset[text_field]]

    to_upsert = list(zip(ids, embeddings, meta))

    for i in tqdm(range(0, shape[0], batch_size)):
        i_end = min(i + batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])
    return index


def llm_client(cohere_api_key: str):
    co = cohere.Client(api_key=cohere_api_key)
    return co


def llm_response(llm_client, original_query: str, prompt: str, model: str = "command-r-plus", augment_prompt=False):
    print("==================================")
    if augment_prompt:
        print(f"Augmented query:{prompt}")
    else:
        print(f"User Question:\n{original_query}\n")
    response = llm_client.chat(
        model=model,
        message=prompt,
    )
    wrapped_response = textwrap.fill(response.text, break_on_hyphens=True, width=120)
    print(f"Response:\n{wrapped_response}\n")


def augment_prompt(query, model, index, k: int = 5, text_field: str = 'text') -> tuple[str, str]:
    """
    Augment the prompt with the top 3 results from the knowledge base
    Args:

        :param query: The query to augment
        :param model: the llm model
        :param index: The vectorstore object
        :param k: required amount of top results from knowledge base
        :param text_field: The field in the dataset that contains the text
    Returns:
        tuple: augmented_prompt, source_knowledge
    """
    results = [float(val) for val in list(model.encode(query))]
    query_results = index.query(
        vector=results,
        top_k=k,
        include_values=True,
        include_metadata=True)['matches']
    text_matches = [match['metadata'][text_field] for match in query_results]

    source_knowledge = "\n\n".join(text_matches)

    augmented_prompt = f"""Using the contexts below, answer the question.Mention the date of the incident. 
    Add an explanation about the risks from this type of incidents.
    Your answer should answer the query and summarize the context in one paragraph. 
    Contexts: {source_knowledge}
    If the answer is not included in the source knowledge - say that you don't know.
    Query: {query}"""
    return augmented_prompt, source_knowledge


def basic_QA(llm_client, query, model: str = "command-r-plus"):
    prompt = f"""Answer the question.Mention the date of the incident. 
        Add an explanation about the risks from this type of incidents.
        Your answer should answer the query and summarize the context in one paragraph. 
        If the answer is not included in the source knowledge - say that you don't know.
        Query: {query}"""

    print(f"User Question:\n{query}\n")
    response = llm_client.chat(
        model=model,
        message=prompt,
    )
    wrapped_response = textwrap.fill(response.text, break_on_hyphens=True, width=120)
    print(f"Response:\n{wrapped_response}\n")


def explain():
    prepar = """Preparation As a preparation step, you need to prepare a vector database as an external knowledge 
    source that holds all additional information. This vector database is populated by following these steps:
    1. Collect and load your data
    2. Chunk your documents
    3. Embed and store chunks
    """
    collect_data = """The first step is to collect and load our data — For this part, We used the AI incidents Database, 
    which describe incidents occurred by known companies in industries. We manually collected 322 incidents record from 
    January 2022 until today (June 2024), and store it in a  XLSX file format (which we modified to a csv file). 
    Then, we cleaned up the data to remove special characters, 
    unite columns into one textual column (so it can be served as index). 
    We experimented different approaches for preprocessing, and the most appropriate approach was to design the data 
    as you can see in the format_data() function. 
    """
    preprocess_data = """Because the file in its original state is too long to fit into the LLM’s context window, 
    we chunked it into smaller pieces. 
    For this simple task, we chunked it into batches with size of 14 in each batch. To enable semantic search 
    across the text chunks, we generated the vector embeddings for each chunk and then store them together 
    with their embeddings. To generate the vector embeddings we used the "all-MiniLM-L6-v2" model 
    from SentenceTransformer package and used the Pinecone VectorDatabase to store them. 
    """
    upserting = """Once the vector database is populated, we define it as the retriever component, which fetches the 
    additional context based on the semantic similarity between the user query and the embedded chunks. 
    Next, to augment the prompt with the additional context, we customized the prompt template from the TA so it 
    will fill our needs properly."""
    final_chain = """Finally, we build a chain for the RAG pipeline, chaining together the retriever, the prompt 
    template and the LLM. Once the RAG chain is defined we can invoke it and answer the questions."""


def delete_index(pc_index, index_name: str):
    pc_index.delete_index(index_name)


def main():
    show_index_stat = False
    PINECONE_API_KEY, COHERE_API_KEY = initialize_client_connection()
    DATASET_NAME = "incident_ds.csv"

    model = embedding_model()
    llm_model = 'command-r-plus'
    use_RAG = False
    if use_RAG:
        dataset, embeddings = load_and_embedd_dataset(
            dataset_name=DATASET_NAME,
            rec_num=400,
            model=model,
            split='train',
            text_field='text'
        )
        embeddings_shape = embeddings.shape
        INDEX_NAME = 'ai-incidents'
        metric = 'cosine'

        pc = create_pinecone_index(pc_api_key=PINECONE_API_KEY, index_name=INDEX_NAME, dimension=embeddings_shape[1],
                                   metric=metric)
        index = pc.Index(INDEX_NAME)
        if show_index_stat:
            print(index.describe_index_stats())

        index_upserted = upsert_vectors(index, embeddings, dataset)
    questions = ["What kind of incident are the most reported?",
                 "What are the latest incident Open-AI was involved in?",
                 "Were there any cases that AI cause discrimination last year? how do you think they could avoid that?",
                 "What incidents regarding safety and reliability of autonomous driving took place in 2024?",
                 "Can you describe any notable cases where AI-generated papers influenced academic evaluations?"]

    co = llm_client(cohere_api_key=COHERE_API_KEY)
    for query in questions:
        if not use_RAG:
            basic_QA(llm_client=co, query=query, model=llm_model)
            continue
        else:
            augmented_prompt, source_knowledge = augment_prompt(query, model=model, index=index)
            llm_response(llm_client=co, original_query=query, prompt=augmented_prompt, model=llm_model,
                         augment_prompt=False)


if __name__ == '__main__':
    main()
