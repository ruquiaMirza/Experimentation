import faiss
from chromadb.config import Settings
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from gemini_connect import GeminiCustomGenerativeAI
import numpy as np

class EmbeddingGenerator():
    from gemini_connect import GeminiCustomGenerativeAI

    class Chroma:
        def __init__(self,llmcall):
            print("CHROMA initialized!")
            persist_directory = "/Users/ruquiamirza/PycharmProjects/GAAP_v2/chroma_db"
            chroma_client = chromadb.Client(
                Settings(
                    persist_directory=persist_directory,
                    # chroma_api_impl="local",  # Use "cloud" for cloud-based ChromaDB
                )
            )
            collection_name = "default_collection"
            self.collection = chroma_client.get_or_create_collection(collection_name)
            self.llmcall = llmcall


        def load_data_to_vector(self,data):
            

            embeddings=[]
            for text in data:
                embedding = self.llmcall.get_embeddings(text)
                embeddings.append(embedding)
            for idx, (text, embedding) in enumerate(zip(data, embeddings)):
                self.collection.add(
                    ids=[str(idx)],
                    documents=[text],
                    embeddings=[embedding],
                )

        def chroma_search_similar_texts(self, chroma_query, k=3):
           
            query_embedding = self.llmcall.get_embeddings(chroma_query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
            )
            return [
                {"text": doc, "score": score}
                for doc, score in zip(results["documents"][0], results["distances"][0])
            ]
    class FAISS:

        def __init__(self,llmcall):
            self.llmcall=llmcall


        def load_data_to_vector(self,data):
        
            # Initialize FAISS index
            # Generate embeddings and add them to the index

            embeddings=[]
            for text in tqdm(data):
                embedding = self.llmcall.get_embeddings(text)
                #print(f"Values:{embedding}")
                embeddings.append(embedding)

            #print(f"embeddings:{embeddings,len(embeddings)}")
            embeddings_array = np.array(embeddings).astype('float32')
            embedding_dimension = embeddings_array.shape[1]
            print(f"Generated embeddings with dimension: {embedding_dimension}")

            index = faiss.IndexFlatL2(embedding_dimension)
            print("FAISS index initialized.")

            index.add(embeddings_array)

            print(f"FAISS index populated with {index.ntotal} embeddings.")
            return index


        def search_faiss(self,index: faiss.IndexFlatL2, query, llm, top_k = 5):



            query_embedding = self.llmcall.get_embeddings(query)

            query_embeddings_array = np.array([query_embedding]).astype('float32')
            distances, indices = index.search(query_embeddings_array, top_k)
            return distances[0], indices[0]

    def __init__(self, method="FAISS",llm="GEMINI"):
        """
        Initialize EmbedGenerator with the specified method.

        Args:
            method (str): The embedding method to use ("FAISS" or "CHROMA").
        """
        self.method = method.upper()
        self.llm = llm.upper()
        if self.llm=='GEMINI':
            llmcall=GeminiCustomGenerativeAI()
            print(llmcall)
        if self.method == "FAISS":
            self.generator = self.FAISS(llmcall)
        elif self.method == "CHROMA":
            self.generator = self.CHROMA(llmcall)
        else:
            raise ValueError("Invalid method! Use 'FAISS' or 'CHROMA'.")



    def load_data_to_vector(self, data,llmcall):

        return self.generator.load_data_to_vector(data)


if __name__ == "__main__":


    # Initialize the Gemini embedding generator
    embd = EmbeddingGenerator(method='FAISS',llm='GEMINI')


    # Example data to index

    data = [
        "How does AI work?",
        "What is machine learning?",
        "Explain neural networks.",
        "Artificial intelligence is rapidly changing the world.",
        "Summer is the warmest season of the year.",
        "Bananas are a good source of potassium.",
        "The capital of France is Paris."
    ]

    # Load data into FAISS

    faiss_index = embd.load_data_to_vector(data, 'GEMINI')

    # Save the FAISS index for later use
    faiss.write_index(faiss_index, "faiss_index.bin")
    print("FAISS index saved to faiss_index.bin.")

    # # Perform a search
    # query = "What is the capital of France?"
    # distances,indices  = gemini.search_faiss(faiss_index, query, 'gemini')
    #
    #
    # # Print the results
    # print("\nSearch Results:")
    # for i, (idx, dist) in enumerate(zip(indices, distances)):
    #     print(f"Rank {i + 1}:")
    #     print(f"Text: {data[idx]}")
    #     print(f"Distance: {dist}")

    # gemini.load_data_to_chroma(data)
    #
    # chroma_query = "Which is the warmest season of the year?"
    # try:
    #
    #     results = gemini.chroma_search_similar_texts(chroma_query, k=3)
    #
    #     # Print the search results
    #     print("Search Results:")
    #     for result in results:
    #         print(f"Text: {result['text']}, Similarity Score: {result['score']}")
    # except Exception as e:
    #     print(f"Failed to perform search: {e}")
