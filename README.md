[Access the deployed app here](https://amyr-chatpdf.streamlit.app/)

# ChatPDF

> By Amyr Edmar L. Francisco
> 

<aside>
💡

This documents my thinking process in developing a simple Retrieval Augmented Generation (RAG) application from planning to implementation.

</aside>

# I. Requirement Planning

I laid out the most important requirements for this app in order to get a high-level understanding of what the finished product would look like. This will also help me identify what tools I would use in order to implement these requirements.

These are the minimum requirements I would implement for this project:

- Be able to upload PDF
- Be able to ask question in natural language
- Be able to receive accurate and concise answer from the model

In addition, I also added these bonus requirements that I would implement if time permits:

- Be able to upload multiple PDFs
- Be able to ask follow-up questions and maintain context
- Be able to provide text snippets from the PDF for each output for reference
- Be able to display the output in markdown for complex outputs

# II. Tech Stack Planning

After identifying the requirements, I would plan and research on what tools and libraries to use to implement these requirements. 

Since I have an experience building a RAG application from my internship, I have a knowledge in implementing this end-to-end. But the difference is, I used LlamaIndex before so, I have to research LangChain and read through its documentation. I also used a different embedding model and vectorDB before so I need to research if I can apply it to the same framework or I need to use a new one.

## Breaking down the architecture of a RAG application

To systematically identify the tools I would use, I will first break down the components that make up Retrieval-Augmented Generation (RAG). For each component, I’ll research and decide on what tool or library to use.

![rag_indexing-8160f90a90a33253d0154659cf7d453f](https://github.com/user-attachments/assets/fafb1415-5d7f-40bb-9113-a6879f770eff)

Image from Langchain [https://python.langchain.com/docs/tutorials/rag/](https://python.langchain.com/docs/tutorials/rag/)

### 1. Load

- Load the documents needed to answer user queries.
- Data formats can be PDF, text, CSV, JSON, etc.

<aside>
⚙

DirectoryLoader

[https://python.langchain.com/docs/how_to/document_loader_directory/](https://python.langchain.com/docs/how_to/document_loader_directory/)

</aside>

### 2. Split

- Divide the content of the data source in chunks or sections.
- This helps the model gather manageable amount of data and reduce unnecessary information to be fed to the model.
- *In the best-case scenario, the data is chunked such that each section answers different questions.*

<aside>
⚙

RecursiveTextSplitter

[https://python.langchain.com/docs/how_to/recursive_text_splitter/](https://python.langchain.com/docs/how_to/recursive_text_splitter/)

ChunkViz (Chunk size helper)

[https://chunkviz.up.railway.app/](https://chunkviz.up.railway.app/)

</aside>

### 3. Embed

- This converts the chunked document into a vector representation, essentially numbers that capture their semantic meaning.
- I like to think of it this way, if we have a 2D space, embeddings are like plotting a chunk of text in this space. then, if a query is made, its embeddings is also computed and plotted in this space. the chunk of text closer to the query text are more related to the text and might contribute to answering that question.
    
  <img width="641" alt="Screenshot 2025-06-24 at 10 20 38 AM" src="https://github.com/user-attachments/assets/3661445b-06e3-4ecc-8206-30809f2d895d" />
    
    Image from Microsoft [https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-generate-embeddings](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-generate-embeddings)
    
- This aids the model by first extracting the relevant chunk of text before feeding it to the model so we get a more accurate response related to the query.

<aside>
⚙

sentence-transformers/all-mpnet-base-v2 (HuggingFace)

</aside>

### 4. Store

- We need a place for the embeddings to be stored so they can be searched later.
- This is usually done with a vector store.

<aside>
⚙

ChromaDB

</aside>

### 5. Retrieve and generate

![rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a](https://github.com/user-attachments/assets/5ce4ede5-fa10-4bbb-b5c2-d36fdd3f2cb3)


Image from Langchain [https://python.langchain.com/docs/tutorials/rag/](https://python.langchain.com/docs/tutorials/rag/)

- The query is converted to an embedding using the same embedding model used in the documents, the system gathers document embeddings close to it, these chunks are considered to be the most relevant to the user’s query.
- The extracted text chunks along with the user’s query are then fed into the LLM to generate a response.
- This is done through prompt engineering, basically we instruct the LLM to answer the user’s query using the relevant text chunks we provided.
- It’s like copying and pasting the user’s query and the reference texts to ChatGPT and instructing it to generate a response. The only difference is, this is all automated with the help of the framework we will be using (LangChain, LlamaIndex, etc.).

<aside>
⚙

Groq API (meta-llama/llama-4-scout-17b-16e-instruct)

[https://console.groq.com/docs/model/llama-3.3-70b-versatile](https://console.groq.com/docs/model/llama-3.3-70b-versatile)

</aside>

# III. Experimentation (Jupyter Notebook)

Before writing the functions and creating the files needed for the final product, I needed to learn how LangChain works and experiment with different techniques to achieve the requirements. I did this by creating a simple RAG chain in Jupyter notebook in order to get a glimpse of how LangChain works in different stages of the RAG pipeline.

*(Explain how i achieved chat memory)…*

# IV. Challenges

When I start to write the functions and classes needed in order to streamline the RAG pipeline, I encountered countless bugs.

## Tool Calling Error

<img width="181" alt="Screenshot_2025-06-16_at_3 05 25_PM" src="https://github.com/user-attachments/assets/5da00ddb-3c9e-43a3-8c04-50144e860c86" />

```jsx
@tool(response_format="content_and_artifact", description="Retrieve information related to a query.",)
	def retrieve(query: str, config: RunnableConfig):
		vector_store = config["configurable"].get("vector_store")
		retrieved_docs = vector_store.similarity_search(query, k=self.retrieval_k)
		serialized = "\n\n----------\n\n".join(
			(f"Source: {doc.metadata["source"]}\nContent: {doc.page_content}")
			for doc in retrieved_docs
		)
		return serialized, retrieved_docs
```

I’m trying to achieve the graph shown above where if the query of the user doesn’t need a retrieval, it will not search the vector store and generate a response as soon as the query arrives. If the query of the user needs a retrieval, it will call the retrieve tool and return a content and an artifact which can be used by the LLM to generate an accurate response.

<aside>
⚠️

The Problem:

At first, I tried to make `retrieve()` an instance method inside the class LangChainRAG. However, I found out that it’s not designed to be used in an object because of the “self” parameter.

</aside>

<aside>
✅

The Solution:

I wrapped it inside `get_graph(self)`, an instance method that will be called by the frontend which will trigger the graph to compile. I included all the nodes inside this instance method for encapsulation.

```python
def get_graph(self):
	@tool(
	response_format="content_and_artifact",
	description="Retrieve information related to a query.",
	)
		def retrieve(query: str, config: RunnableConfig):
			vector_store = config["configurable"].get("vector_store")
			retrieved_docs = vector_store.similarity_search(query, k=self.retrieval_k)
			serialized = "\n\n----------\n\n".join(
				(f"Source: {doc.metadata["source"]}\nContent: {doc.page_content}")
				for doc in retrieved_docs
			)
			return serialized, retrieved_docs
		
		...
```

</aside>

## Vector Store Leak Bug

Because I’m using Streamlit, there is a tendency that some variables I use will be leaked on other users because they share the same global variables. This is the reason I use `st.session_state` to store exclusive variables for a user including the vector store containing the embeddings of the uploaded files of the user.

However, It did not occur to me that my vector store db (ChromaDB) will behave in a way I did not expect. 

> *Scenario*
If two users a and b are using the app at the same time, files uploaded by user a will leak to the vector store used by user b, thus, leaking knowledge as well!
> 

<aside>
⚠️

The Problem:

I did not specify a collection name to the vector store, so it stores all embeddings in a shared collection for all users.

</aside>

<aside>
✅

The Solution:

Simply specify a collection name, in this case, I used the unique thread id generated for each user session.

```python
vector_store = Chroma.from_documents(
        docs, embeddings, collection_name=collection_name
    )
```

</aside>

## Vector Store Cleanup Bug

When a user uploads or deletes a file, I want the vector store to only contain the embeddings of the currently uploaded files and I want it to react to changes such as deletion. 

What I do is I just replace `self.vector_store` every time a user deletes a file. However, It did not occur to me that it still uses the same database collection as before, preserving the embeddings of the deleted file.

> *Scenario*
If a user uploaded two files and deletes a file, leaving only 1 file, the vector store will still contain the embeddings of both files including the deleted one. This behavior results in the LLM still having knowledge of the deleted file.
> 

<aside>
⚠️

The Problem:

I’m not resetting the vector store db every time I recalculate the embeddings. Resulting in the persistence of the embeddings of the deleted file.

</aside>

<aside>
✅

The Solution:

I just delete the collection if it exists. I do this inside the function that recalculates the embeddings. Making the vector store fresh every time.

```python
# Reset collection if exists
if self.vector_store:
	self.vector_store._client.delete_collection(self.thread_id)
```

</aside>

# V. Conclusion

Finally after 2 sleepless nights, I finished the requirements I mentioned at the start of this project:

- Be able to upload multiple PDFs
- Be able to ask question in natural language
- Be able to receive accurate and concise answer from the model
- Be able to ask follow-up questions and maintain context
- Be able to provide text snippets from the PDF for each output for reference
- Be able to display the output in markdown for complex outputs

I have learned a lot from this and this serves as a refresher for me on the architecture of RAG. There are few things I’d like to explore more in LangChain because it offers an efficient way to develop AI applications. Overall, this has been a fun journey and I hope I get to learn more about this field and work with Thinking Machines as a Machine Learning Engineer focused on Generative AI to further improve my skills and develop tools that uses AI to provide values to the company and its clients.

Thank you for this opportunity! 😊
