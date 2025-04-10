import os.path

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_ollama import OllamaEmbeddings

from ..models.llm_ouput import QuestionsForRAG
from ..models.llm_ouput import Summary, ChunkSummary, QueryAnswer
from ..knowledgebase.knowledge import Knowledge
from .make_prompt import *

from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.schema import Document
from langchain_core.output_parsers import JsonOutputParser

import logging


class LlamaWorker:
    def __init__(self, model_name="llama3.2"):
        """
        Initialize the LlamaWork class.

        Parameters:
        - model_name (str): The name of the Llama model.
        """
        self.retriever = None
        self.model_name = model_name
        self.embedding_model = "nomic-embed-text"
        # Initialize the Ollama LLM
        self.llm = OllamaLLM(
            model=self.model_name,
            num_thread=8,
            temperature=0
        )
        self.knowledge_base = Knowledge("summaries", "my_summaries")
        self.documents = []
        self.supportive_documents = []

    def load_text(self, text):
        """
        Load and store text content.

        Parameters:
        - text (str): The text content to be loaded.
        """
        self.documents.append(Document(page_content=text))

    def load_pdf(self, pdf_path):
        """
        Load and store a PDF document.

        Parameters:
        - pdf_path (str): The file path to the PDF document.
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
            loader = PyPDFLoader(pdf_path, extract_images=True)
            pdf_documents = loader.load()
            logging.info("PDF documents: ", pdf_documents)
            self.supportive_documents.extend(pdf_documents)
        except Exception as e:
            logging.error(f"Error loading PDF document: {e}")

    def load_image(self, image_path):
        """
        Load and store an image.

        Parameters:
        - image_path (str): The file path to the image.
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found at path: {image_path}")
            loader = UnstructuredImageLoader(image_path)
            data = loader.load()
            logging.info("Image data: ", data)
            self.supportive_documents.extend(data)
        except Exception as e:
            logging.error(f"Error loading image: {e}")

    def summarize_with_stuffing(self):
        prompt = get_stuffing_prompt()
        self.llm.format = Summary.model_json_schema()
        stuffing_chain = prompt | self.llm | JsonOutputParser()
        summary = stuffing_chain.invoke({"text": self.documents})
        return summary

    def summarize_with_map_reduce(self):
        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)
        docs = text_splitter.split_documents(self.documents)
        chunk_parser = JsonOutputParser()
        chunk_prompt = get_summarize_chunk_prompt()

        # Create an LLMChain
        self.llm.format = ChunkSummary.model_json_schema()
        chunk_chain = chunk_prompt | self.llm | chunk_parser

        chunk_summaries = []
        for doc in docs:
            try:
                chunk_summary = chunk_chain.invoke({"text": doc.page_content})
            except Exception as e:
                chunk_summary = {"summary": ""}
                logging.error(f"Error summarizing chunk: {e}")
            logging.info("chunk_summary: ", chunk_summary)
            chunk_summaries.append(chunk_summary)

        # Combine all key information
        final_key_info = "\n\n".join(chunk_summary["summary"] for chunk_summary in chunk_summaries)
        combine_parser = JsonOutputParser()
        combine_prompt = get_combine_chunk_prompt()
        self.llm.format = Summary.model_json_schema()
        combine_chain = combine_prompt | self.llm | combine_parser
        combined_summary = combine_chain.invoke({"text": final_key_info})
        return combined_summary

    def format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])

    def summarize_with_rag(self):
        prompt = get_stuffing_rag_prompt()
        self.build_retriever()
        self.llm.format = Summary.model_json_schema()

        chain = RunnableMap({
            "supportive_information": (lambda x: x["text"]) | RunnableLambda(self.format_docs) | self.retriever | self.format_docs,
            "text": lambda x: x["text"],
        }) | prompt | self.llm | JsonOutputParser()

        summary_rag = chain.invoke({"text": self.documents})
        return summary_rag

    def summarize(self, map_reduce=False, rag=False):
        """
        Extract key information, knowledge points, key tasks, and time points from the documents.

        Returns:
        - key_info (str): The extracted key information.
        """
        if not self.documents:
            return "No documents to extract information from."

        if rag:
            summary = self.summarize_with_rag()
        else:
            summary = self.summarize_with_map_reduce() if map_reduce else self.summarize_with_stuffing()
        self.add_summary_to_knowledge(summary)
        return summary

    def extract_query(self):
        propmt = get_extract_query_prompt()
        self.llm.format = QuestionsForRAG.model_json_schema()
        chain = propmt | self.llm | JsonOutputParser()
        questions = chain.invoke({"text": self.documents})
        return questions

    def build_retriever(self):
        if not self.supportive_documents:
            raise ValueError("No supportive documents loaded.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(self.supportive_documents)

        embedding_model = OllamaEmbeddings(model=self.embedding_model)
        vector_store = FAISS.from_documents(split_docs, embedding_model)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        self.retriever = retriever

    def add_summary_to_knowledge(self, summary):
        if isinstance(summary, dict):
            # convert to string
            summary = str(summary)
        self.knowledge_base.add([summary])

    def query_knowledge_base(self, query):
        return self.knowledge_base.query(query, top_k=5)

    def search_and_answer(self, query):
        supportive_information = self.query_knowledge_base(query)
        info_string = "\n\n".join([doc[1] for doc in supportive_information])
        if not supportive_information:
            return "No related information found."
        prompt = get_query_prompt()
        self.llm.format = QueryAnswer.model_json_schema()
        chain = prompt | self.llm | JsonOutputParser()
        answer = chain.invoke({
            "query": query,
            "supportive_information": info_string
        })
        return supportive_information, answer