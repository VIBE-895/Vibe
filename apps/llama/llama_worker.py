import os.path

from models.llm_ouput import QuestionsForRAG
from .. import PROJECT_BASE_PATH
from ..models.llm_ouput import Summary, ChunkSummary
from .make_prompt import get_summarize_chunk_prompt, get_combine_chunk_prompt, get_stuffing_prompt, get_extract_query_prompt

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
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
        self.model_name = model_name
        # Initialize the Ollama LLM
        self.llm = OllamaLLM(
            model=self.model_name,
            num_threads=8,
            temperature=0
        )
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
            self.documents.extend(pdf_documents)
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
            self.documents.extend(data)
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

    def summarize(self, map_reduce=False):
        """
        Extract key information, knowledge points, key tasks, and time points from the documents.

        Returns:
        - key_info (str): The extracted key information.
        """
        if not self.documents:
            return "No documents to extract information from."

        if map_reduce:
            return self.summarize_with_map_reduce()
        return self.summarize_with_stuffing()

    def extract_query(self):
        propmt = get_extract_query_prompt()
        self.llm.format = QuestionsForRAG.model_json_schema()
        chain = propmt | self.llm | JsonOutputParser()
        questions = chain.invoke({"text": self.documents})
        return questions

    # Placeholder methods for future integration with tarily and ChromaDB
    def set_vector_store(self, vector_store):
        """
        Set the vector store for Retrieval Augmented Generation (RAG).

        Parameters:
        - vector_store: The vector store instance (e.g., ChromaDB).
        """
        self.vector_store = vector_store

    def set_retrieval_augmentation(self, retriever):
        """
        Set the retriever for RAG.

        Parameters:
        - retriever: The retriever instance.
        """
        self.retriever = retriever
