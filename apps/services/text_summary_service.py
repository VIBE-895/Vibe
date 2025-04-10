# Created by guxu at 10/24/24
import os
import logging

from ..llama.llama_worker import LlamaWorker
from ..utils.utils import is_image, is_pdf


class TextSummaryService:
    def __init__(self, model_name="stablelm-zephyr:3b"):
        self.worker = LlamaWorker(model_name)

    def summarize_text(self, text, supporting_documents=None):
        self.worker.load_text(text)
        if supporting_documents:
            for file in supporting_documents:
                if not os.path.exists(file):
                    logging.warning(f"File {file} not found.")
                    continue
                if is_pdf(file):
                    logging.info(f"PDF file {file} found.")
                    self.worker.load_pdf(file)
                elif is_image(file):
                    logging.info(f"Image file {file} found.")
                    self.worker.load_image(file)
        return self.worker.summarize()

    def intelligent_query(self, query):
        supportive_doc, answer = self.worker.search_and_answer(query)
        return supportive_doc, answer