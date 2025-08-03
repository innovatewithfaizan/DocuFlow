import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY # type: ignore

class DocumetAnalyser:
    def __init__(self):
        self.log = CustomLogger.get_logger(__name__)
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm

            #prepare parsers
        
        except Exception as e:
            self.log.error(f"error initializing DocumentAnalyser: {e}")
            raise DocumentPortalException("Error in Document Analyzer initializarion", sys)