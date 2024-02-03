import base64
import io
import os
from typing import List
from pydantic import BaseModel
from indexify_extractor_sdk import Extractor, Content, Feature, EmbeddingSchema, ExtractorSchema



class InputParams(BaseModel):
    a: int = 0
    b: str = ""


NOUGAT_CMD = "nougat --markdown --out 'out'"


class MyExtractor(Extractor):
    detection_transform = None
    model = None
    device = None

    def __init__(self):
        super().__init__()


    def extract(
        self, content: List[Content], params: InputParams
    ) -> List[List[Content]]:
        markdown_data = []
        for bytestream in content:
            with io.BytesIO(bytestream.data) as open_pdf_file:
                with open("./temp/temp.pdf", 'wb') as f:
                    f.write(base64.b64decode(open_pdf_file))

        pdf_path = '/temp'
        for pdf in os.listdir(pdf_path):
            os.system(f"{NOUGAT_CMD} pdf /temp/{pdf}")

        markdown_path = '/out'
        for markdown in os.listdir(markdown_path):
            with open(f"{markdown_path}/{markdown}") as markdown_file:
                markdown_data.append(markdown_file.read())
        return markdown_data

    @classmethod
    def schemas(cls) -> ExtractorSchema:
        """
        Returns a list of options for indexing.
        """
        return ExtractorSchema(
            features={"embedding": EmbeddingSchema(distance_metric="cosine", dim=3)},
        )
