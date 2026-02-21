from google.cloud import storage, bigquery, logging as cloud_logging
from google.cloud import documentai_v1 as documentai
from google.cloud import aiplatform
import config
from datetime import datetime

class GCPServices:
    def __init__(self):
        self.project_id = config.GCP_PROJECT_ID
        self.location = config.GCP_LOCATION
        self.bucket_name = config.GCP_BUCKET_NAME
    
    def upload_to_gcs(self, file_content: bytes, filename: str) -> str:
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(f"documents/{filename}")
        blob.upload_from_string(file_content)
        return f"gs://{self.bucket_name}/documents/{filename}"
    
    def log_query(self, query: str, response: str, model_type: str):
        client = cloud_logging.Client(project=self.project_id)
        logger = client.logger("docqna-queries")
        logger.log_struct({
            "query": query,
            "response": response,
            "model_type": model_type,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def save_to_bigquery(self, data: dict, table_id: str):
        client = bigquery.Client(project=self.project_id)
        table_ref = f"{self.project_id}.docqna.{table_id}"
        errors = client.insert_rows_json(table_ref, [data])
        return len(errors) == 0
    
    def process_with_document_ai(self, file_content: bytes, processor_id: str) -> str:
        client = documentai.DocumentProcessorServiceClient()
        name = f"projects/{self.project_id}/locations/{self.location}/processors/{processor_id}"
        
        raw_document = documentai.RawDocument(
            content=file_content,
            mime_type="application/pdf"
        )
        
        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document
        )
        
        result = client.process_document(request=request)
        return result.document.text
