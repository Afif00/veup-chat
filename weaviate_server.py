import weaviate
import subprocess
from contextlib import contextmanager
import os
from weaviate.classes.config import Property, DataType, Configure


@contextmanager
def suppress_subprocess_output():
    """
    Context manager that suppresses the standard output and error 
    of any subprocess.Popen calls within this context.
    """
    # Store the original Popen
    original_popen = subprocess.Popen

    def patched_popen(*args, **kwargs):
        # Redirect the stdout and stderr to subprocess.DEVNULL
        kwargs['stdout'] = subprocess.DEVNULL
        kwargs['stderr'] = subprocess.DEVNULL
        return original_popen(*args, **kwargs)

    try:
        # Apply the patch by replacing subprocess.Popen with patched_popen
        subprocess.Popen = patched_popen
        # Yield control back to the context
        yield
    finally:
        # Ensure that the original Popen method is restored
        subprocess.Popen = original_popen
with suppress_subprocess_output():
    client = weaviate.connect_to_embedded(
        persistence_data_path="./dataset/collections",
        #version="1.28.3",
        environment_variables = {
            "ENABLE_API_BASED_MODULES": "true",
            "ENABLE_MODULES": 'text2vec-transformers, reranker-transformers',
            "DEFAULT_VECTORIZER_MODULE": 'text2vec-transformers',
            "TRANSFORMERS_INFERENCE_API":"http://127.0.0.1:5009/",
            "RERANKER_INFERENCE_API":"http://127.0.0.1:5009/"
        }
    )

    # Drop if present
    if client.collections.exists("Faq"):
        client.collections.delete("Faq")
    
    # Create with vectorizer
    client.collections.create(
        name="Faq",
        properties=[
            Property(name="question", data_type=DataType.TEXT),
            Property(name="answer",  data_type=DataType.TEXT),
            Property(name="type",    data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.text2vec_transformers()
    )

    
    if client.collections.exists("Products"):
        client.collections.delete("Products")

    client.collections.create(
        name="Products",
        vectorizer_config=Configure.Vectorizer.text2vec_transformers(),  # or .text2vec_transformers(), or .none()
        properties=[
            Property(name="gender", data_type=DataType.TEXT),
            Property(name="masterCategory", data_type=DataType.TEXT),
            Property(name="subCategory", data_type=DataType.TEXT),
            Property(name="articleType", data_type=DataType.TEXT),
            Property(name="baseColour", data_type=DataType.TEXT),
            Property(name="season", data_type=DataType.TEXT),
            Property(name="year", data_type=DataType.INT),
            Property(name="usage", data_type=DataType.TEXT),
            Property(name="productDisplayName", data_type=DataType.TEXT),
            Property(name="price", data_type=DataType.NUMBER),
            Property(name="product_id", data_type=DataType.INT),
        ],
    )

    
    
