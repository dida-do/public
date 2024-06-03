# Download the models
Run ```download_weights.ipynb``` to donwload ```BAAI/bge-large-en-v1.5``` and ```BAAI/bge-reranker-large```.

# Create Database
I use qdrant for storing my data. To create the vector database run `create_vector_database.py`
If you want to host qdrant, see this [link](https://qdrant.tech/documentation/quick-start/) for more detail.


# langfuse
To use langfuse locally you need to install docker. `langfuse` then can be started as following:
```
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# Run server and database
docker compose up -d
```
You can also create a free acount on [langfuse-demo](https://langfuse.com/docs/demo).

I recommend to create a free account. After create an account and login, you have to create a `New project` and then go into `Settings` and `Create new API keys` and copy `Secret Key` to `<LANGFUSE_SECRET_KEY>`, see below.

# Run workshop script
To run `workshop-lcel.ipynb` you need to create a `.env` file. In this file you have to add:
```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
LANGFUSE_SECRET_KEY=<LANGFUSE_SECRET_KEY>
LANGFUSE_PUBLIC_KEY=<LANGFUSE_PUBLIC_KEY>
LANGFUSE_HOST=https://cloud.langfuse.com
```
