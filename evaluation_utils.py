import json
import itertools
import warnings

from copy import deepcopy

import numpy as np

from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
    )


def get_accuracy(ranks: list) -> float:
  tmp = [1 if np.nonzero(i)[0].shape[0] != 0 else 0 for i in ranks]
  return sum(tmp) / len(tmp)


def get_mrr(correct_doc: np.ndarray) -> np.ndarray:
  doc_rank = np.array(list(range(1, correct_doc.shape[0] + 1)))
  return correct_doc / doc_rank


def get_vectorstore(docs: list, model_name: str = "all-MiniLM-L6-v2"):
  # load the model wrapped as an embedding function
  embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

  # encode and load the data into Chroma
  db = Chroma(collection_name=model_name)
  db = db.from_documents(docs, embedding_function)
  return db


def evaluate_mrr_vectorstore(docs: list,
                             eval_data: list,
                             model_name: str = "all-MiniLM-L6-v2",
                             search_type: str = "similarity",
                             k: int = 4,
                             thres: float | None = None) -> tuple:
  warnings.filterwarnings("ignore")
  db = get_vectorstore(docs, model_name)

  if search_type == "similarity_score_threshold":
    search_kwargs = {"k": k, "score_threshold": thres}
  else:
    search_kwargs = {"k": k}

  retriever = db.as_retriever(
      search_type=search_type,
      search_kwargs=search_kwargs
      )

  mrr_scores = []

  tmp_data = deepcopy(eval_data)

  for context_questions in tmp_data:
    context = context_questions.pop("context")
    for question in context_questions:
      correct_doc = np.zeros(k)
      retrieved_chunks = retriever.get_relevant_documents(context_questions[question])
      for i, chunk in enumerate(retrieved_chunks):
        if chunk.page_content == context:
          correct_doc[i] = 1
        else:
          correct_doc[i] = 0
      if correct_doc.shape[0] > 0:
        mrr_scores.append(get_mrr(correct_doc=correct_doc))
  accuracy = get_accuracy(mrr_scores)
  mrr_scores = list(itertools.chain(*mrr_scores))
  mrr_scores = np.array(mrr_scores)[np.nonzero(np.array(mrr_scores))]
  db._collection.delete(db._collection.get()["ids"])
  db.delete_collection()
  return accuracy, (np.average(mrr_scores) if len(mrr_scores)>0 else 0)


def evaluate(k: list, search_type: list, models: list, thresholds: list, docs: list, eval_data: list) -> dict:
  res = {}

  for model_name in tqdm(models):
    types = {}
    for s_t in search_type:
      mrr_list = []
      accuracy_list = []
      thres_dict = {}
      # if we want to use a threshold
      if s_t == "similarity_score_threshold":
        for thres in thresholds:
          mrr_list = []
          accuracy_list = []
          scores = {}
          for k_n in k:
            accuracy, mrr = evaluate_mrr_vectorstore(
              docs=docs,
              eval_data=eval_data,
              model_name=model_name,
              search_type=s_t,
              k=k_n,
              thres=thres
            )
            mrr_list.append(round(mrr, 3))
            accuracy_list.append(accuracy)
          scores["mrr"] = mrr_list
          scores["accuracy"] = accuracy_list
          thres_dict[str(thres)] = scores
        types[s_t] = thres_dict
      # otherwise
      else:
        scores = {}
        for k_n in k:
          accuracy, mrr = evaluate_mrr_vectorstore(
            docs=docs,
            eval_data=eval_data,
            model_name=model_name,
            search_type=s_t,
            k=k_n
          )
          mrr_list.append(round(mrr, 3))
          accuracy_list.append(round(accuracy, 3))
        scores["mrr"] = mrr_list
        scores["accuracy"] = accuracy_list
        types[s_t] = scores
    res[model_name] = types
  return res


if __name__ == "__main__":
  # Load Document
  loader = WikipediaLoader(query="Aristophanes")

  loader.requests_kwargs = {"verify": False}

  document = loader.load()

  # Split
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  splits = text_splitter.split_documents(document)

  with open("./question_context_aristophanes.json", "r") as file:
    eval_data = json.load(file)

  search = {
    "k" : list(range(2, 11)),
    "search_type" : ["similarity", "mmr", "similarity_score_threshold"],
    "models" : ["distiluse-base-multilingual-cased-v2", "all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    "thresholds" : [0.3, 0.5, 0.9],
    "docs": splits,
    "eval_data": eval_data
  }

  results = evaluate(**search)
