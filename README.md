# Simple RAG system about Aristophanes and Ancient Greek Comedy

This project explores a simple retrieval augmented generation system designed to answer questions about Aristophanes and ancient Greek comedy. It is developed using the [transformers](https://huggingface.co/docs/transformers/index), [Langchain](https://python.langchain.com/docs/get_started/introduction), and the [Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma) integration in Langchain. The LLM used is [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), downloaded from [Hugging Face Hub](https://huggingface.co/models).

The repository contains the following files:
1. A <b>SimpleRag.ipynb</b> notebook that defines the RAG system.
2. A <b>DataGeneration.ipynb</b> notebook that generates synthetic datasets for the evaluation of the system.
3. An <b>Evaluation.ipynb</b> notebook that evaluates the Chroma Vectorstore and the LLM based on MRR score and ROUGE/BLEU metrics respectively.
4. A helper script named <b>evaluation_utils.py</b> that defines helper evaluation functions used in the Evaluation.ipynb notebook.

***Generation example***

question: 

```Name the surviving Aristophanes' works and give a short description```

answer:

```Aristophanes is best known for his comedies. Some surviving works include "The Clouds," "Lysistrata," and "Frogs." In "The Clouds," Socrates is satirized as a teacher who corrupts young men. "Lysistrata" revolves around women withholding sex from their husbands to end the Peloponnesian War. "Frogs" features Dionysus traveling to the underworld to retrieve Euripides. Aristophanes also contributed to language studies, compiling lists of foreign words and unusual expressions.```

## SimpleRag notebook

The data used to augment the model's generation are retrieved from wikipedia using the keyword <b>"Aristophanes"</b> with [Wikipedia](https://python.langchain.com/docs/integrations/tools/wikipedia)'s integration in Langchain. The steps taken are as follows:
1. Download the data.
2. Use the RecursiveCharacterSplitter from Langchain.
3. Create a vectorstore using Chroma with [MiniLM](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) as an embedding model within the [SentenceTransformers](https://www.sbert.net/index.html) framework.
4. Use Langchain to define the retrieval augmented generation chain.

## Evaluation
The evaluation step aims to evaluate the ability of the Vectorstore the retrieve relevant documents for the model and the ability of the model to generate useful answers. Both evaluations have limitations such as the lack of ground truth information about what is relevant and what is useful. In an attempt to address this issue first 3 synthetic datasets are generated using an LLM (in this case Mistral-7B-Instruct) that will serve as ground truths.
### Generated datasets
1. The ```question_context_aristophanes.json``` contains 52 samples of ```{
                                            "context": "text used as reference to generate 4 questions",
                                            "question1": "generated question",
                                            "question2": "generated question",
                                            "question3": "generated question",
                                            "question4": "generated question"
                                            } 
                                            ``` groups.
2. The ```question_answer_pairs_aristophanes.json``` contains 106 ```{"question": "generated question", "answer": "generated answer"}``` pairs.
3. The ```context_question_answer_triplets_aristophanes.json``` is identical with the ```question_answer_pairs_aristophanes.json``` with the addition that it contains a field with the context used to generate the question answer pairs.

### Evaluate Vectorstore
Evaluation of the Mean Reciprocal Rank scores for <b>Maximal Marginal Relevance</b> and <b>Similarity</b> search. 

![mmr search](/images/mmr_mrr.png)

![similarity search](/images/similarity_mrr.png)

From the two search methods <b>MMR</b> yields the best performance.

### Evaluate Model generation

***Default Values***

The default parameters used in the RAG system where:
* <b>Similarity</b> search
* <b>k</b> = 5

<b>Rouge</b> score between the <b>answer</b> (reference) and model response:
* ```Rouge1 = 0.38```
* ```Rouge2 = 0.20```
* ```RougeL = 0.30```

<b>Rouge</b> score between the <b>context</b> (reference) and model response:
* ```Rouge1 = 0.36```
* ```Rouge2 = 0.15```
* ```RougeL = 0.25```

<b>Bleu</b> score between the <b>answer</b> (reference) and model response:
* ```Bleu = 0.15```

<b>Bleu</b> score between the <b>context</b> (reference) and model response:
* ```Bleu = 0.09```

***New Values***

The new values obtained from the Vectorstore evaluation
* <b>MMR</b> search
* <b>k</b> = 10

<b>Rouge</b> score between the <b>answer</b> (reference) and model response:
* ```Rouge1 = 0.39```
* ```Rouge2 = 0.22```
* ```RougeL = 0.31```

<b>Rouge</b> score between the <b>context</b> (reference) and model response:
* ```Rouge1 = 0.36```
* ```Rouge2 = 0.15```
* ```RougeL = 0.25```

<b>Bleu</b> score between the <b>answer</b> (reference) and model response:
* ```Bleu = 0.15```

<b>Bleu</b> score between the <b>context</b> (reference) and model response:
* ```Bleu = 0.09```

## Conclusions
The results show that there is slight improvement with the new parameters for the system in terms of <b>Rouge</b> score. Different models can be evaluated as well as different vectorstores other than Chroma and search methods it supports.

