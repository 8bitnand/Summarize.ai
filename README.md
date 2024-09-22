# Summarize.ai

This repository contains code for training a lightweight text summarization model using the t5-small architecture and deploying it via a RESTful API using FastAPI. The model is fine-tuned on the `CNN/DailyMail` dataset. This README provides an overview of the training process, differences before and after training, detailed code explanations, and instructions to run the API.

**Before Training**

Model: Pre-trained t5-small from Hugging Face.

- Capabilities: 
    * General language understanding and generation.
    * Not specialized in summarization of news articles.
- Limitations:
  - May produce less accurate summaries.
   - Lacks domain-specific knowledge from the CNN/DailyMail 
   
**After Training**

Model: Fine-tuned t5-small on CNN/DailyMail dataset.

- Capabilities:
    - Improved summarization of news articles.
    - Generates more concise and relevant summaries.
- Benefits:
    - Adapts to the style and content of the dataset.
    - Better performance on domain-specific summarization tasks.

#### How to run
> [!Note]
>
> for training, as it's a collab notebook you don't need to do much just run each cell.  


Install the dependencies

- transformers 
- datasets
- torch
- fastapi 
- uvicorn 
- pydantic
- sentencepiece

**Dataset Processing** 

Using 🤗 api load the `cnn_dailymail` whcih is an English-language dataset containing news articles The current which can be used for machine reading and comprehension and abstractive question answering.

get_tokenized_datasets

The T5 model treats every NLP task as a `text-to-text` problem. This means that both the input and output are text strings.By prepending `summarize` to the input text, we explicitly instruct the T5 model that the task at hand is summarization. This helps the model activate the appropriate learned parameters associated with summarization tasks during inference.

Creates a processed copy of the tokenized dataset and stores it into the local storage `t5_processed_data`

```
dataset preview 
```

**Model training**

We used the [T5ForConditionalGeneration](https://huggingface.co/transformers/v3.0.2/model_doc/t5.html#t5forconditionalgeneration) model from the Hugging Face Transformers library because it is specifically designed for sequence-to-sequence (seq2seq) tasks like text summarization. This model is part of the T5 (Text-to-Text Transfer Transformer) family, which treats every NLP task as a text-to-text problem, making it highly versatile and effective for generating summaries from input texts.

with the help of [transformers](https://huggingface.co/docs/transformers/en/training) library load the `t5-small` model from the 🤗 hub and finetune it on the dataset we processed.

> [!Note]
>
> Here im using the Google Colab to train my model with the free tiere gpu so the model size is a bottle neck. with more resources you can finetune a larger model. 

```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,   # can be adjusted if the model performance worse 
    per_device_train_batch_size=32, # limited by 15GB of Vram un GPU
    per_device_eval_batch_size=16,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=1000, # save intermediate output so as to not lose trained model if runtime disconnected 
    logging_steps=100,
    learning_rate=5e-5, # as the model is pretrained a small LR works better, as I was training This was too low, but due to resource limitations I cannot experiment with other values like 3.3e-4, 3.3e-3 
    weight_decay=0.01, # reduce LR after steps
    save_total_limit=3,
    fp16=torch.cuda.is_available(), # used for faster training and save space in memory without using  32 bit 
)

trainer.train() # start training process 
# use  trainer.train(resume_from_checkpoint=True) to load from the saved checkpoints  
```
Model is automatically saved in the results/ folder and the related tensorboard graphs 

Finaly save the trained model 
```python
trainer.save_model('/content/drive/MyDrive/Colab Notebooks/fireai/t5-small-cnn_dailymail')
tokenizer.save_pretrained('/content/drive/MyDrive/Colab Notebooks/fireai/t5-small-cnn_dailymail')

```

Next step is to push the model to 🤗 My model is [here](https://huggingface.co/nand-tmp/t5-small-cnn_dailymail/) We will load the model in our api code from HF directly.

#### API 
```bash 
fastapi dev api/main.py
```
I used github codespaces it lets you use and forward ports and here is the request response format 

```bash
curl -i -X POST -H 'Content-Type: application/json' -d '{"prompt":"T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format. T5 works well on a variety of tasks out-of-the-box by prepending a different prefix to the input corresponding to each task, e.g.: for translation: translate English to German: …, summarize: …. For more information about which prefix to use, it is easiest to look into Appendix D of the paper ." }' http://127.0.0.1:8000/summarise
```

![API](extra/API.png)


