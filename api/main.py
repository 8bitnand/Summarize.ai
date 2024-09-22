from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI()
HUGGINGFACE_URL = "nand-tmp/t5-small-cnn_dailymail"
model = T5ForConditionalGeneration.from_pretrained(HUGGINGFACE_URL)
tokenizer = T5Tokenizer.from_pretrained(HUGGINGFACE_URL)

class SummarizeRequest(BaseModel):
    prompt: str

@app.post("/summarise")
async def summarise_text(request: SummarizeRequest):
    try:
        input_ids = tokenizer.encode('summarize: ' + request.prompt, return_tensors='pt', max_length=512, truncation=True)
        outputs = model.generate(input_ids=input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
