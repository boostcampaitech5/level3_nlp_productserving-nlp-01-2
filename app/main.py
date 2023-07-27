from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from model import work

app = FastAPI()

class ItemIn(BaseModel):
    text: str

class ItemOut(BaseModel):
    summary: str
    genre: str
    mood: str


@app.post("/summarize/", response_model=ItemOut)
def create_summary(item: ItemIn):
    print('\n',"="*100,item.text,"="*100)
    summary, genre, mood = work(item.text)
    return {"summary": summary, "genre": ",".join(genre), "mood": mood}


# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8000)