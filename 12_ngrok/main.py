from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root_index():
    return {"messages": "저는 김동영입니다."}

@app.get("/name")
def name_response(name: str):
    return JSONResponse({"messages": f"저는 {name}입니다."})

# POST 요청에서 사용할 데이터 모델 정의
class NameRequest(BaseModel):
    name: str

@app.post("/name")
def post_name_response(request: NameRequest):
    return JSONResponse({"messages": f"POST 요청으로 받은 이름은 {request.name}입니다."})