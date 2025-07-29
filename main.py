from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import load_model_config, get_llm_instance
import uvicorn
app = FastAPI()
model_map = load_model_config()

class InferenceRequest(BaseModel):
    agent: str
    prompt: str

@app.post("/invoke")
async def invoke_model(request: InferenceRequest):
    prompt = request.prompt
    agent_name = request.agent
    print(f"Received request for agent: {agent_name} with prompt: {prompt}")
    config = model_map.get(agent_name)
    print(f"Config for agent {agent_name}: {config}")
    if not config:
        raise HTTPException(status_code=404, detail="Agent config not found")

    llm = get_llm_instance(config)
    try:
        output = await llm.ainvoke(request.prompt)
        print(f"Model response: {output.content.strip()}")
        return {"response": output.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/{agent}")
async def get_model_config(agent: str):
    config = model_map.get(agent)
    if not config:
        raise HTTPException(status_code=404, detail="Agent config not fxound")
    return config
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
