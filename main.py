from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import load_model_config, get_llm_instance, extract_json
import uvicorn
import logging
app = FastAPI()
model_map = load_model_config()
from logging_config import setup_logging

setup_logging(log_file='model_selector.log')
logger = logging.getLogger(__name__)

class InferenceRequest(BaseModel):
    purpose: str
    prompt: str

@app.post("/invoke")
async def invoke_model(request: InferenceRequest):
    prompt = request.prompt
    purpose = request.purpose
    logger.info(f"Received request for purpose: {purpose} with prompt: {prompt}")
    # print(f"Received request for purpose: {purpose} with prompt: {prompt}")
    config = model_map.get(purpose)
    logger.info(f"Config for agent {purpose}: {config}")
    # print(f"Config for agent {purpose}: {config}")
    if not config:
        raise HTTPException(status_code=404, detail="Agent config not found")

    llm = get_llm_instance(config)
    try:
        raw_output = await llm.ainvoke(request.prompt)
        # print(f"Raw model output: {raw_output}")
        output= extract_json(raw_output.content)
        # print(f"Extracted JSON: {output}")
        # print(f"Model response: {output}")
        logger.info(f"Model response: {output}")
        return {"response": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7001)
