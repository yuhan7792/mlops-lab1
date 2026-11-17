import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()
model = joblib.load('model.pkl')

class Req(BaseModel):
    instances: list[list[float]] = Field(min_length=1)

@app.post('/predict')
def predict(req: Req):
    if any(len(row) != 4 for row in req.instances):
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail='Each instance must have four features')
    return {'predictions': model.predict(req.instances).tolist()}

@app.get('/healthz')
def healthz():
    return {'ok': True}
