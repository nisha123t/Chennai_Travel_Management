
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from router import find_route, get_stops

app = FastAPI(title="Chennai Route Planner API")

# Allow requests from the React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RouteRequest(BaseModel):
    start: str
    destination: str


@app.get("/")
def home():
    return {"message": "Chennai Route Planner API is running"}


@app.get("/stops")
def list_stops():
    return get_stops()



@app.post("/route")
def calculate_route(request: RouteRequest, preference: str = "fastest"):
    try:
        result = find_route(
            request.start,
            request.destination,
            preference
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No route found between these locations"
        )

    return result
