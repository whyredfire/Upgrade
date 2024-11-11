from fastapi import FastAPI

from model.main import get_hybrid_recommendations
from dependencies import if_exists

app = FastAPI()


@app.post("/get_recommendations")
async def get_recommendations(movie: str):

    search_result = await if_exists(user=0, movie=movie)
    if search_result:
        print(len(search_result))
        movie_index, movie_name = search_result
    else:
        return {"response": "movie not found"}

    result = get_hybrid_recommendations(movie_index)
    return result
