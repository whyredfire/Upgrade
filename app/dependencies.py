import csv


async def if_exists(movie: str):
    with open("app/model/Final_data.csv", "r") as movies:
        reader = csv.DictReader(movies)

        for index, row in enumerate(reader):
            if row["title"] == movie:
                return index, row

    return False
