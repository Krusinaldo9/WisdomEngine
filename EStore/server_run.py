import uvicorn

if __name__ == "__main__":

    uvicorn.run("embedding_server:app", host="localhost", port=5000, log_level="info", reload=False, debug=False, workers=1)
