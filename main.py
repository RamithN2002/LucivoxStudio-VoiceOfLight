from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload_routes import router as upload_router
from routes.chat_routes import router as chat_router
from auth.auth_routes import router as auth_router

app = FastAPI(title="Lucivox Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

app.include_router(chat_router)
app.include_router(upload_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
