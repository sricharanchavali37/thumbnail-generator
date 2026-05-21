from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.database.mongodb import connect_to_mongodb, close_mongodb_connection
from backend.middleware.auth import AuthMiddleware
from backend.routers.videos import router as videos_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application starting...")
    await connect_to_mongodb()
    print("Application ready.")
    yield
    print("Application shutting down...")
    await close_mongodb_connection()
    print("Goodbye.")


app = FastAPI(
    title="Thumbnail Generator API",
    description="Automatic video thumbnail generation service",
    version="1.0.0",
    lifespan=lifespan
)

# Register auth middleware
# Runs before every single request
app.add_middleware(AuthMiddleware)

# Register video routes
# All endpoints now available under /videos/
app.include_router(videos_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}