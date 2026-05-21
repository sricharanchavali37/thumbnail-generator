from motor.motor_asyncio import AsyncIOMotorClient
from backend.config import settings

# ─────────────────────────────────────────────
# What is happening here:
#
# We create ONE client object that holds
# the connection to MongoDB.
#
# This client is created once when the
# application starts and reused forever.
#
# motor_client  = the connection itself
# database      = which database inside MongoDB
#                 we are working with
# videos_collection = which collection inside
#                     that database we use
#                     (like a table in SQL)
# ─────────────────────────────────────────────

motor_client: AsyncIOMotorClient = None
database = None
videos_collection = None


async def connect_to_mongodb():
    """
    Called once when FastAPI application starts.

    Establishes connection to MongoDB.
    Sets up the database and collection references
    that the rest of the application will use.
    """
    global motor_client, database, videos_collection

    # Create the connection using the URL from .env
    # For example: mongodb://localhost:27017
    motor_client = AsyncIOMotorClient(settings.MONGODB_URL)

    # Select which database to use
    # This is the MONGODB_DB_NAME from .env
    # Example: thumbnail_generator
    database = motor_client[settings.MONGODB_DB_NAME]

    # Select which collection to use
    # A collection is like a table in SQL
    # All video records go into "videos" collection
    videos_collection = database["videos"]

    # Create indexes so queries run fast
    # Without indexes MongoDB reads every document
    # With indexes MongoDB jumps directly to what we need

    # Index 1: user_id + uploaded_at
    # Used for history queries:
    # "Give me all videos for this user, newest first"
    await videos_collection.create_index(
        [("user_id", 1), ("uploaded_at", -1)]
    )

    # Index 2: job_id
    # Used for polling queries:
    # "Give me the status of this specific job"
    await videos_collection.create_index("job_id", unique=True)

    print(f"Connected to MongoDB: {settings.MONGODB_DB_NAME}")


async def close_mongodb_connection():
    """
    Called once when FastAPI application shuts down.

    Cleanly closes the MongoDB connection.
    Always close connections properly.
    Leaving them open wastes resources.
    """
    global motor_client

    if motor_client:
        motor_client.close()
        print("MongoDB connection closed")


def get_videos_collection():
    """
    Returns the videos collection.

    Every file that needs to read or write
    video data imports and calls this function.

    This way nobody creates their own connection.
    Everyone uses the same one.
    """
    return videos_collection