from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# ─────────────────────────────────────────────────────────
# WHAT THIS FILE IS
#
# Middleware runs before every single request
# reaches any router or endpoint.
#
# Think of it as a security guard at the entrance.
# Every person entering must show their ID.
# Guard checks the ID.
# Valid ID → person enters.
# No ID    → person turned away immediately.
#
# In our case:
# Valid user_id in header → request continues
# No user_id in header   → 401 returned immediately
#
# IMPORTANT:
# We do NOT verify or decode JWT here.
# Frontend team handles JWT completely.
# Frontend decodes the JWT on their side
# and sends us the user_id directly in a header.
# We just read that header and trust it.
# ─────────────────────────────────────────────────────────


# These routes do not need user_id
# They are public and open to anyone
PUBLIC_ROUTES = [
    "/health",
    "/docs",        # FastAPI auto-generated docs page
    "/openapi.json" # FastAPI auto-generated schema
]


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Runs before every request.

    Reads X-User-ID from request headers.
    Attaches it to request.state.user_id
    so every endpoint can access it easily.

    If X-User-ID is missing on a protected route
    returns 401 Unauthorized immediately.
    Request never reaches the router.
    """

    async def dispatch(self, request: Request, call_next):
        # Check if this is a public route
        # Public routes do not need user_id
        if request.url.path in PUBLIC_ROUTES:
            return await call_next(request)

        # Read the user_id from request header
        # Frontend team sends this after decoding JWT
        # Header name: X-User-ID
        # Example: X-User-ID: user_abc123
        user_id = request.headers.get("X-User-ID")

        # If user_id is missing return 401 immediately
        # Request goes no further
        if not user_id:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "X-User-ID header is required"
                }
            )

        # Attach user_id to request state
        # Now every endpoint can access it like this:
        # request.state.user_id
        # No need to read the header again anywhere
        request.state.user_id = user_id

        # Continue to the actual endpoint
        response = await call_next(request)
        return response