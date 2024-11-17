from fastapi import APIRouter

from app.api.endpoints.root import router as root_router
from app.api.endpoints.bike_sharing import router as bike_sharing_router

api_router = APIRouter()

api_router.include_router(root_router, tags=["root"])
api_router.include_router(bike_sharing_router, prefix="/bike_sharing", tags=["bike_sharing"])
