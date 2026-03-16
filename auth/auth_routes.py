from fastapi import APIRouter, HTTPException, status
from auth.auth_models import RegisterRequest, LoginRequest, TokenResponse
from auth.auth_utils import create_user, authenticate_user, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    if len(req.username.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    success = create_user(req.username.strip(), req.password, req.email.strip())
    if not success:
        raise HTTPException(status_code=409, detail="Username already exists")

    token = create_access_token({"sub": req.username.strip()})
    return TokenResponse(access_token=token, username=req.username.strip())


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    user = authenticate_user(req.username.strip(), req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = create_access_token({"sub": user["username"]})
    return TokenResponse(access_token=token, username=user["username"])