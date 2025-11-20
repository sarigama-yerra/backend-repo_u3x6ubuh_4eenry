import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Literal, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from jose import jwt, JWTError
from bson import ObjectId

from database import db, create_document, get_documents
from schemas import User, Wallet, Match, Raise, Move, Transaction, ModerationFlag, QueueEntry

API_JWT_SECRET = os.getenv("API_JWT_SECRET", "dev-secret-change")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

app = FastAPI(title="StakeChess API (FastAPI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utilities

def _oid(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")


def _collection(name: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    return db[name]


def create_jwt(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=TOKEN_EXPIRE_MINUTES)).timestamp()),
        "iss": "stakechess-api",
    }
    return jwt.encode(payload, API_JWT_SECRET, algorithm=ALGORITHM)


class TokenData(BaseModel):
    user_id: str


async def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, API_JWT_SECRET, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = _collection("user").find_one({"_id": _oid(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Auth and onboarding

class ExchangeBody(BaseModel):
    email: str
    name: Optional[str] = None
    image: Optional[str] = None


@app.post("/auth/exchange")
def auth_exchange(body: ExchangeBody):
    users = _collection("user")
    wallets = _collection("wallet")

    user = users.find_one({"email": body.email})
    created = False
    if not user:
        user_doc = User(email=body.email, name=body.name, image=body.image).model_dump()
        user_id = users.insert_one({**user_doc, "created_at": datetime.now(timezone.utc)}).inserted_id
        # Create wallet with 1000 VC
        wallets.insert_one({
            "userId": str(user_id),
            "soft_balance": 1000,
            "hard_balance": 0,
            "hold_soft": 0,
            "hold_hard": 0,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        })
        user = users.find_one({"_id": user_id})
        created = True
    token = create_jwt(str(user["_id"]))
    return {"token": token, "created": created, "user": {"id": str(user["_id"]), "email": user.get("email"), "name": user.get("name"), "image": user.get("image"), "elo": user.get("elo", 1200)}}


# Wallet

@app.get("/me/wallet")
async def get_wallet(current=Depends(get_current_user)):
    w = _collection("wallet").find_one({"userId": str(current["_id"])})
    if not w:
        raise HTTPException(status_code=404, detail="Wallet not found")
    w["id"] = str(w.pop("_id"))
    return w


class TopupBody(BaseModel):
    amount: int = Field(gt=0)


@app.post("/wallet/topup")
async def wallet_topup(body: TopupBody, current=Depends(get_current_user)):
    wallets = _collection("wallet")
    w = wallets.find_one({"userId": str(current["_id"])})
    if not w:
        raise HTTPException(status_code=404, detail="Wallet not found")
    wallets.update_one({"_id": w["_id"]}, {"$inc": {"soft_balance": body.amount}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    txn = Transaction(userId=str(current["_id"]), type="topup", currency="VC", amount=body.amount)
    _collection("transaction").insert_one({**txn.model_dump(), "created_at": datetime.now(timezone.utc)})
    return await get_wallet(current)


# Matchmaking

class QueueJoinBody(BaseModel):
    stake: int = Field(..., description="Buy-in per player", enum=[50, 100, 250, 500])
    time_control: Literal['3+2','5+0','10+5']
    private_code: Optional[str] = None


def _ensure_wallet_buffer(user_id: str, stake: int):
    w = _collection("wallet").find_one({"userId": user_id})
    if not w:
        raise HTTPException(status_code=400, detail="Wallet not found")
    required = int(stake * 1.1)
    if w.get("soft_balance", 0) - w.get("hold_soft", 0) < required:
        raise HTTPException(status_code=400, detail="Insufficient balance with buffer")


def _create_match_and_escrow(white_id: str, black_id: str, stake: int, time_control: str) -> Dict[str, Any]:
    # Escrow holds
    for uid in [white_id, black_id]:
        _collection("wallet").update_one({"userId": uid}, {"$inc": {"soft_balance": -stake, "hold_soft": stake}, "$set": {"updated_at": datetime.now(timezone.utc)}})
        _collection("transaction").insert_one({
            "userId": uid,
            "matchId": None,
            "type": "escrow_hold",
            "currency": "VC",
            "amount": stake,
            "created_at": datetime.now(timezone.utc)
        })
    pot = stake * 2
    m = Match(base_stake=stake, pot_amount=pot, time_control=time_control, status='in_progress')
    m_doc = {**m.model_dump(), "whiteId": white_id, "blackId": black_id, "fen": "startpos", "created_at": datetime.now(timezone.utc)}
    mid = _collection("match").insert_one(m_doc).inserted_id
    _collection("queueentry").delete_many({"userId": {"$in": [white_id, black_id]}})
    match = _collection("match").find_one({"_id": mid})
    match["id"] = str(match.pop("_id"))
    return match


@app.post("/queue/join")
async def queue_join(body: QueueJoinBody, current=Depends(get_current_user)):
    user_id = str(current["_id"]) 
    _ensure_wallet_buffer(user_id, body.stake)
    users = _collection("user")
    elo = users.find_one({"_id": _oid(user_id)}).get("elo", 1200)
    now = datetime.now(timezone.utc)

    # Try to find an opponent
    opp = _collection("queueentry").find_one({
        "userId": {"$ne": user_id},
        "time_control": body.time_control,
        "stake": body.stake,
        "private_code": body.private_code,
    })
    if opp:
        # Assign colors
        white_id, black_id = (user_id, opp["userId"]) if int(now.timestamp()) % 2 == 0 else (opp["userId"], user_id)
        match = _create_match_and_escrow(white_id, black_id, body.stake, body.time_control)
        return {"matched": True, "match": match}

    qe = QueueEntry(userId=user_id, elo=elo, stake=body.stake, time_control=body.time_control, private_code=body.private_code, created_at=now)
    _collection("queueentry").update_one({"userId": user_id}, {"$set": {**qe.model_dump()}}, upsert=True)
    return {"matched": False}


@app.post("/queue/leave")
async def queue_leave(current=Depends(get_current_user)):
    _collection("queueentry").delete_many({"userId": str(current["_id"])})
    return {"left": True}


# Matches and gameplay

class MoveBody(BaseModel):
    uci: Optional[str] = None
    san: Optional[str] = None


def _load_board(fen: str):
    import chess
    board = chess.Board() if fen == "startpos" else chess.Board(fen)
    return board


def _save_board(board) -> str:
    return board.fen()


def _current_color(match: Dict[str, Any]) -> Literal['white','black']:
    return match.get("turn_color", "white")


def _user_color(match: Dict[str, Any], user_id: str) -> Literal['white','black','none']:
    if match.get("whiteId") == user_id:
        return 'white'
    if match.get("blackId") == user_id:
        return 'black'
    return 'none'


def _end_match(match_id: ObjectId, winner_id: Optional[str], reason: str):
    m = _collection("match").find_one({"_id": match_id})
    if not m or m.get("status") == "ended":
        return
    # Release escrow and payout
    stake = m.get("base_stake", 0)
    pot = m.get("pot_amount", 0)
    white_id = m.get("whiteId")
    black_id = m.get("blackId")

    # Release holds
    for uid in [white_id, black_id]:
        _collection("wallet").update_one({"userId": uid}, {"$inc": {"hold_soft": -stake}, "$set": {"updated_at": datetime.now(timezone.utc)}})
        _collection("transaction").insert_one({
            "userId": uid,
            "matchId": str(m["_id"]),
            "type": "escrow_release",
            "currency": "VC",
            "amount": stake,
            "created_at": datetime.now(timezone.utc)
        })
    # Payout
    if winner_id:
        _collection("wallet").update_one({"userId": winner_id}, {"$inc": {"soft_balance": pot}, "$set": {"updated_at": datetime.now(timezone.utc)}})
        _collection("transaction").insert_one({
            "userId": winner_id,
            "matchId": str(m["_id"]),
            "type": "payout",
            "currency": "VC",
            "amount": pot,
            "created_at": datetime.now(timezone.utc)
        })
    else:
        # Draw: split pot
        for uid in [white_id, black_id]:
            _collection("wallet").update_one({"userId": uid}, {"$inc": {"soft_balance": pot // 2}, "$set": {"updated_at": datetime.now(timezone.utc)}})
            _collection("transaction").insert_one({
                "userId": uid,
                "matchId": str(m["_id"]),
                "type": "payout",
                "currency": "VC",
                "amount": pot // 2,
                "created_at": datetime.now(timezone.utc)
            })
    _collection("match").update_one({"_id": match_id}, {"$set": {"status": "ended", "ended_at": datetime.now(timezone.utc), "winnerId": winner_id, "end_reason": reason}})


@app.get("/matches/{match_id}")
async def get_match(match_id: str, current=Depends(get_current_user)):
    m = _collection("match").find_one({"_id": _oid(match_id)})
    if not m:
        raise HTTPException(status_code=404, detail="Match not found")
    m["id"] = str(m.pop("_id"))
    moves = list(_collection("move").find({"matchId": match_id}).sort("ply", 1))
    for mv in moves:
        mv["id"] = str(mv.pop("_id"))
    raises = list(_collection("raise").find({"matchId": match_id}).sort("_id", 1))
    for r in raises:
        r["id"] = str(r.pop("_id"))
    return {"match": m, "moves": moves, "raises": raises}


@app.post("/matches/{match_id}/move")
async def play_move(match_id: str, body: MoveBody, current=Depends(get_current_user)):
    m = _collection("match").find_one({"_id": _oid(match_id)})
    if not m or m.get("status") != "in_progress":
        raise HTTPException(status_code=400, detail="Match not in progress")
    # Ensure no pending raise
    if m.get("pending_raise_id"):
        # Check expiry
        r = _collection("raise").find_one({"_id": _oid(m["pending_raise_id"])})
        if r and r.get("accepted") is None:
            # still pending
            raise HTTPException(status_code=409, detail="Raise pending, respond first")
    user_id = str(current["_id"]) 
    color = _user_color(m, user_id)
    if color not in ["white", "black"]:
        raise HTTPException(status_code=403, detail="Not a participant")
    if _current_color(m) != color:
        raise HTTPException(status_code=409, detail="Not your turn")

    import chess
    board = _load_board(m.get("fen", "startpos"))
    try:
        if body.uci:
            move = chess.Move.from_uci(body.uci)
        elif body.san:
            move = board.parse_san(body.san)
        else:
            raise ValueError("Provide uci or san")
        if move not in board.legal_moves:
            raise ValueError("Illegal move")
        board.push(move)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid move: {str(e)}")

    new_fen = _save_board(board)

    ply = _collection("move").count_documents({"matchId": match_id}) + 1
    san = board.san(board.peek()) if hasattr(board, 'peek') else body.san or body.uci
    mv = Move(matchId=match_id, ply=ply, san=san, fen_after=new_fen)
    _collection("move").insert_one({**mv.model_dump(), "created_at": datetime.now(timezone.utc)})

    # Update turn
    turn_color = 'white' if board.turn and True else 'black'  # board.turn True = White to move
    turn_color = 'white' if board.turn else 'black'

    update = {"fen": new_fen, "turn_color": turn_color}

    # Check end conditions
    result_reason = None
    winner_id: Optional[str] = None
    if board.is_checkmate():
        result_reason = "checkmate"
        winner_id = m.get("whiteId") if turn_color == 'black' else m.get("blackId")
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        result_reason = "draw"

    _collection("match").update_one({"_id": m["_id"]}, {"$set": update})

    if result_reason == "checkmate":
        _end_match(m["_id"], winner_id, result_reason)
    elif result_reason == "draw":
        _end_match(m["_id"], None, result_reason)

    return await get_match(match_id, current)


class RaiseBody(BaseModel):
    percent: Literal[25, 50, 100]


@app.post("/matches/{match_id}/raise")
async def propose_raise(match_id: str, body: RaiseBody, current=Depends(get_current_user)):
    m = _collection("match").find_one({"_id": _oid(match_id)})
    if not m or m.get("status") != "in_progress":
        raise HTTPException(status_code=400, detail="Match not in progress")
    if m.get("pending_raise_id"):
        raise HTTPException(status_code=409, detail="Another raise pending")
    user_id = str(current["_id"]) 
    color = _user_color(m, user_id)
    if color not in ["white","black"]:
        raise HTTPException(status_code=403, detail="Not a participant")
    # Only token holder can raise, and only at start of their turn before moving
    if m.get("raise_token_holder") != color or m.get("turn_color") != color:
        raise HTTPException(status_code=409, detail="You cannot raise now")
    # Caps
    if m.get("raise_count", 0) >= 6:
        raise HTTPException(status_code=400, detail="Max raises reached")
    max_pot = m.get("base_stake", 0) * 16
    current_pot = m.get("pot_amount", 0)
    delta = int(current_pot * (body.percent / 100.0))
    new_pot = current_pot + delta
    if new_pot > max_pot:
        raise HTTPException(status_code=400, detail="Pot ceiling reached")

    r = Raise(matchId=match_id, proposerId=user_id, amount_delta=delta, pot_after=new_pot)
    rid = _collection("raise").insert_one({**r.model_dump(), "created_at": datetime.now(timezone.utc), "expires_at": datetime.now(timezone.utc) + timedelta(seconds=25)}).inserted_id
    _collection("match").update_one({"_id": m["_id"]}, {"$set": {"pending_raise_id": str(rid)}})

    return {"raiseId": str(rid), "pot_after": new_pot, "delta": delta}


class RaiseRespondBody(BaseModel):
    accept: bool


@app.post("/matches/{match_id}/raise/{raise_id}/respond")
async def respond_raise(match_id: str, raise_id: str, body: RaiseRespondBody, current=Depends(get_current_user)):
    m = _collection("match").find_one({"_id": _oid(match_id)})
    if not m or m.get("status") != "in_progress":
        raise HTTPException(status_code=400, detail="Match not in progress")
    if m.get("pending_raise_id") != raise_id:
        raise HTTPException(status_code=400, detail="No such pending raise")
    r = _collection("raise").find_one({"_id": _oid(raise_id)})
    if not r or r.get("accepted") is not None:
        raise HTTPException(status_code=400, detail="Raise not pending")

    # Only responder (opponent) can respond
    user_id = str(current["_id"]) 
    if user_id == r.get("proposerId"):
        raise HTTPException(status_code=403, detail="Proposer cannot respond")

    # TTL check
    if datetime.now(timezone.utc) > r.get("expires_at"):
        # timeout = refusal
        body.accept = False

    if body.accept:
        # Accept: increase pot, raise_count++, alternate token holder
        _collection("match").update_one({"_id": m["_id"]}, {"$inc": {"pot_amount": r.get("amount_delta", 0), "raise_count": 1}})
        new_holder = 'white' if m.get("raise_token_holder") == 'black' else 'black'
        _collection("match").update_one({"_id": m["_id"]}, {"$set": {"raise_token_holder": new_holder, "pending_raise_id": None}})
        _collection("raise").update_one({"_id": r["_id"]}, {"$set": {"accepted": True, "responderId": user_id}})
    else:
        # Refuse: immediate loss for responder
        loser_id = user_id
        winner_id = m.get("whiteId") if loser_id == m.get("blackId") else m.get("blackId")
        _collection("raise").update_one({"_id": r["_id"]}, {"$set": {"accepted": False, "responderId": user_id, "refusal_loss_player_id": loser_id}})
        _collection("match").update_one({"_id": m["_id"]}, {"$set": {"pending_raise_id": None}})
        _end_match(m["_id"], winner_id, "raise_refused")

    return await get_match(match_id, current)


@app.post("/matches/{match_id}/resign")
async def resign(match_id: str, current=Depends(get_current_user)):
    m = _collection("match").find_one({"_id": _oid(match_id)})
    if not m or m.get("status") != "in_progress":
        raise HTTPException(status_code=400, detail="Match not in progress")
    user_id = str(current["_id"]) 
    if user_id not in [m.get("whiteId"), m.get("blackId")]:
        raise HTTPException(status_code=403, detail="Not a participant")
    winner_id = m.get("whiteId") if user_id == m.get("blackId") else m.get("blackId")
    _end_match(m["_id"], winner_id, "resign")
    return await get_match(match_id, current)


@app.post("/matches/{match_id}/offer-draw")
async def offer_draw(match_id: str, current=Depends(get_current_user)):
    # For MVP, we don't store offers; client should call accept to finalize
    return {"offered": True}


@app.post("/matches/{match_id}/accept-draw")
async def accept_draw(match_id: str, current=Depends(get_current_user)):
    m = _collection("match").find_one({"_id": _oid(match_id)})
    if not m or m.get("status") != "in_progress":
        raise HTTPException(status_code=400, detail="Match not in progress")
    _end_match(m["_id"], None, "draw_agreed")
    return await get_match(match_id, current)


# Leaderboard and profiles

@app.get("/leaderboard")
async def leaderboard():
    # Simple global leaderboard by ELO and by wallet VC
    users = list(_collection("user").find().limit(100))
    for u in users:
        u["id"] = str(u.pop("_id"))
        w = _collection("wallet").find_one({"userId": u["id"]})
        u["vc"] = w.get("soft_balance", 0) if w else 0
    users.sort(key=lambda x: (-x.get("elo", 1200), -x.get("vc", 0)))
    return {"global": users[:50]}


@app.get("/profile/{user_id}")
async def profile(user_id: str):
    u = _collection("user").find_one({"_id": _oid(user_id)})
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    w = _collection("wallet").find_one({"userId": user_id})
    u["id"] = str(u.pop("_id"))
    return {"user": u, "wallet": w}


# Health and database test

@app.get("/")
def read_root():
    return {"message": "StakeChess API ready"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:20]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
