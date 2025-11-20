from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from datetime import datetime

# Core domain schemas (one class = one collection)

class User(BaseModel):
    email: str
    name: Optional[str] = None
    image: Optional[str] = None
    elo: int = 1200
    country: Optional[str] = None
    flags: Optional[List[str]] = None

class Wallet(BaseModel):
    userId: str
    soft_balance: int = 0
    hard_balance: int = 0
    hold_soft: int = 0
    hold_hard: int = 0

class Match(BaseModel):
    status: Literal['pending','in_progress','ended'] = 'pending'
    whiteId: Optional[str] = None
    blackId: Optional[str] = None
    pot_amount: int = 0
    base_stake: int
    raise_count: int = 0
    turn_color: Literal['white','black'] = 'white'
    raise_token_holder: Literal['white','black'] = 'white'
    time_control: str  # e.g., "3+2", "5+0", "10+5"
    fen: str = 'startpos'
    created_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    winnerId: Optional[str] = None
    end_reason: Optional[str] = None
    pending_raise_id: Optional[str] = None

class Raise(BaseModel):
    matchId: str
    proposerId: str
    amount_delta: int  # absolute amount to add to pot
    pot_after: int
    accepted: Optional[bool] = None
    responderId: Optional[str] = None
    refusal_loss_player_id: Optional[str] = None

class Move(BaseModel):
    matchId: str
    ply: int
    san: str
    fen_after: str
    clock_white_ms: Optional[int] = None
    clock_black_ms: Optional[int] = None

class Transaction(BaseModel):
    userId: str
    matchId: Optional[str] = None
    type: Literal['escrow_hold','escrow_release','payout','topup']
    currency: Literal['VC','HC'] = 'VC'
    amount: int
    idempotency_key: Optional[str] = None

class ModerationFlag(BaseModel):
    matchId: Optional[str] = None
    userId: Optional[str] = None
    reason: str
    status: Literal['open','reviewing','closed'] = 'open'

class QueueEntry(BaseModel):
    userId: str
    elo: int
    stake: int
    time_control: str
    private_code: Optional[str] = None
    created_at: Optional[datetime] = None
