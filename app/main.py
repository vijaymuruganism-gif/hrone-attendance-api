"""
Employee Attendance & Analytics API
Implements OpenAPI 3.0.3 contract and business rules R1 - R10.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date as dt_date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path, status
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

load_dotenv()

# --------------------------------------------------------------------------- #
# MongoDB Connection & Configuration
# --------------------------------------------------------------------------- #
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

client: MongoClient = MongoClient(MONGO_URI, tz_aware=True)
db = client[MONGO_DB]

IST = timezone(timedelta(hours=5, minutes=30))


def init_indexes():
    """Create all required indexes at startup idempotently."""
    # employees indexes
    db.employees.create_index([("emp_code", ASCENDING)], unique=True)
    db.employees.create_index([("department", ASCENDING), ("emp_code", ASCENDING)])
    db.employees.create_index([("joined_on", ASCENDING)])

    # attendance_logs indexes
    # Natural key: exactly one record per (emp_code, date)
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)], unique=True
    )
    # Covering index for list_attendance: date DESC, emp_code ASC
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)]
    )
    # Fast punch-out lookup: emp_code + punch_in DESC
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("punch_in", DESCENDING)]
    )
    # Fast filtering for analytics
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING), ("status", ASCENDING)])
    db.attendance_logs.create_index([("date", ASCENDING), ("status", ASCENDING)])


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_indexes()
    yield


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Date & Time Utilities (R1, R2, R3, R4, R5, R8)
# --------------------------------------------------------------------------- #
def round_half_up(value: float, decimals: int = 2) -> float:
    """Half-up rounding per R4, R5, R8 (not banker's rounding)."""
    fmt = "0." + "0" * decimals if decimals > 0 else "0"
    d = Decimal(str(value)).quantize(Decimal(fmt), rounding=ROUND_HALF_UP)
    return float(d)


def ms_to_datetime(ms: int) -> datetime:
    """Convert epoch milliseconds to a UTC datetime truncated to whole seconds (R1)."""
    truncated_sec = ms // 1000
    return datetime.fromtimestamp(truncated_sec, tz=timezone.utc)


def datetime_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Convert datetime to epoch milliseconds."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1000


def get_current_utc_truncated() -> datetime:
    """Current UTC datetime truncated to whole seconds."""
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=0)


def compute_attendance_date(punch_in_utc: datetime, shift_start: str, shift_end: str) -> str:
    """
    R1: Attendance date in IST.
    Overnight shift: shift_end <= shift_start.
    A punch-in whose IST time is earlier than shift_end belongs to the shift that started the previous day.
    """
    punch_in_ist = punch_in_utc.astimezone(IST)
    is_overnight = shift_end <= shift_start

    if is_overnight:
        end_h, end_m = map(int, shift_end.split(":"))
        shift_end_time = time(end_h, end_m)
        if punch_in_ist.time() < shift_end_time:
            att_date = (punch_in_ist.date() - timedelta(days=1)).isoformat()
            return att_date

    return punch_in_ist.date().isoformat()


def compute_late_minutes(punch_in_utc: datetime, shift_start: str, attendance_date_str: str) -> int:
    """
    R2: Grace period 10 minutes.
    Punch-in is late only if strictly > 10:00 after shift_start.
    When late: floor of whole minutes elapsed since shift_start.
    """
    punch_in_ist = punch_in_utc.astimezone(IST)
    d = dt_date.fromisoformat(attendance_date_str)
    h, m = map(int, shift_start.split(":"))
    shift_start_dt = datetime(d.year, d.month, d.day, h, m, 0, tzinfo=IST)

    elapsed_seconds = (punch_in_ist - shift_start_dt).total_seconds()
    if elapsed_seconds > 600:  # strictly > 10 minutes (600s)
        return int(elapsed_seconds // 60)
    return 0


def compute_overtime_minutes(punch_out_utc: datetime, shift_start: str, shift_end: str, attendance_date_str: str) -> int:
    """
    R3: Overtime = whole minutes (floor) between shift_end and punch_out.
    Counted only when >= 30; otherwise 0.
    For overnight shifts, shift_end falls on attendance_date + 1 day.
    """
    punch_out_ist = punch_out_utc.astimezone(IST)
    d = dt_date.fromisoformat(attendance_date_str)
    start_h, start_m = map(int, shift_start.split(":"))
    end_h, end_m = map(int, shift_end.split(":"))

    if shift_end <= shift_start:
        end_date = d + timedelta(days=1)
    else:
        end_date = d

    shift_end_dt = datetime(end_date.year, end_date.month, end_date.day, end_h, end_m, 0, tzinfo=IST)
    elapsed_seconds = (punch_out_ist - shift_end_dt).total_seconds()
    if elapsed_seconds >= 1800:  # >= 30 minutes
        return int(elapsed_seconds // 60)
    return 0


def compute_work_hours(punch_in_utc: datetime, punch_out_utc: datetime) -> float:
    """R4: (punch_out - punch_in) in seconds / 3600, rounded 2 decimals half-up."""
    diff_sec = (punch_out_utc - punch_in_utc).total_seconds()
    hours = diff_sec / 3600.0
    return round_half_up(hours, 2)


def compute_half_day(work_hours: Optional[float]) -> bool:
    """R5: half_day = true when rounded work_hours < 4.50."""
    if work_hours is None:
        return False
    return work_hours < 4.50


def count_working_days(month_str: str, joined_on_str: str) -> int:
    """R7: Count Mon-Fri days in month starting from max(start_of_month, joined_on)."""
    year, month = map(int, month_str.split("-"))
    start_of_month = dt_date(year, month, 1)
    if month == 12:
        end_of_month = dt_date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_of_month = dt_date(year, month + 1, 1) - timedelta(days=1)

    joined_date = dt_date.fromisoformat(joined_on_str)
    start_date = max(start_of_month, joined_date)
    if start_date > end_of_month:
        return 0

    working_days = 0
    curr = start_date
    while curr <= end_of_month:
        if curr.weekday() < 5:  # Mon-Fri
            working_days += 1
        curr += timedelta(days=1)
    return working_days


# --------------------------------------------------------------------------- #
# Pydantic Schemas
# --------------------------------------------------------------------------- #
class EmployeeCreate(BaseModel):
    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(default=None, ge=100000000000, le=4102444800000)
    status: str = Field(default="PRESENT", pattern=r"^(PRESENT|WFH|ON_DUTY)$")


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(default=None, ge=100000000000, le=4102444800000)


class RegularizeRequest(BaseModel):
    status: Optional[str] = Field(default=None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$")
    punch_in: Optional[int] = Field(default=None, ge=100000000000, le=4102444800000)
    punch_out: Optional[int] = Field(default=None, ge=100000000000, le=4102444800000)
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# Serialization Helpers
# --------------------------------------------------------------------------- #
def serialize_employee(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": datetime_to_ms(doc.get("created_at")),
    }


def serialize_history_entry(entry: dict) -> dict:
    changes = {}
    for k, v in entry.get("changes", {}).items():
        from_val = v.get("from")
        to_val = v.get("to")
        if isinstance(from_val, datetime):
            from_val = datetime_to_ms(from_val)
        if isinstance(to_val, datetime):
            to_val = datetime_to_ms(to_val)
        changes[k] = {"from": from_val, "to": to_val}

    return {
        "at": datetime_to_ms(entry.get("at")),
        "by": entry.get("by"),
        "reason": entry.get("reason"),
        "changes": changes,
    }


def serialize_attendance(doc: dict) -> dict:
    history = [serialize_history_entry(h) for h in doc.get("history", [])]
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": datetime_to_ms(doc.get("punch_in")),
        "punch_out": datetime_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["system"])
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unavailable: {str(e)}",
        )


@app.post("/employees", status_code=status.HTTP_201_CREATED, tags=["employees"])
def create_employee(body: EmployeeCreate):
    if body.shift_start == body.shift_end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="shift_start must differ from shift_end",
        )

    doc = body.model_dump()
    doc["created_at"] = get_current_utc_truncated()

    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="emp_code already exists",
        )

    return serialize_employee(doc)


@app.get("/employees", tags=["employees"])
def list_employees(
    department: Optional[str] = Query(None, description="Exact, case-sensitive match."),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {}
    if department:
        q["department"] = department

    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    cursor = db.employees.find(q).sort("emp_code", ASCENDING).skip(skip).limit(page_size)
    items = [serialize_employee(doc) for doc in cursor]

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post("/attendance/punch-in", status_code=status.HTTP_201_CREATED, tags=["attendance"])
def punch_in(body: PunchInRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown employee")

    punch_in_dt = (
        ms_to_datetime(body.punched_at)
        if body.punched_at is not None
        else get_current_utc_truncated()
    )

    attendance_date = compute_attendance_date(
        punch_in_dt, emp["shift_start"], emp["shift_end"]
    )
    late_minutes = compute_late_minutes(
        punch_in_dt, emp["shift_start"], attendance_date
    )

    doc = {
        "emp_code": body.emp_code,
        "date": attendance_date,
        "status": body.status,
        "punch_in": punch_in_dt,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_minutes,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="already punched in for this date",
        )

    return serialize_attendance(doc)


@app.post("/attendance/punch-out", status_code=status.HTTP_200_OK, tags=["attendance"])
def punch_out(body: PunchOutRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown employee")

    punch_out_dt = (
        ms_to_datetime(body.punched_at)
        if body.punched_at is not None
        else get_current_utc_truncated()
    )

    log = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": punch_out_dt}},
        sort=[("punch_in", DESCENDING)],
    )

    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no punch-in found"
        )

    if log.get("punch_out") is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="record is already punched out",
        )

    punch_in_dt = log["punch_in"]
    if punch_in_dt.tzinfo is None:
        punch_in_dt = punch_in_dt.replace(tzinfo=timezone.utc)

    if punch_out_dt <= punch_in_dt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="punched_at must be strictly greater than punch_in",
        )

    if (punch_out_dt - punch_in_dt).total_seconds() > 86400:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="punch_out cannot be more than 24 hours after punch_in",
        )

    work_hours = compute_work_hours(punch_in_dt, punch_out_dt)
    overtime = compute_overtime_minutes(
        punch_out_dt, emp["shift_start"], emp["shift_end"], log["date"]
    )
    half_day = compute_half_day(work_hours)

    updated = db.attendance_logs.find_one_and_update(
        {"_id": log["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": punch_out_dt,
                "work_hours": work_hours,
                "overtime_minutes": overtime,
                "half_day": half_day,
            }
        },
        return_document=True,
    )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="record was already punched out concurrently",
        )

    return serialize_attendance(updated)


@app.get("/attendance", tags=["attendance"])
def list_attendance(
    emp_code: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    status: Optional[str] = Query(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="date_from must be <= date_to",
        )

    q: Dict[str, Any] = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    cursor = (
        db.attendance_logs.find(q)
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )

    items = [serialize_attendance(doc) for doc in cursor]

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.patch("/attendance/{emp_code}/{date}", tags=["attendance"])
def regularize_attendance(
    emp_code: str = Path(...),
    date: str = Path(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
    body: RegularizeRequest = ...,
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown employee")

    log = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no attendance record found for this date",
        )

    new_status = body.status if body.status is not None else log["status"]
    is_absent_leave = new_status in ("ABSENT", "LEAVE")

    if is_absent_leave and (body.punch_in is not None or body.punch_out is not None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ABSENT or LEAVE status cannot have punch times",
        )

    if is_absent_leave:
        final_punch_in = None
        final_punch_out = None
    else:
        final_punch_in = (
            ms_to_datetime(body.punch_in)
            if body.punch_in is not None
            else log.get("punch_in")
        )
        final_punch_out = (
            ms_to_datetime(body.punch_out)
            if body.punch_out is not None
            else log.get("punch_out")
        )

    if not is_absent_leave and final_punch_in is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="presence status requires a punch_in",
        )

    if final_punch_in is not None:
        computed_date = compute_attendance_date(
            final_punch_in, emp["shift_start"], emp["shift_end"]
        )
        if computed_date != date:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"punch_in belongs to attendance date {computed_date}, not {date}",
            )

    if final_punch_out is not None:
        if final_punch_in is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="punch_out cannot exist without punch_in",
            )
        if final_punch_in.tzinfo is None:
            final_punch_in = final_punch_in.replace(tzinfo=timezone.utc)
        if final_punch_out.tzinfo is None:
            final_punch_out = final_punch_out.replace(tzinfo=timezone.utc)

        if final_punch_out <= final_punch_in:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="punch_out must be strictly greater than punch_in",
            )
        if (final_punch_out - final_punch_in).total_seconds() > 86400:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="punch_out cannot be more than 24 hours after punch_in",
            )

    if is_absent_leave:
        final_late_minutes = 0
        final_work_hours = None
        final_overtime_minutes = 0
        final_half_day = False
    else:
        final_late_minutes = compute_late_minutes(final_punch_in, emp["shift_start"], date)
        if final_punch_out is not None:
            final_work_hours = compute_work_hours(final_punch_in, final_punch_out)
            final_overtime_minutes = compute_overtime_minutes(
                final_punch_out, emp["shift_start"], emp["shift_end"], date
            )
            final_half_day = compute_half_day(final_work_hours)
        else:
            final_work_hours = None
            final_overtime_minutes = 0
            final_half_day = False

    changes: Dict[str, dict] = {}
    old_status = log.get("status")
    old_punch_in = log.get("punch_in")
    old_punch_out = log.get("punch_out")
    old_work_hours = log.get("work_hours")
    old_late_minutes = log.get("late_minutes", 0)
    old_overtime_minutes = log.get("overtime_minutes", 0)
    old_half_day = log.get("half_day", False)

    if new_status != old_status:
        changes["status"] = {"from": old_status, "to": new_status}

    def dt_equal(a: Optional[datetime], b: Optional[datetime]) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return int(a.timestamp()) == int(b.timestamp())

    if not dt_equal(old_punch_in, final_punch_in):
        changes["punch_in"] = {"from": old_punch_in, "to": final_punch_in}
    if not dt_equal(old_punch_out, final_punch_out):
        changes["punch_out"] = {"from": old_punch_out, "to": final_punch_out}
    if old_work_hours != final_work_hours:
        changes["work_hours"] = {"from": old_work_hours, "to": final_work_hours}
    if old_late_minutes != final_late_minutes:
        changes["late_minutes"] = {"from": old_late_minutes, "to": final_late_minutes}
    if old_overtime_minutes != final_overtime_minutes:
        changes["overtime_minutes"] = {"from": old_overtime_minutes, "to": final_overtime_minutes}
    if old_half_day != final_half_day:
        changes["half_day"] = {"from": old_half_day, "to": final_half_day}

    if not changes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="request changes nothing",
        )

    history_entry = {
        "at": get_current_utc_truncated(),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    current_history_len = len(log.get("history", []))
    update_result = db.attendance_logs.find_one_and_update(
        {
            "_id": log["_id"],
            "history": {"$size": current_history_len},
        },
        {
            "$set": {
                "status": new_status,
                "punch_in": final_punch_in,
                "punch_out": final_punch_out,
                "work_hours": final_work_hours,
                "late_minutes": final_late_minutes,
                "overtime_minutes": final_overtime_minutes,
                "half_day": final_half_day,
            },
            "$push": {"history": history_entry},
        },
        return_document=True,
    )

    if not update_result:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="concurrent modification conflict",
        )

    return serialize_attendance(update_result)


# --------------------------------------------------------------------------- #
# Analytics Pipelines
# --------------------------------------------------------------------------- #
def build_employee_monthly_pipeline(emp_code: str, month: str) -> list:
    return [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$regex": f"^{month}"},
            }
        },
        {
            "$addFields": {
                "dayOfWeek": {
                    "$isoDayOfWeek": {"$dateFromString": {"dateString": "$date"}}
                }
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$lte": ["$dayOfWeek", 5]},
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                ]
                            },
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": "$late_minutes"},
                "total_overtime_minutes": {"$sum": "$overtime_minutes"},
            }
        },
    ]


@app.get("/analytics/employees/{emp_code}/monthly", tags=["analytics"])
def employee_monthly(
    emp_code: str = Path(...),
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown employee")

    working_days = count_working_days(month, emp["joined_on"])
    pipeline = build_employee_monthly_pipeline(emp_code, month)
    res = list(db.attendance_logs.aggregate(pipeline))

    if res:
        data = res[0]
        present_days = data["present_days"]
        leave_days = data["leave_days"]
        late_count = data["late_count"]
        total_late = data["total_late_minutes"]
        total_overtime = data["total_overtime_minutes"]
    else:
        present_days = 0.0
        leave_days = 0
        late_count = 0
        total_late = 0
        total_overtime = 0

    attendance_pct = (
        round_half_up((present_days / working_days) * 100.0, 2)
        if working_days > 0
        else None
    )

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late,
        "total_overtime_minutes": total_overtime,
        "attendance_pct": attendance_pct,
    }


def build_department_summary_pipeline(month: str, department: Optional[str] = None) -> list:
    year, m = map(int, month.split("-"))
    if m == 12:
        end_date_str = f"{year}-12-31"
    else:
        last_day = (dt_date(year, m + 1, 1) - timedelta(days=1)).day
        end_date_str = f"{year}-{m:02d}-{last_day:02d}"

    emp_match = {"joined_on": {"$lte": end_date_str}}
    if department:
        emp_match["department"] = department

    return [
        {"$match": emp_match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"ecode": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$ecode"]},
                                    {"$regexMatch": {"input": "$date", "regex": f"^{month}"}},
                                ]
                            }
                        }
                    },
                    {
                        "$addFields": {
                            "dayOfWeek": {
                                "$isoDayOfWeek": {"$dateFromString": {"dateString": "$date"}}
                            }
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {
            "$group": {
                "_id": "$department",
                "employees_set": {"$addToSet": "$emp_code"},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ifNull": ["$logs", False]},
                                    {"$lte": ["$logs.dayOfWeek", 5]},
                                    {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                ]
                            },
                            {"$cond": [{"$eq": ["$logs.half_day", True]}, 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ifNull": ["$logs", False]},
                                    {"$gt": ["$logs.late_minutes", 0]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$logs.late_minutes", 0]}
                },
                "leave_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": [{"$ifNull": ["$logs.status", None]}, "LEAVE"]},
                            1,
                            0,
                        ]
                    }
                },
                "on_duty_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": [{"$ifNull": ["$logs.status", None]}, "ON_DUTY"]},
                            1,
                            0,
                        ]
                    }
                },
                "work_hours_list": {
                    "$push": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ifNull": ["$logs", False]},
                                    {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$ne": ["$logs.work_hours", None]},
                                ]
                            },
                            "$logs.work_hours",
                            "$$REMOVE",
                        ]
                    }
                },
            }
        },
        {"$sort": {"_id": ASCENDING}},
    ]


@app.get("/analytics/departments/summary", tags=["analytics"])
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None, description="Restrict to one department (exact match)."),
):
    pipeline = build_department_summary_pipeline(month, department)
    results = list(db.employees.aggregate(pipeline))

    items = []
    for row in results:
        headcount = len(row["employees_set"])
        if headcount == 0:
            continue

        work_hours_list = row.get("work_hours_list", [])
        avg_work = (
            round_half_up(sum(work_hours_list) / len(work_hours_list), 2)
            if work_hours_list
            else None
        )

        items.append(
            {
                "department": row["_id"],
                "headcount": headcount,
                "present_days": row["present_days"],
                "avg_work_hours": avg_work,
                "late_count": row["late_count"],
                "total_late_minutes": row["total_late_minutes"],
                "leave_count": row["leave_count"],
                "on_duty_count": row["on_duty_count"],
            }
        )

    return {"month": month, "items": items}


def build_leaderboard_pipeline(month: str, department: Optional[str] = None) -> list:
    pipeline: list = [
        {
            "$match": {
                "date": {"$regex": f"^{month}"},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]

    if department:
        pipeline.append({"$match": {"emp.department": department}})

    pipeline.append(
        {"$sort": {"total_late_minutes": DESCENDING, "_id": ASCENDING}}
    )
    return pipeline


@app.get("/analytics/leaderboard/late", tags=["analytics"])
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = Query(None),
):
    pipeline = build_leaderboard_pipeline(month, department)
    raw_items = list(db.attendance_logs.aggregate(pipeline))

    ranked_items = []
    current_rank = 1
    for i, item in enumerate(raw_items):
        if i > 0 and item["total_late_minutes"] < raw_items[i - 1]["total_late_minutes"]:
            current_rank = i + 1

        if current_rank <= limit:
            ranked_items.append(
                {
                    "rank": current_rank,
                    "emp_code": item["_id"],
                    "name": item["emp"]["name"],
                    "department": item["emp"]["department"],
                    "total_late_minutes": item["total_late_minutes"],
                    "late_count": item["late_count"],
                }
            )

    return {"month": month, "items": ranked_items}


def build_department_trend_pipeline(department: str, date_from: str, date_to: str) -> list:
    return [
        {
            "$match": {
                "department": department,
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"ecode": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$ecode"]},
                                    {"$gte": ["$date", date_from]},
                                    {"$lte": ["$date", date_to]},
                                ]
                            }
                        }
                    }
                ],
                "as": "logs",
            }
        },
    ]


@app.get("/analytics/departments/{department}/trend", tags=["analytics"])
def department_trend(
    department: str = Path(...),
    date_from: str = Query(..., alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: str = Query(..., alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    if date_to < date_from:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="to must be >= from",
        )

    start_d = dt_date.fromisoformat(date_from)
    end_d = dt_date.fromisoformat(date_to)
    num_days = (end_d - start_d).days + 1
    if num_days > 92:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="date range cannot exceed 92 days",
        )

    dept_employees = list(db.employees.find({"department": department}))
    if not dept_employees:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="unknown department",
        )

    emp_codes = [e["emp_code"] for e in dept_employees]
    logs_cursor = db.attendance_logs.find(
        {"emp_code": {"$in": emp_codes}, "date": {"$gte": date_from, "$lte": date_to}}
    )

    logs_by_date: Dict[str, list] = {}
    for log in logs_cursor:
        logs_by_date.setdefault(log["date"], []).append(log)

    items = []
    recent_rates: List[Optional[float]] = []

    curr = start_d
    while curr <= end_d:
        curr_str = curr.isoformat()
        is_working_day = curr.weekday() < 5

        headcount = sum(1 for e in dept_employees if e["joined_on"] <= curr_str)

        day_logs = logs_by_date.get(curr_str, [])
        present_count = 0.0
        late_count = 0

        for l in day_logs:
            if l.get("status") in ("PRESENT", "WFH", "ON_DUTY"):
                present_count += 0.5 if l.get("half_day") else 1.0
            if l.get("late_minutes", 0) > 0:
                late_count += 1

        if is_working_day and headcount > 0:
            att_rate = round_half_up(present_count / headcount, 4)
            recent_rates.append(att_rate)
        else:
            att_rate = None
            recent_rates.append(None)

        window = recent_rates[-7:]
        valid_window = [r for r in window if r is not None]
        moving_avg = (
            round_half_up(sum(valid_window) / len(valid_window), 4)
            if valid_window
            else None
        )

        items.append(
            {
                "date": curr_str,
                "is_working_day": is_working_day,
                "headcount": headcount,
                "present_count": present_count,
                "late_count": late_count,
                "attendance_rate": att_rate,
                "moving_avg_7d": moving_avg,
            }
        )
        curr += timedelta(days=1)

    return {"department": department, "items": items}


@app.get("/admin/explain/{endpoint}", tags=["admin"])
def explain_endpoint(
    endpoint: str = Path(..., pattern=r"^(attendance_list|employee_monthly|department_summary|late_leaderboard|department_trend)$"),
    emp_code: Optional[str] = Query(None),
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    status: Optional[str] = Query(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$"),
    from_date: Optional[str] = Query(None, alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: Optional[str] = Query(None, alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        if date_from and date_to and date_from > date_to:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="date_from must be <= date_to",
            )
        q: Dict[str, Any] = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status:
            q["status"] = status

        skip = (page - 1) * page_size
        cursor = (
            db.attendance_logs.find(q)
            .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
            .skip(skip)
            .limit(page_size)
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": cursor.explain(),
        }

    elif endpoint == "employee_monthly":
        if not emp_code or not month:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="emp_code and month are required",
            )
        pipeline = build_employee_monthly_pipeline(emp_code, month)
        explain_data = db.command(
            "explain",
            {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": explain_data,
        }

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="month is required",
            )
        pipeline = build_department_summary_pipeline(month, department)
        explain_data = db.command(
            "explain",
            {"aggregate": "employees", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "employees",
            "explain": explain_data,
        }

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="month is required",
            )
        pipeline = build_leaderboard_pipeline(month, department)
        explain_data = db.command(
            "explain",
            {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": explain_data,
        }

    elif endpoint == "department_trend":
        if not department or not from_date or not to_date:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="department, from, and to are required",
            )
        pipeline = build_department_trend_pipeline(department, from_date, to_date)
        explain_data = db.command(
            "explain",
            {"aggregate": "employees", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "employees",
            "explain": explain_data,
        }

    raise HTTPException(status_code=400, detail="unhandled endpoint")