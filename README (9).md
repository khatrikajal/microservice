# 🍽️ Kitchen Ledger — Canteen Inventory Management System

A complete backend REST API built with **FastAPI + PostgreSQL** for managing canteen inventory, stock requests, approvals, FIFO stock issue, expiry tracking, and reporting.

---

## 📁 Folder Structure

```
kitchen_ledger/
│
├── backend/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app entry point — all routers registered
│   ├── config.py                      # Environment variables via pydantic-settings
│   │
│   ├── connection/
│   │   ├── __init__.py
│   │   └── db_connection.py           # psycopg2 PostgreSQL connection
│   │
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── otp_table.py               # Creates otp_verifications table on startup
│   │   ├── schemas.py                 # Pydantic schemas for auth endpoints
│   │   ├── utils.py                   # Password hashing, JWT, OTP gen, email sending
│   │   ├── service.py                 # Auth business logic
│   │   ├── router.py                  # Auth endpoints
│   │   └── dependencies.py            # get_current_user, require_role()
│   │
│   ├── users/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   └── router.py
│   │
│   ├── vendors/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   └── router.py
│   │
│   ├── categories/
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── router.py
│   │
│   ├── products/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   └── router.py
│   │
│   ├── inventory/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── service.py                 # Purchase recording, FIFO stock, expiry alerts
│   │   └── router.py
│   │
│   ├── requests/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── service.py                 # Create request, approve/reject/modify, FIFO issue
│   │   └── router.py
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── service.py                 # Store manager + admin dashboard queries
│   │   └── router.py
│   │
│   └── reports/
│       ├── __init__.py
│       ├── service.py                 # All 7 report queries
│       └── router.py
│
├── create_tables.py                   # Run once to create all DB tables, ENUMs, indexes
├── requirements.txt
├── .env.example                       # Copy this to .env and fill in your values
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites

Make sure you have the following installed:

- Python 3.10+
- PostgreSQL 13+
- pip

---

### 2. Clone / Download the Project

```bash
# If using git
git clone <your-repo-url>
cd kitchen_ledger

# Or extract the downloaded ZIP and open the folder
```

---

### 3. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Set Up PostgreSQL Database

Open your PostgreSQL client (pgAdmin or psql) and run:

```sql
CREATE DATABASE canteen_db;
```

---

### 6. Configure Environment Variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and update:

```env
# Your PostgreSQL connection string
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/canteen_db

# JWT — change this to a long random string in production
SECRET_KEY=your_super_secret_key_change_this_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Gmail SMTP — use App Password (not your Gmail login password)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your_email@gmail.com
EMAIL_HOST_PASSWORD=your_gmail_app_password
DEFAULT_FROM_EMAIL=your_email@gmail.com

# OTP expires in N minutes
OTP_EXPIRE_MINUTES=10
```

> **Gmail App Password**: Go to Google Account → Security → 2-Step Verification → App Passwords → Generate one for "Mail".

---

### 7. Create Database Tables

Run this once to create all tables, ENUMs, and indexes:

```bash
python create_tables.py
```

You should see:
```
All tables, ENUMs, and indexes created successfully!
```

---

### 8. Start the Server

```bash
uvicorn backend.main:app --reload
```

Server runs at: `http://localhost:8000`

---

### 9. Open API Documentation

FastAPI auto-generates interactive docs:

| URL | Description |
|---|---|
| `http://localhost:8000/docs` | Swagger UI — test all endpoints |
| `http://localhost:8000/redoc` | ReDoc — clean API reference |

---

## 🔐 Authentication Flow

### Signup (2 steps)

```
Step 1 — POST /auth/signup/request
  Body: { name, email, password, role, contact }
  → OTP sent to email

Step 2 — POST /auth/signup/verify
  Body: { email, otp }
  → Account created
```

### Login

```
POST /auth/login
  Body: { email, password }
  → Returns: { access_token, token_type, user_id, name, role }
```

### Using the Token

Add this header to all protected requests:
```
Authorization: Bearer <your_access_token>
```

### Forgot / Reset Password

```
Step 1 — POST /auth/forgot-password
  Body: { email }
  → OTP sent to email

Step 2 — POST /auth/reset-password
  Body: { email, otp, new_password }
  → Password updated
```

---

## 📋 Complete API Reference

### Auth `/auth`
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/auth/signup/request` | Send OTP to email | ❌ |
| POST | `/auth/signup/verify` | Verify OTP & create account | ❌ |
| POST | `/auth/login` | Login, get JWT token | ❌ |
| POST | `/auth/forgot-password` | Send password reset OTP | ❌ |
| POST | `/auth/reset-password` | Reset password with OTP | ❌ |
| GET | `/auth/me` | Get current user info | ✅ |

### Users `/users`
| Method | Endpoint | Description | Role |
|---|---|---|---|
| GET | `/users/` | List all users | admin |
| GET | `/users/chefs` | List chefs for dropdown | any |
| GET | `/users/me` | My profile | any |
| GET | `/users/{id}` | Get user by ID | admin |
| PUT | `/users/{id}` | Update user | admin |
| PATCH | `/users/{id}/deactivate` | Deactivate user | admin |
| PATCH | `/users/{id}/activate` | Activate user | admin |
| POST | `/users/change-password` | Change own password | any |

### Vendors `/vendors`
| Method | Endpoint | Description | Role |
|---|---|---|---|
| GET | `/vendors/` | List vendors | any |
| POST | `/vendors/` | Create vendor | admin, store_manager |
| GET | `/vendors/{id}` | Get vendor | any |
| PUT | `/vendors/{id}` | Update vendor | admin, store_manager |
| PATCH | `/vendors/{id}/deactivate` | Deactivate vendor | admin, store_manager |

### Categories `/categories`
| Method | Endpoint | Description | Role |
|---|---|---|---|
| GET | `/categories/` | List categories | any |
| POST | `/categories/` | Create category | admin, store_manager |
| PUT | `/categories/{id}` | Update category | admin, store_manager |
| PATCH | `/categories/{id}/deactivate` | Deactivate | admin, store_manager |

### Products `/products`
| Method | Endpoint | Description | Role |
|---|---|---|---|
| GET | `/products/` | List products | any |
| GET | `/products/low-stock` | Products below min level | any |
| GET | `/products/{id}` | Get product | any |
| POST | `/products/` | Create product | admin, store_manager |
| PUT | `/products/{id}` | Update product | admin, store_manager |
| PATCH | `/products/{id}/deactivate` | Deactivate | admin, store_manager |

### Inventory `/inventory`
| Method | Endpoint | Description | Role |
|---|---|---|---|
| GET | `/inventory/batches` | List all batches | any |
| GET | `/inventory/stock` | Stock summary per product | any |
| GET | `/inventory/expiring?within_days=30` | Expiring batches | any |
| POST | `/inventory/purchase` | Record new stock purchase | admin, store_manager |
| POST | `/inventory/run-expiry-alerts` | Trigger expiry alert check | admin |

### Requests `/requests`
| Method | Endpoint | Description | Role |
|---|---|---|---|
| GET | `/requests/` | List requests | any (chefs see own only) |
| POST | `/requests/` | Create stock request | any |
| GET | `/requests/{id}` | Request detail + items + approvals | any |
| POST | `/requests/{id}/approve` | Approve / Reject / Modify | admin |
| POST | `/requests/{id}/issue` | Issue stock via FIFO | admin, store_manager |

### Dashboard `/dashboard`
| Method | Endpoint | Description | Role |
|---|---|---|---|
| GET | `/dashboard/store-manager` | Store manager summary | admin, store_manager |
| GET | `/dashboard/admin` | Admin summary | admin |

### Reports `/reports`
| Method | Endpoint | Description | Role |
|---|---|---|---|
| GET | `/reports/daily-stock` | Current stock all products | admin, store_manager |
| GET | `/reports/consumption?date_from=&date_to=` | Item consumption | admin, store_manager |
| GET | `/reports/vendor-purchases?date_from=&date_to=` | Vendor purchase history | admin, store_manager |
| GET | `/reports/expiry` | Expiry status all batches | admin, store_manager |
| GET | `/reports/monthly-usage?year=2025` | Monthly usage by year | admin, store_manager |
| GET | `/reports/item-wise?product_id=&date_from=&date_to=` | Product transaction history | admin, store_manager |
| GET | `/reports/audit-logs?date_from=&date_to=` | Full audit trail | admin |

---

## 👥 User Roles

| Role | Description |
|---|---|
| `admin` | Full access — approve requests, view all reports, manage users |
| `store_manager` | Manage stock, create purchases, issue approved requests |
| `chef` | Create requests, view own request status |

---

## 🔄 Stock Request Workflow

```
1. Chef / Store Manager creates request  →  Status: PENDING
2. Admin reviews → APPROVED / REJECTED / MODIFIED
3. Store Manager issues stock           →  Status: ISSUED
   └── FIFO deduction runs automatically across batches
```

---

## 📦 FIFO Logic

When stock is issued, the system deducts from the **oldest batch first** (by purchase date). If a single batch doesn't cover the full quantity, it moves to the next batch automatically.

**Example:**
```
Product: Rice
Batch 1 (Jan 01) → 10 kg available
Batch 2 (Feb 01) → 20 kg available

Request: 12 kg
→ Deduct 10 kg from Batch 1 (now 0)
→ Deduct  2 kg from Batch 2 (now 18)
```

---

## 🚨 Expiry Alerts

Batches are flagged at three levels:

| Level | `alert_threshold` | Trigger |
|---|---|---|
| Warning | 30 | 30 days or less to expiry |
| Urgent | 7 | 7 days or less to expiry |
| Expired | 0 | Expiry date reached |

Call `POST /inventory/run-expiry-alerts` daily via a scheduler (e.g. APScheduler or a cron job) to update alert status on all batches.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI |
| Database | PostgreSQL |
| DB Driver | psycopg2 |
| Auth | JWT (python-jose) |
| Password | bcrypt (passlib) |
| Email | Gmail SMTP (smtplib) |
| Validation | Pydantic v2 |
| Server | Uvicorn |

---

## 📝 Notes

- All protected endpoints require `Authorization: Bearer <token>` header.
- OTP is valid for 10 minutes (configurable in `.env`).
- Each OTP is single-use — once verified it is marked as used.
- Audit logs are written automatically for all purchases, issues, and approvals.
- The `_pending_signups` dict in `auth/router.py` stores signup data between Step 1 and Step 2 in memory. For multi-server deployments, replace it with a `pending_signups` DB table or Redis.
