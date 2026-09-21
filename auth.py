"""
SQLite-based authentication and persistent session management.

Features:
- Name + Gmail based registration
- Email validation
- Name validation
- Strong password validation
- Password confirmation validation
- Duplicate email validation
- PBKDF2-SHA256 password hashing
- Unique random salt for every user
- Secure random session tokens
- Persistent sessions stored in SQLite
- Session expiration
- Session validation
- Session logout / invalidation
- Logout from all devices
- Expired session cleanup

Important:
This is suitable for a personal project, college project, or internal demo.

For production applications with real users, use a dedicated
authentication provider such as Auth0, Supabase Auth, Firebase Auth,
or another properly managed authentication system.
"""

import sqlite3
import hashlib
import os
import secrets
import time
import re


# ============================================================
# Database Configuration
# ============================================================

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "users.db"
)


# ============================================================
# Security Configuration
# ============================================================

# PBKDF2 iterations
PBKDF2_ITERATIONS = 100_000

# Session lifetime:
# 7 days
SESSION_DURATION = 7 * 24 * 60 * 60


# ============================================================
# Validation Configuration
# ============================================================

# ------------------------------------------------------------
# Name
# ------------------------------------------------------------
# Allows:
#   John
#   John Doe
#   Rahul Kumar
#
# Does NOT allow:
#   John123
#   John@
#   John_Doe
#   John-Doe
#
NAME_PATTERN = re.compile(
    r"^[A-Za-z]+(?: [A-Za-z]+)*$"
)

# ------------------------------------------------------------
# Gmail
# ------------------------------------------------------------
# Allows:
#   example@gmail.com
#   example@gmail.in
#
# Does NOT allow:
#   example@gmail
#   example@yahoo.com
#   example@gmail.co.uk
#   example@gmail.com.xyz
#
EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@gmail\.(com|in)$",
    re.IGNORECASE
)

# ------------------------------------------------------------
# Password
# ------------------------------------------------------------
# At least:
#   1 uppercase
#   1 lowercase
#   1 number
#   1 special character
#   minimum 6 characters
#
PASSWORD_MIN_LENGTH = 6


# ============================================================
# Database Connection
# ============================================================

def get_connection():
    """
    Create and return a SQLite database connection.

    Foreign keys are explicitly enabled for this connection.
    """

    conn = sqlite3.connect(DB_PATH)

    # Enable foreign-key support
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# ============================================================
# Create Database
# ============================================================

def create_db():
    """
    Create users and sessions tables if they don't already exist.
    """

    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------------
    # Users table
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )

    # --------------------------------------------------------
    # Sessions table
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,

            FOREIGN KEY (username)
            REFERENCES users(username)
            ON DELETE CASCADE
        )
        """
    )

    # --------------------------------------------------------
    # Index for faster email lookup
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_users_email
        ON users(email)
        """
    )

    # --------------------------------------------------------
    # Index for session username lookup
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_sessions_username
        ON sessions(username)
        """
    )

    conn.commit()
    conn.close()


# ============================================================
# Name Validation
# ============================================================

def validate_name(name):
    """
    Validate user's full name.

    Rules:
    - Required
    - Only English letters
    - Spaces allowed between names
    - No numbers
    - No special characters
    - No @ symbol
    """

    if name is None:
        return False, "Name is required."

    name = name.strip()

    if not name:
        return False, "Name is required."

    if len(name) < 2:
        return False, "Name must contain at least 2 characters."

    if len(name) > 100:
        return False, "Name must not exceed 100 characters."

    if not NAME_PATTERN.fullmatch(name):
        return (
            False,
            "Name can contain only letters and spaces."
        )

    return True, "Valid name."


# ============================================================
# Email Validation
# ============================================================

def validate_email(email):
    """
    Validate Gmail address.

    Allowed:
        example@gmail.com
        example@gmail.in

    Not allowed:
        example@yahoo.com
        example@gmail
        example@gmail.co.uk
    """

    if email is None:
        return False, "Gmail is required."

    email = email.strip().lower()

    if not email:
        return False, "Gmail is required."

    if len(email) > 254:
        return False, "Email address is too long."

    if not EMAIL_PATTERN.fullmatch(email):
        return (
            False,
            "Please enter a valid Gmail address ending with @gmail.com or @gmail.in."
        )

    return True, "Valid email."


# ============================================================
# Password Strength
# ============================================================

def get_password_strength(password):
    """
    Calculate password strength.

    Returns:
        {
            "score": int,
            "label": "Weak" / "Medium" / "Strong"
        }
    """

    if not password:
        return {
            "score": 0,
            "label": "Weak"
        }

    score = 0

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    if len(password) >= 6:
        score += 1

    if len(password) >= 8:
        score += 1

    if len(password) >= 12:
        score += 1

    # --------------------------------------------------------
    # Character types
    # --------------------------------------------------------

    if re.search(r"[A-Z]", password):
        score += 1

    if re.search(r"[a-z]", password):
        score += 1

    if re.search(r"[0-9]", password):
        score += 1

    if re.search(r"[^A-Za-z0-9]", password):
        score += 1

    # --------------------------------------------------------
    # Strength result
    # --------------------------------------------------------

    if score <= 3:
        label = "Weak"
    elif score <= 5:
        label = "Medium"
    else:
        label = "Strong"

    return {
        "score": score,
        "label": label
    }

# ============================================================
# Password Validation
# ============================================================

def validate_password(password):
    """
    Validate password strength.

    Requirements:
    - Minimum 6 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 number
    - At least 1 special character
    """

    if password is None or password == "":
        return False, "Password is required."

    errors = []

    # --------------------------------------------------------
    # Minimum length
    # --------------------------------------------------------

    if len(password) < PASSWORD_MIN_LENGTH:
        errors.append(
            "Password must be at least 6 characters long."
        )

    # --------------------------------------------------------
    # Uppercase
    # --------------------------------------------------------

    if not re.search(r"[A-Z]", password):
        errors.append(
            "Password must contain at least 1 uppercase letter."
        )

    # --------------------------------------------------------
    # Lowercase
    # --------------------------------------------------------

    if not re.search(r"[a-z]", password):
        errors.append(
            "Password must contain at least 1 lowercase letter."
        )

    # --------------------------------------------------------
    # Number
    # --------------------------------------------------------

    if not re.search(r"[0-9]", password):
        errors.append(
            "Password must contain at least 1 number."
        )

    # --------------------------------------------------------
    # Special character
    # --------------------------------------------------------

    if not re.search(r"[^A-Za-z0-9]", password):
        errors.append(
            "Password must contain at least 1 special character."
        )

    # --------------------------------------------------------
    # Return validation result
    # --------------------------------------------------------

    if errors:
        return False, " ".join(errors)

    return True, "Strong password."


# ============================================================
# Confirm Password Validation
# ============================================================

def validate_confirm_password(password, confirm_password):
    """
    Validate that password and confirm password match.
    """

    if confirm_password is None or confirm_password == "":
        return False, "Please confirm your password."

    if password != confirm_password:
        return False, "Passwords do not match."

    return True, "Passwords match."


# ============================================================
# Complete Signup Validation
# ============================================================

def validate_signup(name, email, password, confirm_password):
    """
    Validate all signup fields.

    Returns
    -------
    dict
        {
            "valid": True/False,
            "message": "...",
            "field": "name/email/password/confirm_password"
        }
    """

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    valid, message = validate_name(name)

    if not valid:
        return {
            "valid": False,
            "field": "name",
            "message": message
        }

    # --------------------------------------------------------
    # Email
    # --------------------------------------------------------

    valid, message = validate_email(email)

    if not valid:
        return {
            "valid": False,
            "field": "email",
            "message": message
        }

    # --------------------------------------------------------
    # Password
    # --------------------------------------------------------

    valid, message = validate_password(password)

    if not valid:
        return {
            "valid": False,
            "field": "password",
            "message": message,
            "strength": get_password_strength(password)
        }

    # --------------------------------------------------------
    # Confirm Password
    # --------------------------------------------------------

    valid, message = validate_confirm_password(
        password,
        confirm_password
    )

    if not valid:
        return {
            "valid": False,
            "field": "confirm_password",
            "message": message
        }

    # --------------------------------------------------------
    # Everything valid
    # --------------------------------------------------------

    return {
        "valid": True,
        "field": None,
        "message": "Signup validation successful.",
        "strength": "Strong"
    }


# ============================================================
# Password Hashing
# ============================================================

def _hash_password(password, salt):
    """
    Hash password using PBKDF2-HMAC-SHA256.

    Parameters
    ----------
    password : str
        Plaintext password.

    salt : str
        Hexadecimal salt.

    Returns
    -------
    str
        Hexadecimal password hash.
    """

    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PBKDF2_ITERATIONS,
    ).hex()


# ============================================================
# Check Duplicate Email
# ============================================================

def email_exists(email):
    """
    Check whether an email already exists.

    Returns
    -------
    bool
        True if email already exists.
    """

    if not email:
        return False

    email = email.strip().lower()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 1
        FROM users
        WHERE email = ?
        LIMIT 1
        """,
        (email,),
    )

    exists = cursor.fetchone() is not None

    conn.close()

    return exists


# ============================================================
# Register User
# ============================================================

def register_user(
    name,
    email,
    password,
    confirm_password
):
    """
    Register a new user.

    Returns
    -------
    dict
        Example:

        {
            "success": True,
            "message": "Account created successfully."
        }

    or

        {
            "success": False,
            "field": "email",
            "message": "An account with this email already exists."
        }
    """

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    name = name.strip() if name else ""
    email = email.strip().lower() if email else ""

    # --------------------------------------------------------
    # Validate signup fields
    # --------------------------------------------------------

    validation = validate_signup(
        name,
        email,
        password,
        confirm_password
    )

    if not validation["valid"]:

        return {
            "success": False,
            "field": validation.get("field"),
            "message": validation["message"]
        }

    # --------------------------------------------------------
    # Duplicate email check
    # --------------------------------------------------------

    if email_exists(email):

        return {
            "success": False,
            "field": "email",
            "message": "An account with this email already exists."
        }

    # --------------------------------------------------------
    # Username
    #
    # Existing session structure uses username as the user ID.
    # Here we use email because email is unique.
    # --------------------------------------------------------

    username = email

    # --------------------------------------------------------
    # Generate unique random salt
    # --------------------------------------------------------

    salt = secrets.token_hex(16)

    # --------------------------------------------------------
    # Hash password
    # --------------------------------------------------------

    password_hash = _hash_password(
        password,
        salt
    )

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                username,
                name,
                email,
                salt,
                password_hash
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                username,
                name,
                email,
                salt,
                password_hash,
            ),
        )

        conn.commit()

        return {
            "success": True,
            "message": "Account created successfully."
        }

    except sqlite3.IntegrityError:

        return {
            "success": False,
            "field": "email",
            "message": "An account with this email already exists."
        }

    finally:

        conn.close()


# ============================================================
# Login User
# ============================================================

def login_user(email, password):
    """
    Verify login credentials using email and password.

    Returns
    -------
    dict

        Successful login:

        {
            "success": True,
            "message": "Login successful.",
            "username": "...",
            "name": "...",
            "email": "..."
        }

        Wrong email:

        {
            "success": False,
            "field": "email",
            "message": "Email is incorrect."
        }

        Wrong password:

        {
            "success": False,
            "field": "password",
            "message": "Password is incorrect."
        }
    """

    # --------------------------------------------------------
    # Validate email
    # --------------------------------------------------------

    valid, message = validate_email(email)

    if not valid:

        return {
            "success": False,
            "field": "email",
            "message": message
        }

    # --------------------------------------------------------
    # Validate password presence
    # --------------------------------------------------------

    if password is None or password == "":

        return {
            "success": False,
            "field": "password",
            "message": "Password is required."
        }

    email = email.strip().lower()

    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------------
    # Find user by email
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT
            username,
            name,
            email,
            salt,
            password_hash
        FROM users
        WHERE email = ?
        LIMIT 1
        """,
        (email,),
    )

    row = cursor.fetchone()

    conn.close()

    # --------------------------------------------------------
    # Email doesn't exist
    # --------------------------------------------------------

    if row is None:

        return {
            "success": False,
            "field": "email",
            "message": "Email is incorrect."
        }

    username, name, stored_email, salt, stored_hash = row

    # --------------------------------------------------------
    # Hash entered password using stored salt
    # --------------------------------------------------------

    candidate_hash = _hash_password(
        password,
        salt
    )

    # --------------------------------------------------------
    # Constant-time comparison
    # --------------------------------------------------------

    if not secrets.compare_digest(
        candidate_hash,
        stored_hash
    ):

        return {
            "success": False,
            "field": "password",
            "message": "Password is incorrect."
        }

    # --------------------------------------------------------
    # Login successful
    # --------------------------------------------------------

    return {
        "success": True,
        "message": "Login successful.",
        "username": username,
        "name": name,
        "email": stored_email
    }


# ============================================================
# Create Persistent Session
# ============================================================

def create_session(username):
    """
    Create a secure persistent login session.

    Returns
    -------
    str or None
        Secure session token if successful.
    """

    username = username.strip() if username else ""

    if not username:
        return None

    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------------
    # Verify user exists
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT username
        FROM users
        WHERE username = ?
        """,
        (username,),
    )

    user = cursor.fetchone()

    if user is None:

        conn.close()

        return None

    # --------------------------------------------------------
    # Generate secure random session token
    # --------------------------------------------------------

    session_token = secrets.token_urlsafe(48)

    # --------------------------------------------------------
    # Session timestamps
    # --------------------------------------------------------

    created_at = time.time()

    expires_at = (
        created_at +
        SESSION_DURATION
    )

    # --------------------------------------------------------
    # Store session
    # --------------------------------------------------------

    cursor.execute(
        """
        INSERT INTO sessions
        (
            session_token,
            username,
            created_at,
            expires_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            session_token,
            username,
            created_at,
            expires_at,
        ),
    )

    conn.commit()
    conn.close()

    return session_token


# ============================================================
# Validate Session
# ============================================================

def validate_session(session_token):
    """
    Validate a session token.

    Returns
    -------
    str or None
        Username if valid.
        None if invalid or expired.
    """

    if not session_token:
        return None

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT username, expires_at
        FROM sessions
        WHERE session_token = ?
        """,
        (session_token,),
    )

    row = cursor.fetchone()

    if row is None:

        conn.close()

        return None

    username, expires_at = row

    # --------------------------------------------------------
    # Check expiration
    # --------------------------------------------------------

    if time.time() >= expires_at:

        cursor.execute(
            """
            DELETE FROM sessions
            WHERE session_token = ?
            """,
            (session_token,),
        )

        conn.commit()
        conn.close()

        return None

    conn.close()

    return username


# ============================================================
# Get User From Session
# ============================================================

def get_user_from_session(session_token):
    """
    Get complete user information from a valid session.

    Returns
    -------
    dict or None
    """

    if not session_token:
        return None

    username = validate_session(session_token)

    if username is None:
        return None

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT username, name, email
        FROM users
        WHERE username = ?
        """,
        (username,),
    )

    row = cursor.fetchone()

    conn.close()

    if row is None:
        return None

    username, name, email = row

    return {
        "username": username,
        "name": name,
        "email": email
    }


# ============================================================
# Logout Session
# ============================================================

def logout_session(session_token):
    """
    Invalidate a single session.

    Returns
    -------
    bool
        True if session was deleted.
        False otherwise.
    """

    if not session_token:
        return False

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM sessions
        WHERE session_token = ?
        """,
        (session_token,),
    )

    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


# ============================================================
# Logout All Sessions For User
# ============================================================

def logout_all_sessions(username):
    """
    Remove all active sessions belonging to a user.

    Useful for:
    - Logout from all devices
    - Security settings
    - Password change
    """

    username = username.strip() if username else ""

    if not username:
        return False

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM sessions
        WHERE username = ?
        """,
        (username,),
    )

    deleted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return deleted


# ============================================================
# Clean Expired Sessions
# ============================================================

def cleanup_expired_sessions():
    """
    Delete all expired sessions from the database.

    This can be called when the application starts.
    """

    current_time = time.time()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM sessions
        WHERE expires_at <= ?
        """,
        (current_time,),
    )

    deleted_count = cursor.rowcount

    conn.commit()
    conn.close()

    return deleted_count


# ============================================================
# Get Username From Session
# ============================================================

def get_username_from_session(session_token):
    """
    Convenience wrapper around validate_session().
    """

    return validate_session(session_token)


# ============================================================
# Get Password Strength For Frontend
# ============================================================

def password_strength(password):
    """
    Return password strength.

    Useful for frontend/API validation.

    Returns:
        Weak
        Medium
        Strong
    """

    return get_password_strength(password)


# ============================================================
# Initialize Database
# ============================================================

create_db()
cleanup_expired_sessions()