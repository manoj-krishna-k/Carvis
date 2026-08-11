"""
FCM push notification helper for CARVIS Phase 2.

Setup steps (one-time, free):
1. Go to https://console.firebase.google.com -> create a project (free, no card needed).
2. Project settings -> Service accounts -> Generate new private key.
   This downloads a JSON file — save it as `firebase-service-account.json`
   in this backend folder. DO NOT commit this file to GitHub (add to .gitignore).
3. pip install firebase-admin
"""

import firebase_admin
from firebase_admin import credentials, messaging

# Initialize once, at backend startup
_cred = credentials.Certificate("firebase-service-account.json")
firebase_admin.initialize_app(_cred)


def send_intruder_alert(device_token: str, alert_id: str):
    """
    Sends a silent/data-only push so the app itself decides how to show it
    and can handle the "tap to open decision screen" behavior reliably
    on both Android and iOS.
    """
    message = messaging.Message(
        data={
            "type": "intruder_alert",
            "alert_id": alert_id,
        },
        notification=messaging.Notification(
            title="CARVIS Alert",
            body="Unrecognized driving pattern detected. Tap to review.",
        ),
        token=device_token,
        android=messaging.AndroidConfig(priority="high"),
        apns=messaging.APNSConfig(
            payload=messaging.APNSPayload(
                aps=messaging.Aps(content_available=True, sound="default")
            )
        ),
    )
    response = messaging.send(message)
    return response