/*
Drop into your existing React dashboard. Presentation-only, per your
design — no Allow/Deny buttons here, just reflects the decision made
on the phone app. Polls every 3s; swap for a websocket later if you
want true real-time without polling overhead.
*/

import { useEffect, useState } from "react";

const BACKEND_URL = "http://YOUR_BACKEND_HOST:8000"; // <-- set this

export default function IntruderAlertStatus() {
  const [alert, setAlert] = useState(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/carvis/alert/latest`);
        const data = await res.json();
        setAlert(data.status === "none" ? null : data);
      } catch (err) {
        // backend unreachable — leave last known state on screen
      }
    };
    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, []);

  if (!alert) {
    return <div className="alert-status alert-status--idle">No active alerts</div>;
  }

  if (alert.status === "pending") {
    return (
      <div className="alert-status alert-status--pending">
        Unrecognized driver detected — awaiting owner response on phone…
      </div>
    );
  }

  if (alert.status === "denied") {
    return <div className="alert-status alert-status--danger">🚨 Intruder confirmed</div>;
  }

  if (alert.status === "allowed") {
    return (
      <div className="alert-status alert-status--success">
        Access allowed — trip continuing
      </div>
    );
  }

  return null;
}