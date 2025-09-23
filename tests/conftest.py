--- a/tests/conftest.py
+++ b/tests/conftest.py
@@
-from fastapi.testclient import TestClient
-from app.main import app
+from fastapi.testclient import TestClient
+from app.main import app
+
+# --- Test-only switch: disable SlowAPI rate limiting to avoid 429 in CI ---
+# We keep runtime limits; tests don't need them and they cause flakiness when
+# multiple tests call /api/curate in the same minute.
+try:
+    # SlowAPI stores limiter on app.state.limiter in recent versions
+    if hasattr(app.state, "limiter"):
+        app.state.limiter.enabled = False
+except Exception:
+    # Fallback for older layouts: try module-level limiter
+    try:
+        from app.main import limiter as _limiter  # type: ignore
+        _limiter.enabled = False  # pragma: no cover
+    except Exception:
+        pass
@@
 def client():
-    with TestClient(app) as c:
+    with TestClient(app) as c:
         yield c

