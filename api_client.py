import time
import requests


BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
TASK_ENDPOINT = "/content_generation/tasks"


class SeedanceAPIClient:
    """Thin HTTP wrapper around the BytePlus Ark content-generation API."""

    def __init__(self, api_key: str, base_url: str = BASE_URL):
        if not api_key or not api_key.strip():
            raise ValueError("ARK_API_KEY is required. Provide it in the node or set the ARK_API_KEY environment variable.")
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def create_task(self, payload: dict) -> str:
        """Submit a generation task and return its task ID."""
        url = f"{self.base_url}{TASK_ENDPOINT}"
        resp = requests.post(url, json=payload, headers=self.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("id")
        if not task_id:
            raise RuntimeError(f"Unexpected response (no 'id'): {data}")
        return task_id

    def get_task(self, task_id: str) -> dict:
        """Fetch the current status of a task."""
        url = f"{self.base_url}{TASK_ENDPOINT}/{task_id}"
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def poll_task(
        self,
        task_id: str,
        poll_interval: int = 10,
        max_wait: int = 600,
    ) -> dict:
        """
        Block until the task succeeds or fails (or max_wait seconds elapse).

        Returns the full task dict on success.
        Raises RuntimeError on failure or TimeoutError on timeout.
        """
        elapsed = 0
        print(f"[Seedance] Polling task {task_id} ...")
        while elapsed < max_wait:
            result = self.get_task(task_id)
            status = result.get("status", "unknown")
            print(f"[Seedance] Status: {status} ({elapsed}s elapsed)")
            if status == "succeeded":
                return result
            if status == "failed":
                error = result.get("error") or result.get("message") or "Unknown error"
                raise RuntimeError(f"[Seedance] Task {task_id} failed: {error}")
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(
            f"[Seedance] Task {task_id} did not complete within {max_wait}s."
        )
