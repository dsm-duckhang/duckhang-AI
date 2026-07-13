# duckhang-AI

## Current Verification API

This service exposes a single multipart endpoint for Spring-compatible photo verification.

- `POST /verify`
- multipart form fields:
  - `event`: JSON string containing `category`, `title`, optional `artist_name`, `venue_name`, and optional `keywords`
  - `image`: image file to verify

Example `curl` request:

```bash
curl -X POST "http://localhost:8000/verify" \
  -F 'event={"category":"concert","title":"Duck Show","keywords":["duck","show"]}' \
  -F "image=@/path/to/photo.jpg"
```

If you want FastAPI to compute distance, include the event latitude/longitude in the `event` JSON:

```bash
curl -X POST "http://localhost:8000/verify" \
  -F 'event={"category":"concert","title":"Duck Show","keywords":["duck","show"],"latitude":37.5123,"longitude":127.1023}' \
  -F "image=@/path/to/photo.jpg"
```
