import pytest

BASE_EVENT = {"category": "FANMEETING", "title": "테스트 행사"}

SPRING_EVENT = {
    "title": "덕질 콘서트",
    "category": "CONCERT",
    "categoryLabel": "콘서트",
    "description": "첫 단독 콘서트",
    "venueName": "올림픽공원 체조경기장",
    "address": "서울시 송파구",
    "relatedLink": "https://example.com/events/duckhang-concert",
    "userLatitude": 37.234,
    "userLongitude": 127.808,
    "latitude": 37.5211,
    "longitude": 127.1229,
    "startAt": "2026-09-01T10:00:00Z",
    "endAt": "2026-09-01T20:00:00Z",
}


def test_valid_multipart_request_returns_200(client, mock_providers, post_verify, sharp_image_bytes):
    resp = post_verify(BASE_EVENT, sharp_image_bytes)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"VERIFIED", "ADDITIONAL_CAPTURE_REQUIRED", "REJECTED"}
    assert isinstance(body["reasons"], list)


def test_spring_event_payload_with_camel_case_fields_returns_200(
    client,
    mock_providers,
    post_verify,
    sharp_image_bytes,
):
    mock_providers["paddle"].return_value = {"text": "덕질 콘서트 올림픽공원 체조경기장"}

    resp = post_verify(SPRING_EVENT, sharp_image_bytes)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "VERIFIED"
    assert "OCR_STRONG_MATCH" in body["reasons"]


def test_spring_user_location_fields_map_to_radius_check(client, mock_providers, post_verify, sharp_image_bytes):
    event = {
        **SPRING_EVENT,
        "radiusM": 100,
    }

    resp = post_verify(event, sharp_image_bytes)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "REJECTED"
    assert "OUT_OF_RADIUS" in body["reasons"]


def test_malformed_event_json_returns_400(client, post_verify, sharp_image_bytes):
    resp = post_verify("{this is not valid json", sharp_image_bytes)

    assert resp.status_code == 400
    assert "invalid event payload" in resp.json()["detail"]


def test_missing_required_event_field_returns_400(client, post_verify, sharp_image_bytes):
    event = {"category": "FANMEETING"}  # "title" is required

    resp = post_verify(event, sharp_image_bytes)

    assert resp.status_code == 400
    assert "invalid event payload" in resp.json()["detail"]


def test_corrupt_image_returns_rejected_invalid_file(client, mock_providers, post_verify, corrupt_image_bytes):
    resp = post_verify(BASE_EVENT, corrupt_image_bytes)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "REJECTED"
    assert body["reasons"] == ["INVALID_FILE"]


def test_corrupt_image_with_out_of_radius_prioritizes_invalid_file(client, mock_providers, post_verify, corrupt_image_bytes):
    event = {
        **BASE_EVENT,
        "venue_lat": 37.0,
        "venue_lng": 127.0,
        "radius_m": 100,
        "user_lat": 37.5,
        "user_lng": 127.5,
    }

    resp = post_verify(event, corrupt_image_bytes)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "REJECTED"
    assert "INVALID_FILE" in body["reasons"]
    assert "OUT_OF_RADIUS" not in body["reasons"]


def test_blurry_image_returns_additional_capture_required(client, mock_providers, post_verify, blurry_image_bytes):
    resp = post_verify(BASE_EVENT, blurry_image_bytes)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ADDITIONAL_CAPTURE_REQUIRED"
    assert "IMAGE_TOO_BLURRY" in body["reasons"]


def test_out_of_radius_returns_rejected(client, mock_providers, post_verify, sharp_image_bytes):
    event = {
        **BASE_EVENT,
        "venue_lat": 37.0,
        "venue_lng": 127.0,
        "radius_m": 100,
        "user_lat": 37.5,
        "user_lng": 127.5,
    }

    resp = post_verify(event, sharp_image_bytes)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "REJECTED"
    assert "OUT_OF_RADIUS" in body["reasons"]


def test_ocr_strong_match_returns_verified(client, mock_providers, post_verify, sharp_image_bytes):
    mock_providers["paddle"].return_value = {"text": "테스트행사 방문 인증 완료"}
    event = {"category": "FANMEETING", "title": "테스트행사"}

    resp = post_verify(event, sharp_image_bytes)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "VERIFIED"
    assert "OCR_STRONG_MATCH" in body["reasons"]


def test_all_ai_providers_failing_returns_5xx(client, mock_providers, post_verify, sharp_image_bytes):
    mock_providers["paddle"].side_effect = Exception("ocr provider down")
    mock_providers["claude"].side_effect = Exception("claude provider down")
    mock_providers["gemini"].side_effect = Exception("gemini provider down")

    resp = post_verify(BASE_EVENT, sharp_image_bytes)

    assert resp.status_code == 503
    assert resp.json() == {"detail": "OCR and Vision providers all failed or timed out"}


def test_missing_event_part_returns_422(client, sharp_image_bytes):
    resp = client.post(
        "/verify",
        files={"image": ("photo.jpg", sharp_image_bytes, "image/jpeg")},
    )

    assert resp.status_code == 422


def test_missing_image_part_returns_422(client):
    resp = client.post(
        "/verify",
        data={"event": '{"category":"FANMEETING","title":"테스트 행사"}'},
    )

    assert resp.status_code == 422


@pytest.mark.parametrize("vision_confidence", [-5.0, 0.0, 0.5, 1.0, 3.7])
def test_confidence_is_always_between_0_and_1(client, mock_providers, post_verify, sharp_image_bytes, vision_confidence):
    mock_providers["claude"].return_value = {"confidence": vision_confidence, "event_relation": "RELATED"}

    resp = post_verify(BASE_EVENT, sharp_image_bytes)

    assert resp.status_code == 200
    confidence = resp.json()["confidence"]
    assert 0.0 <= confidence <= 1.0
