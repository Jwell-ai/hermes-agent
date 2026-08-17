import json
from unittest.mock import patch

import requests

from tools.alphart_tools import _storybook_image_report_has_permanent_failure
from tools.alphart_tools import _generate_storybook_images_with_retries


def test_storybook_image_report_detects_permanent_relay_authorization_failure():
    report = {
        "required": 1,
        "generated": 0,
        "missing": 1,
        "errors": [
            "relay request failed: status=401 code=UNAUTHORIZED message=internal user uuid does not match resolved user"
        ],
    }

    assert _storybook_image_report_has_permanent_failure(report)


def test_storybook_image_report_does_not_treat_generation_failure_as_permanent():
    report = {
        "required": 1,
        "generated": 0,
        "missing": 1,
        "errors": ["page 1: provider temporarily unavailable"],
    }

    assert not _storybook_image_report_has_permanent_failure(report)


def test_storybook_image_generation_does_not_retry_permanent_authorization_failure():
    response = requests.Response()
    response.status_code = 502
    response._content = json.dumps(
        {
            "image_generation": {
                "required": 1,
                "generated": 0,
                "missing": 1,
                "errors": [
                    "relay request failed: status=401 code=UNAUTHORIZED message=internal user uuid does not match resolved user"
                ],
            }
        }
    ).encode()

    with (
        patch("tools.alphart_tools.requests.post", return_value=response) as post,
        patch("tools.alphart_tools.time.sleep") as sleep,
        patch("tools.alphart_tools._internal_api_url", return_value="http://edu/internal"),
        patch("tools.alphart_tools._internal_relay_headers", return_value={}),
    ):
        _generate_storybook_images_with_retries("storybook-1", {}, timeout=1)

    post.assert_called_once()
    sleep.assert_not_called()
