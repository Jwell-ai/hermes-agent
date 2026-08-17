"""Volcengine Ark and BytePlus (Doubao) provider profiles."""

from providers import register_provider
from providers.base import ProviderProfile


volcengine = ProviderProfile(
    name="volcengine",
    aliases=("doubao", "ark", "volcengine-ark"),
    display_name="Volcengine Ark / Doubao",
    description="Volcengine Ark and Doubao OpenAI-compatible models",
    signup_url="https://console.volcengine.com/ark",
    env_vars=("ARK_API_KEY", "VOLCENGINE_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

byteplus = ProviderProfile(
    name="byteplus",
    aliases=("byteplus-ark", "byteplus-modelark"),
    display_name="BytePlus ModelArk",
    description="BytePlus ModelArk OpenAI-compatible models",
    signup_url="https://console.byteplus.com/ark",
    env_vars=("BYTEPLUS_API_KEY", "ARK_API_KEY"),
    base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
)

register_provider(volcengine)
register_provider(byteplus)
