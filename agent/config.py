"""
Agent ayarları — tek yerden yönetilir, .env'den okunur (pydantic-settings).
Kod içine sabit değer gömmek yerine her şey buradan gelir.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Groq (function-calling destekli, ücretsiz) — https://console.groq.com
    groq_api_key: str | None = None
    model: str = "llama-3.3-70b-versatile"   # araç kullanımını iyi destekler

    # --- Agent döngüsü ---
    max_steps: int = 6                       # sonsuz döngüye karşı üst sınır

    # --- Araç ayarları ---
    search_results: int = 5                  # web_search kaç sonuç getirsin
    max_read_chars: int = 3000               # read_url azami kaç karakter döndürsün


settings = Settings()
