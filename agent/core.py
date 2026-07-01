"""
ReAct döngüsü — agent'ın kalbi.

Akış:
  1) Sistem promptu + kullanıcı sorusu + araç şemalarını Groq'a gönder.
  2) Model ya bir/birden fazla ARAÇ ÇAĞRISI döndürür ya da NİHAİ CEVAP.
  3) Araç çağrısı varsa: her birini çalıştır, sonucu 'tool' mesajı olarak
     konuşmaya ekle ve 1. adıma dön (model artık sonucu görür).
  4) Araç çağrısı yoksa: modelin metni nihai cevaptır — döngü biter.

max_steps, modelin sonsuza dek araç çağırmasını engelleyen güvenlik sınırıdır.
"""

import json

from groq import Groq
import groq as groq_sdk

from agent.config import settings
from agent.tools import TOOL_FUNCS, TOOL_SCHEMAS


SYSTEM_PROMPT = """Sen, kullanıcının sorusunu araştıran otonom bir asistansın.

Elindeki araçlar:
- web_search(query): web'de arama yapar (başlık, URL, özet döndürür)
- read_url(url): bir sayfanın tam metnini okur
- calculator(expression): kesin aritmetik hesap yapar

Kurallar:
- Bilmediğin ya da güncel olabilecek bilgileri KAFADAN uydurma; önce web_search ile ara.
- Arama özetleri yetersiz kalırsa, en umut vadeden URL'yi read_url ile derinlemesine oku.
- Sayısal işlemleri kafadan yapma, calculator aracını kullan.
- Yeterince bilgi topladığında DUR ve nihai cevabı ver.
- Cevabı kullanıcının dilinde, net ve öz yaz; sonunda kullandığın kaynakların
  URL'lerini "Kaynaklar:" başlığı altında listele.
"""


class AgentError(Exception):
    """Yapılandırma ya da LLM çağrısı hatası."""


def _client() -> Groq:
    if not settings.groq_api_key:
        raise AgentError(
            "GROQ_API_KEY tanımlı değil. https://console.groq.com'dan ücretsiz "
            "key al ve .env'e ekle."
        )
    return Groq(api_key=settings.groq_api_key)


def _assistant_msg(msg) -> dict:
    """Modelin araç-çağrılı mesajını, konuşmaya geri eklenebilir dict'e çevir."""
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ],
    }


def run_agent(question: str, on_event=None) -> dict:
    """
    Soruyu ReAct döngüsüyle yanıtla.

    on_event(*ev): opsiyonel callback — adımları canlı göstermek için. Olaylar:
      ("think", text)                 modelin ara düşüncesi (varsa)
      ("tool_call", name, args)       bir araç çağrılıyor
      ("tool_result", name, result)   aracın döndürdüğü sonuç
      ("final", answer)               nihai cevap

    Dönüş: {"answer": str, "steps": [olaylar]}
    """
    client = _client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: list = []

    def emit(*ev):
        trace.append(ev)
        if on_event:
            on_event(*ev)

    for _ in range(settings.max_steps):
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
            )
        except groq_sdk.APIError as e:
            raise AgentError(f"LLM çağrısı başarısız: {e}")

        msg = response.choices[0].message

        # Araç çağrısı yoksa → bu, nihai cevap
        if not msg.tool_calls:
            answer = msg.content or ""
            emit("final", answer)
            return {"answer": answer, "steps": trace}

        # Modelin bu turdaki (araç çağrılı) mesajını konuşmaya ekle
        messages.append(_assistant_msg(msg))
        if msg.content:
            emit("think", msg.content)

        # Çağrılan her aracı çalıştır, sonucunu 'tool' mesajı olarak geri besle
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            emit("tool_call", name, args)

            func = TOOL_FUNCS.get(name)
            if func is None:
                result = f"Bilinmeyen araç: {name}"
            else:
                try:
                    result = func(**args)
                except Exception as e:
                    result = f"Araç hatası ({name}): {e}"
            emit("tool_result", name, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    # Adım limiti doldu → araçsız son bir çağrıyla özet iste
    messages.append({
        "role": "user",
        "content": "Adım limitine ulaşıldı. Topladığın bilgiyle en iyi nihai cevabı şimdi ver.",
    })
    try:
        final = client.chat.completions.create(
            model=settings.model, messages=messages, temperature=0.2
        )
        answer = final.choices[0].message.content or "Cevap üretilemedi."
    except groq_sdk.APIError as e:
        raise AgentError(f"LLM çağrısı başarısız: {e}")
    emit("final", answer)
    return {"answer": answer, "steps": trace}
