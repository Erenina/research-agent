"""
Agent'ın araçları (tools).

Her araç iki parçadan oluşur:
  1) Sıradan bir Python fonksiyonu (işi yapan).
  2) LLM'e "böyle bir araç var, şu argümanları alır" diyen bir JSON şeması.

Model, şemalara bakıp "web_search('...') çağır" gibi yapılandırılmış bir istek
üretir; biz TOOL_FUNCS'tan ilgili fonksiyonu bulup çalıştırır, sonucu modele
geri veririz. Sonuçları HEP string döndürüyoruz, çünkü model metin okur.
"""

import ast
import operator

from agent.config import settings


# --------------------------------------------------------------------------
# 1) web_search — DuckDuckGo (API anahtarı gerektirmez)
# --------------------------------------------------------------------------
def web_search(query: str) -> str:
    """Web'de ara; ilk birkaç sonucun başlık + URL + özetini döndür."""
    from ddgs import DDGS
    try:
        results = list(DDGS().text(query, max_results=settings.search_results))
    except Exception as e:
        return f"Arama başarısız oldu: {e}"
    if not results:
        return "Hiç sonuç bulunamadı."
    blocks = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        url = r.get("href") or r.get("url") or r.get("link") or ""
        snippet = r.get("body") or r.get("snippet") or ""
        blocks.append(f"[{i}] {title}\nURL: {url}\n{snippet}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# 2) read_url — bir sayfayı açıp temiz metnini döndür
# --------------------------------------------------------------------------
def read_url(url: str) -> str:
    """Verilen URL'yi indir, HTML'i temizle, düz metni (kısaltılmış) döndür."""
    import requests
    from bs4 import BeautifulSoup
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "research-agent/1.0"})
        resp.raise_for_status()
    except Exception as e:
        return f"Sayfa alınamadı: {e}"
    soup = BeautifulSoup(resp.text, "html.parser")
    # Anlamsız/gürültülü etiketleri at
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    if not text:
        return "Sayfadan metin çıkarılamadı."
    limit = settings.max_read_chars
    return text[:limit] + ("... [kısaltıldı]" if len(text) > limit else "")


# --------------------------------------------------------------------------
# 3) calculator — güvenli aritmetik (tehlikeli eval KULLANMADAN, ast ile)
# --------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node):
    """AST düğümünü güvenle hesapla — sadece sayı ve izinli operatörlere izin ver."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("İzin verilmeyen ifade.")


def calculator(expression: str) -> str:
    """Aritmetik ifadeyi güvenli biçimde hesapla (ör. '2 * (3 + 4) ** 2')."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"Hesaplanamadı: {e}"


# --------------------------------------------------------------------------
# Kayıt defteri: isim -> fonksiyon  +  LLM'e verilecek JSON şemaları
# --------------------------------------------------------------------------
TOOL_FUNCS = {
    "web_search": web_search,
    "read_url": read_url,
    "calculator": calculator,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Bir konuyu web'de aramak için kullan. Güncel bilgi, olgu ya da "
                "kaynak bulman gerektiğinde çağır. İlk birkaç sonucun başlık, URL "
                "ve özetini döndürür."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Arama sorgusu"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": (
                "Bir web sayfasının tam içeriğini okumak için kullan. web_search "
                "sonuçlarındaki ilginç bir URL'yi derinlemesine okumak istediğinde çağır."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Okunacak sayfanın tam URL'si"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Aritmetik hesaplama için kullan (toplama, çarpma, üs, vb.). Sayısal "
                "işlemleri kafadan yapma; kesinlik için bu aracı çağır."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Hesaplanacak aritmetik ifade, ör. '(1234 * 12) / 7'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]
