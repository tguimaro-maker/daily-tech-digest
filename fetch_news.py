#!/usr/bin/env python3
"""
Daily Tech Digest — fetch_news.py
Scrapes configured sources, sends to Groq API for curation, saves JSON.
"""

import json
import os
import sys
import time
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
import feedparser
from groq import Groq

# Carrega .env em desenvolvimento local (opcional; no GitHub Actions a env já vem do workflow)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = 15
MIN_RELEVANCE = 3
MAX_ITEMS_PER_SOURCE = 12
# Free tier da Groq: 6.000 tokens/minuto, e o limite conta INPUT + max_tokens.
# Com ~40 itens: input ~1.200 + max_tokens 3.500 = ~4.700, com folga sob 6.000.
MAX_TOTAL_ITEMS = 40
GROQ_MAX_TOKENS = 3500
NEWS_DIR = Path("news")

SYSTEM_PROMPT = """Você é um curador de notícias para profissionais de tecnologia, negócios e design no Brasil.
Retorne SEMPRE um único objeto JSON válido, sem texto adicional, sem markdown, sem explicações."""

USER_PROMPT_TEMPLATE = """Para cada item abaixo, gere:
- summary: resumo em português (máx. 2 frases, direto ao ponto)
- category: uma de [Tech, Negócios, Design, IA, Open Source]
- relevance: score de 1-5 (5 = muito relevante para o público-alvo)

Cada item começa com seu identificador entre colchetes, ex: [abc123].
Use EXATAMENTE esse mesmo identificador no campo "id" da resposta.

Itens:
{items}

Retorne um objeto JSON no formato:
{{"results": [{{"id": "abc123", "summary": "...", "category": "...", "relevance": 0}}, ...]}}"""


def load_sources() -> list[dict]:
    with open("sources.json", encoding="utf-8") as f:
        return [s for s in json.load(f) if s.get("active")]


def scrape_source(client: httpx.Client, source: dict) -> list[dict]:
    """Lê o feed RSS/Atom da fonte e extrai título, link e data de cada item."""
    name = source["name"]
    try:
        resp = client.get(source["feed"], headers=HEADERS, timeout=TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        log.warning("Falha ao acessar %s: %s", name, e)
        return []

    feed = feedparser.parse(resp.content)

    items = []
    for entry in feed.entries[:MAX_ITEMS_PER_SOURCE]:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or len(title) < 10 or not link:
            continue
        # Data de publicação do próprio feed, quando disponível
        published = ""
        dt = entry.get("published_parsed") or entry.get("updated_parsed")
        if dt:
            published = time.strftime("%Y-%m-%d", dt)
        item_id = hashlib.md5(f"{name}:{title}".encode()).hexdigest()[:8]
        items.append({
            "id": item_id,
            "title": title,
            "url": link,
            "source": name,
            "category": source["category"],
            "published_at": published,
        })

    log.info("%-25s → %d itens", name, len(items))
    return items


def parse_groq_response(content: str) -> list[dict]:
    """Extrai a lista de resultados da resposta da Groq, tolerante a falhas.

    Tenta, em ordem: objeto {"results": [...]}, array direto, e por fim
    salvamento via regex de objetos {...} completos (caso a resposta venha
    truncada ou com texto extra)."""
    # Remove cercas markdown, se houver
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            if content.lstrip().startswith("json"):
                content = content.lstrip()[4:]
        content = content.strip()

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            for key in ("results", "items", "data"):
                if isinstance(data.get(key), list):
                    return data[key]
            # objeto único inesperado
            return [data] if "id" in data else []
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Salvamento: extrai objetos {...} completos individualmente
    import re
    salvaged = []
    for match in re.finditer(r"\{[^{}]*\}", content):
        try:
            obj = json.loads(match.group(0))
            if "id" in obj:
                salvaged.append(obj)
        except json.JSONDecodeError:
            continue

    if salvaged:
        log.warning("Resposta da Groq malformada; %d itens recuperados via salvamento", len(salvaged))
    else:
        log.error("Não foi possível extrair JSON da Groq. Início: %s", content[:300])
    return salvaged


def curate_with_groq(client: Groq, raw_items: list[dict]) -> list[dict]:
    if not raw_items:
        return []

    items_text = "\n".join(
        f"[{item['id']}] {item['title']} | {item['source']}"
        for item in raw_items
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(items=items_text)},
            ],
            temperature=0.3,
            max_tokens=GROQ_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        log.error("Erro na Groq API: %s", e)
        return []

    content = response.choices[0].message.content.strip()
    curated = parse_groq_response(content)
    if not curated:
        return []

    # Map curated data back to raw items by id
    curated_map = {c["id"]: c for c in curated if isinstance(c, dict) and "id" in c}

    enriched = []
    for item in raw_items:
        extra = curated_map.get(item["id"])
        if not extra:
            continue
        relevance = int(extra.get("relevance", 0))
        if relevance < MIN_RELEVANCE:
            continue
        enriched.append({
            "id": item["id"],
            "title": item["title"],
            "summary": extra.get("summary", ""),
            "url": item["url"],
            "source": item["source"],
            "category": extra.get("category", item["category"]),
            "relevance": relevance,
            "published_at": item.get("published_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        })

    return sorted(enriched, key=lambda x: x["relevance"], reverse=True)


def save_output(items: list[dict], today: str) -> None:
    NEWS_DIR.mkdir(exist_ok=True)

    daily_file = NEWS_DIR / f"{today}.json"
    payload = {
        "date": today,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(items),
        "items": items,
    }
    daily_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Salvo: %s (%d notícias)", daily_file, len(items))

    index_file = NEWS_DIR / "index.json"
    existing = []
    if index_file.exists():
        try:
            existing = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    if today not in existing:
        existing.insert(0, today)
    existing = existing[:30]
    index_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Índice atualizado: %d datas", len(existing))


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.error("GROQ_API_KEY não definida")
        sys.exit(1)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log.info("=== Daily Tech Digest — %s ===", today)

    sources = load_sources()
    log.info("Fontes ativas: %d", len(sources))

    per_source: list[list[dict]] = []
    with httpx.Client() as http:
        for source in sources:
            per_source.append(scrape_source(http, source))

    # Intercala fontes (round-robin) para manter diversidade antes do teto global,
    # evitando que as primeiras fontes ocupem todas as vagas.
    raw_items: list[dict] = []
    for i in range(max((len(s) for s in per_source), default=0)):
        for src_items in per_source:
            if i < len(src_items):
                raw_items.append(src_items[i])

    total_coletado = len(raw_items)
    if len(raw_items) > MAX_TOTAL_ITEMS:
        raw_items = raw_items[:MAX_TOTAL_ITEMS]
        log.info("Coletados %d itens; limitados a %d (teto de tokens/min do free tier)",
                 total_coletado, MAX_TOTAL_ITEMS)
    else:
        log.info("Total bruto: %d itens", total_coletado)

    if not raw_items:
        log.warning("Nenhum item coletado. Encerrando.")
        sys.exit(0)

    groq_client = Groq(api_key=api_key)
    curated = curate_with_groq(groq_client, raw_items)
    log.info("Após curadoria: %d itens (score >= %d)", len(curated), MIN_RELEVANCE)

    save_output(curated, today)
    log.info("Concluído.")


if __name__ == "__main__":
    main()
