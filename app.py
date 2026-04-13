"""
AI Daily News Dashboard
Aggregates AI news, model benchmarks, and agent/CLAW trends.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Cache – refreshed every 30 minutes in the background
# ---------------------------------------------------------------------------
CACHE = {
    "news": [],
    "benchmarks": [],
    "claws": [],
    "tools": [],
    "last_updated": None,
}
CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# RSS / Atom feeds for major AI companies & outlets
# ---------------------------------------------------------------------------
FEEDS = {
    # Company blogs
    "OpenAI": "https://openai.com/blog/rss.xml",
    "Anthropic": "https://www.anthropic.com/rss.xml",
    "Google AI": "https://blog.google/technology/ai/rss/",
    "NVIDIA AI": "https://blogs.nvidia.com/feed/",
    "Amazon Science": "https://www.amazon.science/index.rss",
    "Meta AI": "https://ai.meta.com/blog/rss/",
    "Microsoft AI": "https://blogs.microsoft.com/ai/feed/",
    "Hugging Face": "https://huggingface.co/blog/feed.xml",
    # News outlets
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "Ars Technica AI": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "MIT Tech Review": "https://www.technologyreview.com/feed/",
}

# ---------------------------------------------------------------------------
# Known open-source CLAW / Agent frameworks to track
# ---------------------------------------------------------------------------
CLAW_REPOS = [
    {"name": "OpenClaw", "repo": "openclaw/openclaw", "desc": "Open-source CLAW agent framework"},
    {"name": "NanoClaw", "repo": "nanoclaw/nanoclaw", "desc": "Lightweight CLAW for edge deployment"},
    {"name": "AutoGPT", "repo": "Significant-Gravitas/AutoGPT", "desc": "Autonomous GPT-4 agent"},
    {"name": "CrewAI", "repo": "crewAIInc/crewAI", "desc": "Multi-agent orchestration framework"},
    {"name": "LangGraph", "repo": "langchain-ai/langgraph", "desc": "Stateful multi-agent apps with LangChain"},
    {"name": "OpenHands", "repo": "All-Hands-AI/OpenHands", "desc": "Open-source AI software dev agents"},
    {"name": "Autogen", "repo": "microsoft/autogen", "desc": "Microsoft multi-agent conversation framework"},
    {"name": "smolagents", "repo": "huggingface/smolagents", "desc": "Hugging Face lightweight agent library"},
    {"name": "Phidata", "repo": "phidatahq/phidata", "desc": "Build multi-modal agents with memory & tools"},
    {"name": "Claude Code", "repo": "anthropics/claude-code", "desc": "Anthropic CLI agentic coding tool"},
    {"name": "Composio", "repo": "ComposioHQ/composio", "desc": "Agent tooling platform — 250+ tool integrations"},
    {"name": "BrowserUse", "repo": "browser-use/browser-use", "desc": "AI agents that control a web browser"},
    {"name": "Agno", "repo": "agno-agi/agno", "desc": "Lightweight library for building multi-modal agents"},
    {"name": "Pydantic AI", "repo": "pydantic/pydantic-ai", "desc": "Agent framework powered by Pydantic"},
]

AI_TOOLS = [
    {"name": "Cursor", "url": "https://cursor.sh", "category": "IDE / Coding", "desc": "AI-first code editor"},
    {"name": "Claude Code", "url": "https://claude.ai/code", "category": "IDE / Coding", "desc": "Anthropic CLI agentic coding"},
    {"name": "GitHub Copilot", "url": "https://github.com/features/copilot", "category": "IDE / Coding", "desc": "AI pair programmer"},
    {"name": "v0 by Vercel", "url": "https://v0.dev", "category": "UI Generation", "desc": "Generative UI from text prompts"},
    {"name": "Replit Agent", "url": "https://replit.com", "category": "IDE / Coding", "desc": "AI app builder in the browser"},
    {"name": "Bolt.new", "url": "https://bolt.new", "category": "App Builder", "desc": "Full-stack AI app generation"},
    {"name": "Lovable", "url": "https://lovable.dev", "category": "App Builder", "desc": "AI full-stack engineer"},
    {"name": "Devin", "url": "https://devin.ai", "category": "Autonomous Agent", "desc": "Cognition AI software engineer"},
    {"name": "NotebookLM", "url": "https://notebooklm.google", "category": "Research", "desc": "Google AI research notebook"},
    {"name": "Perplexity", "url": "https://perplexity.ai", "category": "Search / Research", "desc": "AI-powered search engine"},
    {"name": "ChatGPT", "url": "https://chat.openai.com", "category": "Chat / Assistant", "desc": "OpenAI conversational AI"},
    {"name": "Claude", "url": "https://claude.ai", "category": "Chat / Assistant", "desc": "Anthropic conversational AI"},
    {"name": "Gemini", "url": "https://gemini.google.com", "category": "Chat / Assistant", "desc": "Google conversational AI"},
    {"name": "Midjourney", "url": "https://midjourney.com", "category": "Image Generation", "desc": "AI image generation"},
    {"name": "Runway", "url": "https://runwayml.com", "category": "Video Generation", "desc": "AI video generation & editing"},
    {"name": "ElevenLabs", "url": "https://elevenlabs.io", "category": "Audio / Voice", "desc": "AI voice synthesis"},
    {"name": "Suno", "url": "https://suno.com", "category": "Audio / Music", "desc": "AI music generation"},
]

# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": "AI-News-Dashboard/1.0 (Educational/Personal Use)"
}


def fetch_feeds():
    """Fetch and parse all RSS feeds, return sorted list of articles."""
    articles = []
    cutoff = datetime.utcnow() - timedelta(days=3)

    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url, request_headers=HEADERS)
            for entry in feed.entries[:10]:
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    published = datetime(*entry.updated_parsed[:6])
                else:
                    published = datetime.utcnow()

                if published < cutoff:
                    continue

                summary = ""
                if hasattr(entry, "summary"):
                    soup = BeautifulSoup(entry.summary, "html.parser")
                    summary = soup.get_text()[:300]

                articles.append({
                    "title": entry.get("title", "Untitled"),
                    "link": entry.get("link", "#"),
                    "source": source,
                    "published": published.strftime("%Y-%m-%d %H:%M"),
                    "summary": summary,
                    "sort_key": published.timestamp(),
                })
        except Exception:
            continue

    articles.sort(key=lambda x: x["sort_key"], reverse=True)
    return articles[:100]


def fetch_github_stats():
    """Fetch GitHub stats for CLAW / agent repos."""
    results = []
    for item in CLAW_REPOS:
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{item['repo']}",
                headers={**HEADERS, "Accept": "application/vnd.github.v3+json"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Fetch recent commits count (last 7 days)
                recent_activity = "N/A"
                try:
                    since = (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z"
                    commits_resp = requests.get(
                        f"https://api.github.com/repos/{item['repo']}/commits",
                        params={"since": since, "per_page": 1},
                        headers={**HEADERS, "Accept": "application/vnd.github.v3+json"},
                        timeout=10,
                    )
                    if commits_resp.status_code == 200:
                        # Check Link header for total count estimate
                        recent_activity = len(commits_resp.json())
                        if "Link" in commits_resp.headers:
                            recent_activity = "30+"
                except Exception:
                    pass

                pushed_at = data.get("pushed_at", "")
                if pushed_at:
                    pushed_dt = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
                    days_ago = (datetime.now(pushed_dt.tzinfo) - pushed_dt).days
                    last_push = f"{days_ago}d ago" if days_ago > 0 else "today"
                else:
                    last_push = "N/A"

                results.append({
                    "name": item["name"],
                    "repo": item["repo"],
                    "desc": data.get("description", item["desc"])[:120],
                    "stars": data.get("stargazers_count", 0),
                    "forks": data.get("forks_count", 0),
                    "open_issues": data.get("open_issues_count", 0),
                    "language": data.get("language", "N/A"),
                    "last_push": last_push,
                    "recent_commits": recent_activity,
                    "url": data.get("html_url", f"https://github.com/{item['repo']}"),
                    "topics": data.get("topics", [])[:5],
                    "watchers": data.get("subscribers_count", 0),
                })
            else:
                results.append({
                    "name": item["name"],
                    "repo": item["repo"],
                    "desc": item["desc"],
                    "stars": 0, "forks": 0, "open_issues": 0,
                    "language": "N/A", "last_push": "N/A",
                    "recent_commits": "N/A",
                    "url": f"https://github.com/{item['repo']}",
                    "topics": [], "watchers": 0,
                })
        except Exception:
            results.append({
                "name": item["name"],
                "repo": item["repo"],
                "desc": item["desc"],
                "stars": 0, "forks": 0, "open_issues": 0,
                "language": "N/A", "last_push": "N/A",
                "recent_commits": "N/A",
                "url": f"https://github.com/{item['repo']}",
                "topics": [], "watchers": 0,
            })
        time.sleep(0.5)  # rate limiting

    results.sort(key=lambda x: x["stars"], reverse=True)
    return results


def refresh_cache():
    """Refresh all cached data."""
    news = fetch_feeds()
    claws = fetch_github_stats()

    with CACHE_LOCK:
        CACHE["news"] = news
        CACHE["claws"] = claws
        CACHE["tools"] = AI_TOOLS
        CACHE["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Cache refreshed: {len(news)} articles, {len(claws)} repos")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    with CACHE_LOCK:
        return render_template("index.html", last_updated=CACHE.get("last_updated", "Loading..."))


@app.route("/api/news")
def api_news():
    with CACHE_LOCK:
        return jsonify(CACHE["news"])


@app.route("/api/claws")
def api_claws():
    with CACHE_LOCK:
        return jsonify(CACHE["claws"])


@app.route("/api/tools")
def api_tools():
    with CACHE_LOCK:
        return jsonify(CACHE["tools"])


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    threading.Thread(target=refresh_cache, daemon=True).start()
    return jsonify({"status": "refreshing"})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def create_app():
    # Initial data load in background
    threading.Thread(target=refresh_cache, daemon=True).start()

    # Schedule refresh every 30 minutes
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_cache, "interval", minutes=30)
    scheduler.start()

    return app


if __name__ == "__main__":
    create_app()
    print("AI Daily News Dashboard running at http://localhost:5000")
    app.run(debug=False, port=5000)
