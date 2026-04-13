"""
AI Daily News Dashboard — Streamlit Edition
Aggregates AI news, model benchmarks, and agent/CLAW trends.
"""

import time
from datetime import datetime, timezone, timedelta

import feedparser
import requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Daily News",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark sleek overrides */
    .stApp { background-color: #0f0f1a; }
    header[data-testid="stHeader"] { background-color: #1a1a2e; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #1a1a2e; padding: 4px 8px; border-radius: 12px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; color: #888; border-radius: 8px; padding: 8px 20px; }
    .stTabs [aria-selected="true"] { background-color: #6c63ff !important; color: white !important; }
    div[data-testid="stMetric"] { background-color: #1a1a2e; border: 1px solid #25253e; border-radius: 12px; padding: 16px; }
    div[data-testid="stMetric"] label { color: #888; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #6c63ff; }
    .source-badge { display: inline-block; font-size: 11px; padding: 2px 10px; border-radius: 999px; font-weight: 500; margin-right: 6px; }
    .news-card { background: #1a1a2e; border: 1px solid #25253e; border-radius: 12px; padding: 16px; margin-bottom: 10px; transition: all 0.2s; }
    .news-card:hover { border-color: #6c63ff; transform: translateY(-1px); }
    .news-card h4 { margin: 4px 0 6px 0; }
    .news-card a { color: #e0e0e0; text-decoration: none; }
    .news-card a:hover { color: #6c63ff; }
    .news-card .summary { color: #666; font-size: 13px; margin-top: 6px; }
    .news-card .meta { color: #555; font-size: 11px; font-family: 'JetBrains Mono', monospace; }
    .claw-card { background: #1a1a2e; border: 1px solid #25253e; border-radius: 12px; padding: 16px; margin-bottom: 8px; }
    .claw-card:hover { border-color: #6c63ff; }
    .tool-card { background: #1a1a2e; border: 1px solid #25253e; border-radius: 12px; padding: 16px; text-align: center; }
    .tool-card:hover { border-color: #6c63ff; }
    .link-section { background: #1a1a2e; border: 1px solid #25253e; border-radius: 12px; padding: 20px; }
    .link-section h4 { margin-bottom: 12px; }
    .link-section a { color: #aaa; text-decoration: none; display: block; padding: 4px 0; font-size: 14px; }
    .link-section a:hover { color: #6c63ff; }
    .stars-bar { background: linear-gradient(90deg, #6c63ff, #00d4ff); border-radius: 4px; height: 6px; }
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
FEEDS = {
    "OpenAI": "https://openai.com/blog/rss.xml",
    "Anthropic": "https://www.anthropic.com/rss.xml",
    "Google AI": "https://blog.google/technology/ai/rss/",
    "NVIDIA AI": "https://blogs.nvidia.com/feed/",
    "Amazon Science": "https://www.amazon.science/index.rss",
    "Meta AI": "https://ai.meta.com/blog/rss/",
    "Microsoft AI": "https://blogs.microsoft.com/ai/feed/",
    "Hugging Face": "https://huggingface.co/blog/feed.xml",
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "Ars Technica AI": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "MIT Tech Review": "https://www.technologyreview.com/feed/",
}

COMPANY_SOURCES = {"OpenAI", "Anthropic", "Google AI", "NVIDIA AI", "Amazon Science", "Meta AI", "Microsoft AI", "Hugging Face"}

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
    {"name": "Composio", "repo": "ComposioHQ/composio", "desc": "Agent tooling platform - 250+ tool integrations"},
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

SOURCE_COLORS = {
    "OpenAI": "#10a37f",
    "Anthropic": "#d97706",
    "Google AI": "#4285f4",
    "NVIDIA AI": "#76b900",
    "Amazon Science": "#ff9900",
    "Meta AI": "#1877f2",
    "Microsoft AI": "#00bcf2",
    "Hugging Face": "#ffd21e",
    "The Verge AI": "#a855f7",
    "TechCrunch AI": "#0a9e01",
    "VentureBeat AI": "#e11d48",
    "Ars Technica AI": "#ff4500",
    "MIT Tech Review": "#14b8a6",
}

HEADERS = {"User-Agent": "AI-News-Dashboard/1.0 (Educational/Personal Use)"}

# ---------------------------------------------------------------------------
# Data fetching with st.cache_data (auto-expires)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_feeds():
    articles = []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=3)

    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url, request_headers=HEADERS)
            for entry in feed.entries[:10]:
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                else:
                    published = now

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
                    "is_company": source in COMPANY_SOURCES,
                })
        except Exception:
            continue

    articles.sort(key=lambda x: x["sort_key"], reverse=True)
    return articles[:100]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_github_stats():
    results = []
    now = datetime.now(timezone.utc)

    for item in CLAW_REPOS:
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{item['repo']}",
                headers={**HEADERS, "Accept": "application/vnd.github.v3+json"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()

                recent_activity = "N/A"
                try:
                    since = (now - timedelta(days=7)).isoformat()
                    cr = requests.get(
                        f"https://api.github.com/repos/{item['repo']}/commits",
                        params={"since": since, "per_page": 1},
                        headers={**HEADERS, "Accept": "application/vnd.github.v3+json"},
                        timeout=10,
                    )
                    if cr.status_code == 200:
                        recent_activity = len(cr.json())
                        if "Link" in cr.headers:
                            recent_activity = "30+"
                except Exception:
                    pass

                pushed_at = data.get("pushed_at", "")
                if pushed_at:
                    pushed_dt = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
                    days_ago = (now - pushed_dt).days
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
                results.append(_fallback_repo(item))
        except Exception:
            results.append(_fallback_repo(item))
        time.sleep(0.5)

    results.sort(key=lambda x: x["stars"], reverse=True)
    return results


def _fallback_repo(item):
    return {
        "name": item["name"], "repo": item["repo"], "desc": item["desc"],
        "stars": 0, "forks": 0, "open_issues": 0, "language": "N/A",
        "last_push": "N/A", "recent_commits": "N/A",
        "url": f"https://github.com/{item['repo']}", "topics": [], "watchers": 0,
    }


def fmt_num(n):
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("## AI Daily News")
    st.caption("Real-time intelligence feed — AI news, agents, CLAWs & tools")
with col_status:
    st.markdown("")
    if st.button("Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_news, tab_claws, tab_tools, tab_explore = st.tabs([
    "News Feed",
    "Agents & CLAWs",
    "AI Tools",
    "Explore & Links",
])

# ===== NEWS FEED TAB =====================================================
with tab_news:
    with st.spinner("Loading news feeds..."):
        articles = fetch_feeds()

    # Filters
    col_filter, col_search = st.columns([1, 2])
    with col_filter:
        source_filter = st.selectbox("Filter", ["All", "Companies", "News Outlets"], label_visibility="collapsed")
    with col_search:
        search = st.text_input("Search", placeholder="Search articles...", label_visibility="collapsed")

    filtered = articles
    if source_filter == "Companies":
        filtered = [a for a in filtered if a["is_company"]]
    elif source_filter == "News Outlets":
        filtered = [a for a in filtered if not a["is_company"]]
    if search:
        q = search.lower()
        filtered = [a for a in filtered if q in a["title"].lower() or q in a["summary"].lower() or q in a["source"].lower()]

    st.markdown(f"**{len(filtered)}** articles from the last 3 days")

    for a in filtered:
        color = SOURCE_COLORS.get(a["source"], "#666")
        st.markdown(f"""
        <div class="news-card">
            <span class="source-badge" style="background:{color}22; color:{color};">{a['source']}</span>
            <span class="meta">{a['published']}</span>
            <h4><a href="{a['link']}" target="_blank">{a['title']}</a></h4>
            {f'<div class="summary">{a["summary"]}</div>' if a["summary"] else ''}
        </div>
        """, unsafe_allow_html=True)

    if not filtered:
        st.info("No articles match your filters. Try broadening your search.")

# ===== AGENTS & CLAWs TAB ================================================
with tab_claws:
    with st.spinner("Fetching GitHub data for agent frameworks..."):
        repos = fetch_github_stats()

    if repos:
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        total_stars = sum(r["stars"] for r in repos)
        most_active = max(repos, key=lambda r: r["recent_commits"] if isinstance(r["recent_commits"], int) else 0)
        lang_counts = {}
        for r in repos:
            if r["language"] != "N/A":
                lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1
        top_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "N/A"

        m1.metric("Repos Tracked", len(repos))
        m2.metric("Combined Stars", fmt_num(total_stars))
        m3.metric("Most Active (7d)", most_active["name"])
        m4.metric("Top Language", top_lang)

        # Sort options
        sort_by = st.radio("Sort by", ["Stars", "Forks", "Recent Activity"], horizontal=True, label_visibility="collapsed")
        if sort_by == "Stars":
            repos_sorted = sorted(repos, key=lambda r: r["stars"], reverse=True)
        elif sort_by == "Forks":
            repos_sorted = sorted(repos, key=lambda r: r["forks"], reverse=True)
        else:
            def parse_push(v):
                if v == "today": return 0
                if v == "N/A": return 999
                return int(v.replace("d ago", ""))
            repos_sorted = sorted(repos, key=lambda r: parse_push(r["last_push"]))

        max_stars = max(r["stars"] for r in repos_sorted) if repos_sorted else 1

        for i, r in enumerate(repos_sorted):
            bar_pct = max(5, (r["stars"] / max_stars) * 100) if max_stars > 0 else 5
            topics_html = " ".join(
                f'<span style="font-size:10px;background:#25253e;color:#888;padding:2px 6px;border-radius:4px;">{t}</span>'
                for t in r["topics"]
            )
            st.markdown(f"""
            <div class="claw-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div style="flex:1;">
                        <span style="color:#555;font-size:12px;font-family:monospace;">#{i+1}</span>
                        <a href="{r['url']}" target="_blank" style="color:#6c63ff;font-weight:600;font-size:15px;text-decoration:none;margin-left:6px;">{r['name']}</a>
                        <span style="color:#444;font-size:11px;margin-left:8px;font-family:monospace;">{r['repo']}</span>
                        <div style="color:#888;font-size:13px;margin:4px 0 6px 0;">{r['desc']}</div>
                        <div style="margin-bottom:6px;">{topics_html}</div>
                        <div class="stars-bar" style="width:{bar_pct}%;"></div>
                    </div>
                    <div style="text-align:right;min-width:100px;">
                        <div style="font-size:16px;font-weight:700;color:#e0e0e0;">{fmt_num(r['stars'])} <span style="font-size:10px;color:#666;">stars</span></div>
                        <div style="font-size:12px;color:#888;">{fmt_num(r['forks'])} forks</div>
                        <div style="font-size:12px;color:#666;">{r['language']}</div>
                        <div style="font-size:10px;color:#555;">pushed {r['last_push']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Could not fetch GitHub data. GitHub API may be rate-limited.")

# ===== AI TOOLS TAB =======================================================
with tab_tools:
    categories = sorted(set(t["category"] for t in AI_TOOLS))
    selected_cat = st.radio("Category", ["All"] + categories, horizontal=True, label_visibility="collapsed")

    tools_filtered = AI_TOOLS if selected_cat == "All" else [t for t in AI_TOOLS if t["category"] == selected_cat]

    cols = st.columns(3)
    for i, t in enumerate(tools_filtered):
        with cols[i % 3]:
            st.markdown(f"""
            <a href="{t['url']}" target="_blank" style="text-decoration:none;">
                <div class="tool-card">
                    <div style="font-size:15px;font-weight:600;color:#e0e0e0;">{t['name']}</div>
                    <div style="font-size:11px;color:#6c63ff;margin:4px 0;">{t['category']}</div>
                    <div style="font-size:13px;color:#666;">{t['desc']}</div>
                </div>
            </a>
            """, unsafe_allow_html=True)
            st.markdown("")  # spacing

# ===== EXPLORE & LINKS TAB ================================================
with tab_explore:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="link-section">
            <h4 style="color:#6c63ff;">Research & Papers</h4>
            <a href="https://arxiv.org/list/cs.AI/recent" target="_blank">arXiv cs.AI — Latest Papers</a>
            <a href="https://arxiv.org/list/cs.CL/recent" target="_blank">arXiv cs.CL — Computation & Language</a>
            <a href="https://arxiv.org/list/cs.LG/recent" target="_blank">arXiv cs.LG — Machine Learning</a>
            <a href="https://paperswithcode.com" target="_blank">Papers With Code</a>
            <a href="https://huggingface.co/papers" target="_blank">Hugging Face Daily Papers</a>
            <a href="https://www.semanticscholar.org" target="_blank">Semantic Scholar</a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        st.markdown("""
        <div class="link-section">
            <h4 style="color:#ff8c00;">Newsletters & Podcasts</h4>
            <a href="https://www.therundown.ai" target="_blank">The Rundown AI (daily)</a>
            <a href="https://tldr.tech/ai" target="_blank">TLDR AI (daily)</a>
            <a href="https://bensbites.com" target="_blank">Ben's Bites</a>
            <a href="https://importai.substack.com" target="_blank">Import AI — Jack Clark</a>
            <a href="https://thegradient.pub" target="_blank">The Gradient</a>
            <a href="https://www.latent.space/podcast" target="_blank">Latent Space Podcast</a>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="link-section">
            <h4 style="color:#00ff88;">Benchmarks & Leaderboards</h4>
            <a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard" target="_blank">Open LLM Leaderboard (HF)</a>
            <a href="https://chat.lmsys.org/?leaderboard" target="_blank">Chatbot Arena / LMSYS</a>
            <a href="https://artificialanalysis.ai" target="_blank">Artificial Analysis</a>
            <a href="https://livebench.ai" target="_blank">LiveBench — Contamination-free evals</a>
            <a href="https://scale.com/leaderboard" target="_blank">Scale AI SEAL Leaderboard</a>
            <a href="https://klu.ai/llm-leaderboard" target="_blank">Klu.ai LLM Leaderboard</a>
            <a href="https://www.vellum.ai/llm-leaderboard" target="_blank">Vellum LLM Leaderboard</a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        st.markdown("""
        <div class="link-section">
            <h4 style="color:#ff3e8a;">Communities</h4>
            <a href="https://reddit.com/r/MachineLearning" target="_blank">r/MachineLearning</a>
            <a href="https://reddit.com/r/LocalLLaMA" target="_blank">r/LocalLLaMA</a>
            <a href="https://reddit.com/r/artificial" target="_blank">r/artificial</a>
            <a href="https://news.ycombinator.com" target="_blank">Hacker News</a>
            <a href="https://discord.gg/huggingface" target="_blank">Hugging Face Discord</a>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="link-section">
            <h4 style="color:#00d4ff;">Model Hubs & Registries</h4>
            <a href="https://huggingface.co/models" target="_blank">Hugging Face Models</a>
            <a href="https://ollama.com/library" target="_blank">Ollama Model Library</a>
            <a href="https://openrouter.ai/models" target="_blank">OpenRouter — Multi-provider API</a>
            <a href="https://replicate.com/explore" target="_blank">Replicate — Run Models via API</a>
            <a href="https://together.ai/models" target="_blank">Together AI Models</a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        st.markdown("""
        <div class="link-section">
            <h4 style="color:#ccc;">Company Blogs (Direct)</h4>
            <a href="https://openai.com/blog" target="_blank">OpenAI Blog</a>
            <a href="https://www.anthropic.com/news" target="_blank">Anthropic News</a>
            <a href="https://blog.google/technology/ai/" target="_blank">Google AI Blog</a>
            <a href="https://blogs.nvidia.com/blog/category/deep-learning/" target="_blank">NVIDIA AI Blog</a>
            <a href="https://ai.meta.com/blog/" target="_blank">Meta AI Blog</a>
            <a href="https://www.amazon.science" target="_blank">Amazon Science</a>
            <a href="https://blogs.microsoft.com/ai/" target="_blank">Microsoft AI Blog</a>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("AI Daily News Dashboard — Data from RSS feeds & GitHub API — Auto-caches for 30 min")
