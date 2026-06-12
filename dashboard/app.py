import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
import datetime
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');

/* ══════════════════════════════════════════
   BASE
══════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
    color: #f0f0f0 !important;
    background: transparent !important;
}

/* Animated gradient mesh background */
[data-testid="stApp"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(232,160,32,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(100,60,180,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 70% 60% at 50% 50%, rgba(0,0,0,0) 0%, transparent 100%),
        #0f0f0f !important;
    background-attachment: fixed !important;
}

/* Hide chrome */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stSidebar"]      { display: none !important; }
[data-testid="collapsedControl"]{ display: none !important; }
.stDeployButton                 { display: none !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ══════════════════════════════════════════
   TOP NAV BAR
══════════════════════════════════════════ */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(15,15,15,0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 14px 36px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    position: sticky;
    top: 0;
    z-index: 999;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.topbar-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #e8a020, #f5c842);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    box-shadow: 0 0 16px rgba(232,160,32,0.4);
}
.topbar-title {
    font-size: 17px;
    font-weight: 750;
    background: linear-gradient(90deg, #ffffff 0%, #cccccc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.4px;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,60,60,0.12);
    border: 1px solid rgba(255,60,60,0.35);
    border-radius: 20px;
    padding: 3px 10px 3px 8px;
    font-size: 10.5px;
    font-weight: 700;
    color: #ff5555;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.live-dot {
    width: 7px; height: 7px;
    background: #ff4040;
    border-radius: 50%;
    box-shadow: 0 0 6px #ff4040;
    animation: livepulse 1.4s ease-in-out infinite;
}
@keyframes livepulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.3; transform:scale(0.7); }
}

/* ══════════════════════════════════════════
   PAGE WRAP
══════════════════════════════════════════ */
.page-wrap {
    padding: 30px 36px 60px;
    max-width: 1400px;
    margin: 0 auto;
}

/* ══════════════════════════════════════════
   GLASS CARD
══════════════════════════════════════════ */
.glass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    padding: 24px 26px;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.glass:hover {
    transform: translateY(-2px);
    border-color: rgba(232,160,32,0.25);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35), 0 0 0 1px rgba(232,160,32,0.08);
}

/* ══════════════════════════════════════════
   STAT CARDS
══════════════════════════════════════════ */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.stat-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(232,160,32,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.4), 0 0 0 1px rgba(232,160,32,0.15);
}
.stat-val {
    font-size: 40px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 8px;
    letter-spacing: -1.5px;
}
.stat-val.gold {
    background: linear-gradient(135deg, #e8a020 0%, #f5c842 50%, #e8a020 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 8px rgba(232,160,32,0.4));
}
.stat-val.white { color: #ffffff; }
.stat-label {
    font-size: 13px;
    color: rgba(255,255,255,0.45);
    font-weight: 400;
    letter-spacing: 0.2px;
}

/* ══════════════════════════════════════════
   SECTION LABEL
══════════════════════════════════════════ */
.sec-label {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: rgba(255,255,255,0.35);
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* ══════════════════════════════════════════
   WIN PROBABILITY BARS
══════════════════════════════════════════ */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 7px 0;
    border-radius: 8px;
    transition: background 0.2s;
}
.prob-row:hover { background: rgba(255,255,255,0.03); }
.prob-rank {
    font-size: 11px;
    color: rgba(255,255,255,0.25);
    min-width: 18px;
    font-weight: 600;
}
.prob-badge { font-size: 15px; min-width: 22px; }
.prob-team {
    font-size: 14px;
    font-weight: 500;
    color: #e8e8e8;
    min-width: 96px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prob-bar-bg {
    flex: 1;
    height: 7px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
    position: relative;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    position: relative;
    background: linear-gradient(90deg, #c47a10 0%, #e8a020 50%, #f5c842 100%);
    box-shadow: 0 0 10px rgba(232,160,32,0.5), 0 0 4px rgba(245,200,66,0.4);
    animation: barSlide 0.9s cubic-bezier(0.34,1.3,0.64,1) both;
}
@keyframes barSlide {
    from { width: 0% !important; opacity:0; }
}
.prob-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: shimmer 2.5s 1s infinite;
}
@keyframes shimmer {
    from { left: -60%; }
    to   { left: 140%;  }
}
.prob-pct {
    font-size: 13px;
    font-weight: 700;
    color: #e0e0e0;
    min-width: 44px;
    text-align: right;
}

/* ══════════════════════════════════════════
   UPCOMING MATCHES
══════════════════════════════════════════ */
.match-item {
    display: flex;
    align-items: center;
    padding: 14px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    transition: background 0.2s;
    border-radius: 8px;
    padding: 12px 10px;
    margin: 0 -10px;
}
.match-item:last-child { border-bottom: none; }
.match-item:hover { background: rgba(255,255,255,0.03); }
.match-home {
    font-size: 14px;
    font-weight: 650;
    color: #e8e8e8;
    flex: 2;
}
.match-sep {
    font-size: 11px;
    color: rgba(255,255,255,0.25);
    flex: 1;
    text-align: center;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.match-away {
    font-size: 14px;
    font-weight: 650;
    color: #e8e8e8;
    flex: 2;
    text-align: right;
}
.match-prob-col {
    text-align: right;
    min-width: 72px;
}
.match-prob-val {
    font-size: 16px;
    font-weight: 800;
    background: linear-gradient(135deg,#e8a020,#f5c842);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.match-prob-tag {
    font-size: 10px;
    color: rgba(255,255,255,0.3);
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ══════════════════════════════════════════
   RESULT CARDS (predictor)
══════════════════════════════════════════ */
.result-grid {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 12px;
    margin-top: 16px;
}
.result-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px 12px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.result-card:hover { transform: translateY(-2px); }
.result-pct {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.result-pct.win  { color: #4ecb87; text-shadow: 0 0 20px rgba(78,203,135,0.4); }
.result-pct.draw { color: #e8a020; text-shadow: 0 0 20px rgba(232,160,32,0.4); }
.result-pct.loss { color: #e05050; text-shadow: 0 0 20px rgba(224,80,80,0.4); }
.result-label {
    font-size: 11px;
    color: rgba(255,255,255,0.4);
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ══════════════════════════════════════════
   PAGE HEADING
══════════════════════════════════════════ */
.page-heading {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.8px;
    background: linear-gradient(90deg, #ffffff 0%, #aaaaaa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px;
}
.page-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.38);
    margin-bottom: 26px;
    font-weight: 400;
}

/* ══════════════════════════════════════════
   BRACKET GROUP CARDS
══════════════════════════════════════════ */
.grp-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 18px;
    transition: transform 0.25s, border-color 0.25s;
}
.grp-card:hover {
    transform: translateY(-2px);
    border-color: rgba(232,160,32,0.2);
}
.grp-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    margin-bottom: 12px;
}
.grp-team-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 0;
}
.grp-team-name {
    font-size: 13px;
    font-weight: 500;
}
.grp-team-name.q { color: #e0e0e0; }
.grp-team-name.x { color: rgba(255,255,255,0.3); }
.grp-team-pct {
    font-size: 11px;
    font-weight: 600;
    color: rgba(255,255,255,0.3);
}
.grp-qualify-dot {
    width: 6px; height: 6px;
    background: #e8a020;
    border-radius: 50%;
    box-shadow: 0 0 6px rgba(232,160,32,0.7);
    display: inline-block;
    margin-right: 7px;
    flex-shrink: 0;
}

/* ══════════════════════════════════════════
   COUNTDOWN
══════════════════════════════════════════ */
.countdown-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.countdown-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(232,160,32,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.countdown-num {
    font-size: 88px;
    font-weight: 900;
    letter-spacing: -4px;
    line-height: 1;
    background: linear-gradient(135deg, #e8a020 0%, #f5c842 40%, #e8a020 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 24px rgba(232,160,32,0.35));
    margin-bottom: 12px;
}
.countdown-label {
    font-size: 16px;
    color: rgba(255,255,255,0.45);
    font-weight: 400;
    margin-bottom: 6px;
}
.countdown-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.22);
}

/* ══════════════════════════════════════════
   SCHEDULE ROWS
══════════════════════════════════════════ */
.sched-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 10px;
    border-radius: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    transition: background 0.2s;
}
.sched-row:last-child { border-bottom: none; }
.sched-row:hover { background: rgba(255,255,255,0.03); }
.sched-date {
    font-size: 10.5px;
    color: rgba(255,255,255,0.3);
    font-weight: 600;
    min-width: 48px;
    letter-spacing: 0.3px;
}
.sched-home {
    font-size: 13.5px;
    font-weight: 650;
    color: #e0e0e0;
    flex: 2;
}
.sched-vs {
    font-size: 11px;
    color: rgba(255,255,255,0.2);
    text-align: center;
    flex: 0.6;
}
.sched-away {
    font-size: 13.5px;
    font-weight: 650;
    color: #e0e0e0;
    flex: 2;
    text-align: right;
}
.sched-fav {
    min-width: 80px;
    text-align: right;
}
.sched-fav-pct {
    font-size: 14px;
    font-weight: 700;
    color: #e8a020;
}
.sched-fav-tag {
    font-size: 10px;
    color: rgba(255,255,255,0.25);
    font-weight: 500;
}
.sched-venue {
    font-size: 10px;
    color: rgba(255,255,255,0.18);
    min-width: 110px;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ══════════════════════════════════════════
   STREAMLIT OVERRIDES
══════════════════════════════════════════ */
/* Buttons */
.stButton > button {
    background: rgba(255,255,255,0.06) !important;
    color: #e0e0e0 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 9px 20px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.1px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e8a020, #f5c842) !important;
    color: #111 !important;
    border-color: transparent !important;
    box-shadow: 0 0 20px rgba(232,160,32,0.4) !important;
    transform: translateY(-1px) !important;
}
/* Selectbox */
.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e0e0e0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}
/* Metric (hidden) */
[data-testid="stMetric"] { display: none !important; }
/* Divider */
hr { border: none; border-top: 1px solid rgba(255,255,255,0.07) !important; margin: 20px 0 !important; }
/* Plotly chart */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════
TEAMS = [
    "Argentina", "France", "Brazil", "England", "Spain", "Germany",
    "Portugal", "Netherlands", "Belgium", "Croatia", "Uruguay",
    "Colombia", "Mexico", "United States", "Japan", "South Korea",
    "Morocco", "Senegal", "Switzerland", "Sweden", "Norway",
    "Turkey", "Australia", "Ecuador", "Canada", "Iran",
    "Saudi Arabia", "Ivory Coast", "Ghana", "Tunisia", "Algeria",
    "Scotland", "South Africa", "Czechia", "Bosnia-Herzegovina",
    "Curaçao", "Cape Verde Islands", "Congo DR", "Uzbekistan",
    "Haiti", "Jordan", "Iraq", "Paraguay", "Qatar", "New Zealand",
    "Egypt", "Austria", "Poland",
]

FLAGS = {
    "Argentina":"🇦🇷","France":"🇫🇷","Brazil":"🇧🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Spain":"🇪🇸","Germany":"🇩🇪","Portugal":"🇵🇹","Netherlands":"🇳🇱",
    "Belgium":"🇧🇪","Croatia":"🇭🇷","Uruguay":"🇺🇾","Colombia":"🇨🇴",
    "Mexico":"🇲🇽","United States":"🇺🇸","Japan":"🇯🇵","South Korea":"🇰🇷",
    "Morocco":"🇲🇦","Senegal":"🇸🇳","Switzerland":"🇨🇭","Sweden":"🇸🇪",
    "Norway":"🇳🇴","Turkey":"🇹🇷","Australia":"🇦🇺","Ecuador":"🇪🇨",
    "Canada":"🇨🇦","Iran":"🇮🇷","Saudi Arabia":"🇸🇦","Ivory Coast":"🇨🇮",
    "Ghana":"🇬🇭","Tunisia":"🇹🇳","Algeria":"🇩🇿","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "South Africa":"🇿🇦","Czechia":"🇨🇿","Bosnia-Herzegovina":"🇧🇦",
    "Curaçao":"🇨🇼","Cape Verde Islands":"🇨🇻","Congo DR":"🇨🇩",
    "Uzbekistan":"🇺🇿","Haiti":"🇭🇹","Jordan":"🇯🇴","Iraq":"🇮🇶",
    "Paraguay":"🇵🇾","Qatar":"🇶🇦","New Zealand":"🇳🇿","Egypt":"🇪🇬",
    "Austria":"🇦🇹","Poland":"🇵🇱",
}

ABBR = {
    "Argentina":"ARG","France":"FRA","Brazil":"BRA","England":"ENG",
    "Spain":"ESP","Germany":"GER","Portugal":"POR","Netherlands":"NED",
    "Belgium":"BEL","Croatia":"CRO","Uruguay":"URU","Colombia":"COL",
    "Mexico":"MEX","United States":"USA","Japan":"JPN","South Korea":"KOR",
    "Morocco":"MAR","Senegal":"SEN","Switzerland":"SUI","Sweden":"SWE",
    "Norway":"NOR","Turkey":"TUR","Australia":"AUS","Ecuador":"ECU",
    "Canada":"CAN","Iran":"IRN","Saudi Arabia":"KSA","Ivory Coast":"CIV",
    "Ghana":"GHA","Tunisia":"TUN","Algeria":"ALG","Scotland":"SCO",
    "South Africa":"RSA","Czechia":"CZE","Bosnia-Herzegovina":"BIH",
    "Curaçao":"CUW","Cape Verde Islands":"CPV","Congo DR":"COD",
    "Uzbekistan":"UZB","Haiti":"HAI","Jordan":"JOR","Iraq":"IRQ",
    "Paraguay":"PAR","Qatar":"QAT","New Zealand":"NZL","Egypt":"EGY",
    "Austria":"AUT","Poland":"POL",
}

TEAM_STATS = {
    "Argentina":          {"win_rate":0.55,"goals_scored":1.90,"goals_conceded":1.01},
    "France":             {"win_rate":0.51,"goals_scored":1.83,"goals_conceded":1.29},
    "Brazil":             {"win_rate":0.63,"goals_scored":2.18,"goals_conceded":0.90},
    "England":            {"win_rate":0.57,"goals_scored":2.18,"goals_conceded":0.96},
    "Spain":              {"win_rate":0.59,"goals_scored":2.05,"goals_conceded":0.90},
    "Germany":            {"win_rate":0.58,"goals_scored":2.25,"goals_conceded":1.16},
    "Portugal":           {"win_rate":0.50,"goals_scored":1.76,"goals_conceded":1.12},
    "Netherlands":        {"win_rate":0.51,"goals_scored":2.09,"goals_conceded":1.22},
    "Belgium":            {"win_rate":0.45,"goals_scored":1.81,"goals_conceded":1.51},
    "Croatia":            {"win_rate":0.53,"goals_scored":1.74,"goals_conceded":1.01},
    "Uruguay":            {"win_rate":0.44,"goals_scored":1.58,"goals_conceded":1.21},
    "Colombia":           {"win_rate":0.40,"goals_scored":1.29,"goals_conceded":1.16},
    "Mexico":             {"win_rate":0.51,"goals_scored":1.76,"goals_conceded":1.05},
    "United States":      {"win_rate":0.44,"goals_scored":1.52,"goals_conceded":1.33},
    "Japan":              {"win_rate":0.49,"goals_scored":1.83,"goals_conceded":1.15},
    "South Korea":        {"win_rate":0.53,"goals_scored":1.78,"goals_conceded":0.91},
    "Morocco":            {"win_rate":0.50,"goals_scored":1.49,"goals_conceded":0.80},
    "Senegal":            {"win_rate":0.47,"goals_scored":1.38,"goals_conceded":0.94},
    "Switzerland":        {"win_rate":0.36,"goals_scored":1.49,"goals_conceded":1.64},
    "Sweden":             {"win_rate":0.49,"goals_scored":1.97,"goals_conceded":1.29},
    "Norway":             {"win_rate":0.38,"goals_scored":1.55,"goals_conceded":1.62},
    "Turkey":             {"win_rate":0.40,"goals_scored":1.40,"goals_conceded":1.41},
    "Australia":          {"win_rate":0.51,"goals_scored":2.01,"goals_conceded":1.07},
    "Ecuador":            {"win_rate":0.31,"goals_scored":1.21,"goals_conceded":1.53},
    "Canada":             {"win_rate":0.38,"goals_scored":1.30,"goals_conceded":1.31},
    "Iran":               {"win_rate":0.57,"goals_scored":1.89,"goals_conceded":0.79},
    "Saudi Arabia":       {"win_rate":0.47,"goals_scored":1.54,"goals_conceded":1.05},
    "Ivory Coast":        {"win_rate":0.51,"goals_scored":1.64,"goals_conceded":1.00},
    "Ghana":              {"win_rate":0.46,"goals_scored":1.57,"goals_conceded":1.04},
    "Tunisia":            {"win_rate":0.44,"goals_scored":1.44,"goals_conceded":1.01},
    "Algeria":            {"win_rate":0.47,"goals_scored":1.54,"goals_conceded":1.00},
    "Scotland":           {"win_rate":0.47,"goals_scored":1.71,"goals_conceded":1.24},
    "South Africa":       {"win_rate":0.45,"goals_scored":1.35,"goals_conceded":0.95},
    "Czechia":            {"win_rate":0.48,"goals_scored":1.85,"goals_conceded":1.24},
    "Bosnia-Herzegovina": {"win_rate":0.37,"goals_scored":1.37,"goals_conceded":1.37},
    "Curaçao":            {"win_rate":0.37,"goals_scored":1.64,"goals_conceded":1.58},
    "Cape Verde Islands": {"win_rate":0.38,"goals_scored":1.10,"goals_conceded":1.12},
    "Congo DR":           {"win_rate":0.39,"goals_scored":1.49,"goals_conceded":1.20},
    "Uzbekistan":         {"win_rate":0.48,"goals_scored":1.73,"goals_conceded":1.11},
    "Haiti":              {"win_rate":0.42,"goals_scored":1.64,"goals_conceded":1.34},
    "Jordan":             {"win_rate":0.37,"goals_scored":1.28,"goals_conceded":1.11},
    "Iraq":               {"win_rate":0.47,"goals_scored":1.54,"goals_conceded":0.93},
    "Paraguay":           {"win_rate":0.35,"goals_scored":1.28,"goals_conceded":1.42},
    "Qatar":              {"win_rate":0.41,"goals_scored":1.41,"goals_conceded":1.21},
    "New Zealand":        {"win_rate":0.41,"goals_scored":1.76,"goals_conceded":1.50},
    "Egypt":              {"win_rate":0.50,"goals_scored":1.62,"goals_conceded":1.01},
    "Austria":            {"win_rate":0.43,"goals_scored":1.79,"goals_conceded":1.53},
    "Poland":             {"win_rate":0.43,"goals_scored":1.68,"goals_conceded":1.35},
}
DEFAULT_STATS = {"win_rate":0.40,"goals_scored":1.1,"goals_conceded":1.3}

GROUPS = {
    "A": ["Mexico", "South Korea", "Czechia", "South Africa"],
    "B": ["Canada", "Qatar", "Switzerland", "Bosnia-Herzegovina"],
    "C": ["United States", "Paraguay", "Australia", "Turkey"],
    "D": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Spain", "Saudi Arabia", "Uruguay", "Cape Verde Islands"],
    "H": ["Belgium", "Iran", "New Zealand", "Egypt"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Poland"],
}

def get_stats(t): return TEAM_STATS.get(t, DEFAULT_STATS)

@st.cache_data
def load_win_probs():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base,"data","processed","wc2026_predictions.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    tot = sum(v["win_rate"] for v in TEAM_STATS.values())
    return {t: s["win_rate"]/tot for t,s in TEAM_STATS.items()}

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mp = os.path.join(base,"models","best_model.pkl")
    fp = os.path.join(base,"models","feature_cols.pkl")
    if os.path.exists(mp) and os.path.exists(fp):
        with open(mp,"rb") as f: model = pickle.load(f)
        with open(fp,"rb") as f: feat  = pickle.load(f)
        return model, feat
    return None, None

def predict_match(home, away, is_knockout=0):
    model, _ = load_model()
    hs = get_stats(home); as_ = get_stats(away)
    if model is None:
        hw = hs["win_rate"]/(hs["win_rate"]+as_["win_rate"])*0.70
        aw = as_["win_rate"]/(hs["win_rate"]+as_["win_rate"])*0.70
        d  = max(1-hw-aw, 0.05)
    else:
        features = pd.DataFrame([{
            "home_win_rate": hs["win_rate"],
            "home_goals_scored_avg": hs["goals_scored"],
            "home_goals_conceded_avg": hs["goals_conceded"],
            "away_win_rate": as_["win_rate"],
            "away_goals_scored_avg": as_["goals_scored"],
            "away_goals_conceded_avg": as_["goals_conceded"],
            "h2h_home_win_rate": 0.5, "h2h_matches": 3,
            "goal_diff_avg": (hs["goals_scored"]-hs["goals_conceded"])-(as_["goals_scored"]-as_["goals_conceded"]),
            "win_rate_diff": hs["win_rate"]-as_["win_rate"],
            "is_knockout": is_knockout,
            "is_neutral_venue": 1,
        }])
        probs = model.predict_proba(features)[0]
        cm    = {c:i for i,c in enumerate(model.classes_)}
        hw, d, aw = probs[cm[2]], probs[cm[1]], probs[cm[0]]

    # Knockout — no draws
    if is_knockout:
        hw = hw + d * 0.5
        aw = aw + d * 0.5
        d  = 0.0

    return hw, d, aw

WIN_PROBS = load_win_probs()
BADGES    = {1:"🥇", 2:"🥈", 3:"🥉"}

# ══════════════════════════════════════════════════════════════════════════════
#  STATE & NAV
# ══════════════════════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "Standings"

# ── Top bar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-icon">⚽</div>
    <span class="topbar-title">FIFA WC 2026 Predictor</span>
    <span class="live-badge"><span class="live-dot"></span>Live</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Nav buttons (overlaid below topbar via st.columns)
nav_spacer, nav_area = st.columns([5, 5])
with nav_area:
    n1,n2,n3,n4 = st.columns(4)
    if n1.button("⚽ Standings", key="n_s", use_container_width=True): st.session_state.page="Standings"
    if n2.button("🔮 Predict",   key="n_p", use_container_width=True): st.session_state.page="Predict"
    if n3.button("🏆 Bracket",   key="n_b", use_container_width=True): st.session_state.page="Bracket"
    if n4.button("📡 Live",      key="n_l", use_container_width=True): st.session_state.page="Live"

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — STANDINGS
# ══════════════════════════════════════════════════════════════════════════════
if page == "Standings":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    # ── Stat cards ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-val gold">48</div>
        <div class="stat-label">Teams in tournament</div>
      </div>
      <div class="stat-card">
        <div class="stat-val white">104</div>
        <div class="stat-label">Total matches</div>
      </div>
      <div class="stat-card">
        <div class="stat-val gold">10K</div>
        <div class="stat-label">Monte Carlo simulations</div>
      </div>
      <div class="stat-card">
        <div class="stat-val white">0</div>
        <div class="stat-label">Matches played</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout ────────────────────────────────────────────────────
    left, right = st.columns([11, 10], gap="medium")

    # ── Left: Win Probability bars ──
    with left:
        sorted_teams = sorted(WIN_PROBS.items(), key=lambda x: x[1], reverse=True)
        top8         = sorted_teams[:8]
        max_p        = top8[0][1]

        html = '<div class="glass">'
        html += '<div class="sec-label">🏆 Win Probability — Top 8</div>'
        for rank, (team, prob) in enumerate(top8, 1):
            bw   = int((prob/max_p)*100)
            badge= BADGES.get(rank, "")
            flag = FLAGS.get(team,"🏳")
            html += f"""
            <div class="prob-row">
              <div class="prob-rank">{rank}</div>
              <div class="prob-badge">{badge or flag}</div>
              <div class="prob-team">{team}</div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{bw}%"></div>
              </div>
              <div class="prob-pct">{prob*100:.1f}%</div>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    # ── Right: Upcoming + Predictor ──
    with right:
        # Upcoming matches
        try:
            _api_key = st.secrets["FOOTBALL_API_KEY"]
            _headers = {"X-Auth-Token": _api_key}
            _r = __import__("requests").get(
                "https://api.football-data.org/v4/competitions/2000/matches?status=SCHEDULED",
                headers=_headers, timeout=5
            )
            _matches = _r.json().get("matches", [])[:3]
            upcoming = []
            for m in _matches:
                _home = m["homeTeam"]["name"]
                _away = m["awayTeam"]["name"]
                _hw, _d, _aw = predict_match(_home, _away, 0)
                _fav = _home if _hw > _aw else _away
                upcoming.append((_home, _away, max(_hw,_aw), ABBR.get(_fav, _fav[:3].upper())))
        except:
            upcoming = [
                ("Mexico","South Africa",0.58,"MEX"),
                ("USA","Paraguay",0.52,"USA"),
                ("Brazil","Morocco",0.71,"BRA"),
            ]
        html2 = '<div class="glass" style="margin-bottom:16px;">'
        html2 += '<div class="sec-label">📅 Upcoming Matches</div>'
        for home, away, prob, winner in upcoming:
            hf = FLAGS.get(home,"🏳"); af = FLAGS.get(away,"🏳")
            html2 += f"""
            <div class="match-item">
              <div class="match-home">{hf} {home}</div>
              <div class="match-sep">vs</div>
              <div class="match-away">{af} {away}</div>
              <div class="match-prob-col">
                <div class="match-prob-val">{int(prob*100)}%</div>
                <div class="match-prob-tag">{winner}</div>
              </div>
            </div>"""
        html2 += "</div>"
        st.markdown(html2, unsafe_allow_html=True)

        # Quick predictor
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">⚡ Match Predictor</div>', unsafe_allow_html=True)
        qc1, qc2, qc3 = st.columns([5,1,5])
        home_t = qc1.selectbox("", TEAMS, index=0, key="q_home",
                               format_func=lambda x: f"{FLAGS.get(x,'🏳')} {x}",
                               label_visibility="collapsed")
        qc2.markdown("<p style='text-align:center;color:rgba(255,255,255,0.25);padding-top:9px;'>vs</p>",
                     unsafe_allow_html=True)
        away_t = qc3.selectbox("", TEAMS, index=3, key="q_away",
                               format_func=lambda x: f"{FLAGS.get(x,'🏳')} {x}",
                               label_visibility="collapsed")
        if st.button("⚽  Predict Outcome", key="q_pred", use_container_width=True):
            hw, d, aw = predict_match(home_t, away_t)
            ah = ABBR.get(home_t, home_t[:3].upper())
            aa = ABBR.get(away_t, away_t[:3].upper())
            st.markdown(f"""
            <div class="result-grid">
              <div class="result-card">
                <div class="result-pct win">{hw*100:.1f}%</div>
                <div class="result-label">{ah} win</div>
              </div>
              <div class="result-card">
                <div class="result-pct draw">{d*100:.1f}%</div>
                <div class="result-label">Draw</div>
              </div>
              <div class="result-card">
                <div class="result-pct loss">{aw*100:.1f}%</div>
                <div class="result-label">{aa} win</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">Match Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">ML-powered win/draw/loss probabilities for any WC 2026 matchup</div>', unsafe_allow_html=True)

    pc1, pc2, pc3, pc4 = st.columns([4,1,4,2])
    home_t = pc1.selectbox("Team 1", TEAMS, index=0, key="p_home",
                           format_func=lambda x: f"{FLAGS.get(x,'🏳')} {x}")
    pc2.markdown("<p style='text-align:center;color:rgba(255,255,255,0.2);padding-top:36px;'>vs</p>",
                 unsafe_allow_html=True)
    away_t = pc3.selectbox("Team 2", TEAMS, index=3, key="p_away",
                           format_func=lambda x: f"{FLAGS.get(x,'🏳')} {x}")
    stage  = pc4.selectbox("Stage", ["Group Stage","Knockout"], key="p_stage")
    is_ko  = stage == "Knockout"

    st.caption("ℹ️ WC 2026 is played at neutral venues in USA, Canada & Mexico. Team order follows the official fixture draw.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("⚽  Predict Match", key="full_pred", use_container_width=False):
        hw, d, aw = predict_match(home_t, away_t, int(is_ko))
        hf = FLAGS.get(home_t,"🏳"); af = FLAGS.get(away_t,"🏳")
        ah = ABBR.get(home_t,home_t[:3].upper()); aa = ABBR.get(away_t,away_t[:3].upper())
        fav = home_t if hw > aw else away_t

        col_res, col_chart = st.columns([1,1], gap="large")
        with col_res:
            st.markdown(f"""
            <div class="glass">
              <div style="text-align:center;padding-bottom:20px;border-bottom:1px solid rgba(255,255,255,0.07);">
                <span style="font-size:22px;font-weight:800;color:#fff;">{hf} {home_t}</span>
                <span style="color:rgba(255,255,255,0.2);margin:0 14px;font-size:14px;">vs</span>
                <span style="font-size:22px;font-weight:800;color:#fff;">{af} {away_t}</span>
              </div>
              <div class="result-grid" style="margin-top:20px;">
                <div class="result-card">
                  <div class="result-pct win">{hw*100:.1f}%</div>
                  <div class="result-label">{ah} win</div>
                </div>
                <div class="result-card">
                  <div class="result-pct draw">{d*100:.1f}%</div>
                  <div class="result-label">Draw</div>
                </div>
                <div class="result-card">
                  <div class="result-pct loss">{aw*100:.1f}%</div>
                  <div class="result-label">{aa} win</div>
                </div>
              </div>
              <div style="margin-top:18px;padding:12px 16px;background:rgba(232,160,32,0.06);
                   border:1px solid rgba(232,160,32,0.2);border-radius:10px;text-align:center;">
                <span style="font-size:12px;color:rgba(255,255,255,0.4);">Predicted winner · </span>
                <span style="font-size:14px;font-weight:750;color:#e8a020;">
                  {FLAGS.get(fav,'')} {fav}
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            labels = [f"{ah} Win","Draw",f"{aa} Win"]
            values = [hw*100, d*100, aw*100]
            colors = ["#4ecb87","#e8a020","#e05050"]
            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.60,
                marker=dict(
                    colors=colors,
                    line=dict(color="rgba(0,0,0,0)", width=0)
                ),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0),
                height=260,
                legend=dict(
                    font=dict(color="#aaa", size=12, family="Inter"),
                    orientation="h", x=0.5, xanchor="center", y=-0.05,
                ),
                annotations=[dict(
                    text=f"<b>{max(hw,d,aw)*100:.0f}%</b>",
                    x=0.5, y=0.5, font=dict(size=28, color="#fff", family="Inter"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown("<hr>", unsafe_allow_html=True)

    # Full standings
    st.markdown('<div class="sec-label" style="margin-top:16px;">All 48 teams — tournament win probability</div>',
                unsafe_allow_html=True)
    sorted_all = sorted(WIN_PROBS.items(), key=lambda x: x[1], reverse=True)
    max_p2     = sorted_all[0][1]

    html3 = '<div class="glass">'
    for rank, (team, prob) in enumerate(sorted_all, 1):
        bw    = int((prob/max_p2)*100)
        flag  = FLAGS.get(team,"🏳")
        badge = BADGES.get(rank,"")
        html3 += f"""
        <div class="prob-row">
          <div class="prob-rank">{rank}</div>
          <div class="prob-badge">{badge or flag}</div>
          <div class="prob-team" style="min-width:150px;">{team}</div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{bw}%"></div>
          </div>
          <div class="prob-pct">{prob*100:.1f}%</div>
        </div>"""
    html3 += "</div>"
    st.markdown(html3, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — BRACKET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Bracket":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">Tournament Groups</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">48 teams · 12 groups · Top 2 per group advance to the knockout stage</div>',
                unsafe_allow_html=True)

    # Top-8 favourites bar first
    top5_sorted = sorted(WIN_PROBS.items(), key=lambda x: x[1], reverse=True)[:5]
    labels_b = [f"{FLAGS.get(t,'🏳')} {t}" for t,_ in top5_sorted]
    values_b = [p*100 for _,p in top5_sorted]
    fig_bar = go.Figure(go.Bar(
        x=labels_b, y=values_b,
        marker=dict(
            color=values_b,
            colorscale=[[0,"#7a4a10"],[0.5,"#e8a020"],[1,"#f5c842"]],
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in values_b],
        textposition="outside",
        textfont=dict(color="#e0e0e0", size=12, family="Inter"),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=16,b=0),
        height=200,
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        xaxis=dict(showgrid=False, tickfont=dict(color="#aaa", size=12, family="Inter")),
        bargap=0.35,
    )
    fig_bar.update_traces(marker_line_width=0)
    st.markdown('<div class="glass" style="margin-bottom:24px;">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Top 5 favourites to lift the trophy</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Group cards
    group_keys = sorted(GROUPS.keys())
    rows_of_4  = [group_keys[i:i+4] for i in range(0,len(group_keys),4)]
    for row in rows_of_4:
        cols = st.columns(4, gap="small")
        for col, grp in zip(cols, row):
            with col:
                ranked = sorted(GROUPS[grp], key=lambda t: WIN_PROBS.get(t,0), reverse=True)
                html4 = '<div class="grp-card">'
                html4 += f'<div class="grp-label">Group {grp}</div>'
                for i, t in enumerate(ranked):
                    prob = WIN_PROBS.get(t,0)
                    flag = FLAGS.get(t,"🏳")
                    dot  = '<span class="grp-qualify-dot"></span>' if i<2 else '<span style="display:inline-block;width:13px;"></span>'
                    cls  = "q" if i<2 else "x"
                    sep  = 'border-bottom:1px solid rgba(255,255,255,0.05);' if i<3 else ''
                    html4 += f"""
                    <div class="grp-team-row" style="{sep}">
                      <div style="display:flex;align-items:center;">
                        {dot}
                        <span class="grp-team-name {cls}">{flag} {t}</span>
                      </div>
                      <span class="grp-team-pct">{prob*100:.1f}%</span>
                    </div>"""
                html4 += "</div>"
                st.markdown(html4, unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px;padding:14px 18px;
         background:rgba(232,160,32,0.05);border:1px solid rgba(232,160,32,0.15);
         border-radius:12px;font-size:13px;color:rgba(255,255,255,0.4);">
      🟡 <strong style="color:rgba(255,255,255,0.6);">Gold dots</strong> = predicted group qualifiers
      based on 10,000 Monte Carlo simulations.
      Win % shown is full-tournament win probability.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — LIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="page-heading">Live & Schedule</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">WC 2026 kicks off June 11 at MetLife Stadium, New York</div>',
                unsafe_allow_html=True)

    today      = datetime.date.today()
    start_date = datetime.date(2026, 6, 11)
    delta      = (start_date - today).days

    if delta > 0:
        st.markdown(f"""
        <div class="countdown-card">
          <div class="countdown-num">{delta}</div>
          <div class="countdown-label">days until kick-off</div>
          <div class="countdown-sub">June 11, 2026 · MetLife Stadium · East Rutherford, NJ</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        live_path = os.path.join(base,"data","processed","live_matches.csv")
        if os.path.exists(live_path):
            df = pd.read_csv(live_path)
            for _, row in df[df["status"] == "FINISHED"].iterrows():
                hf = FLAGS.get(str(row.get("home_team","")), "🏳")
                af = FLAGS.get(str(row.get("away_team","")), "🏳")
                st.markdown(f"""
                <div class="glass" style="margin-bottom:10px;display:flex;align-items:center;gap:20px;">
                  <span style="font-size:16px;font-weight:700;">{hf} {row.get('home_team','')}</span>
                  <span style="font-size:22px;font-weight:900;color:#e8a020;">
                    {int(row.get('home_goals',0)) if pd.notna(row.get('home_goals',0)) else 0} – {int(row.get('away_goals',0)) if pd.notna(row.get('away_goals',0)) else 0}
                  </span>
                  <span style="font-size:16px;font-weight:700;">{af} {row.get('away_team','')}</span>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # Load live fixtures from updater
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    live_path = os.path.join(base, "data", "processed", "live_matches.csv")
    if os.path.exists(live_path):
        df_sched = pd.read_csv(live_path)
        df_sched = df_sched[df_sched["status"].isin(["SCHEDULED", "TIMED", "IN_PLAY", "FINISHED"])]
        df_sched["date_fmt"] = pd.to_datetime(df_sched["date"]).dt.strftime("%-d %b")  # "11 Jun"
        
        html5 = '<div class="glass">'
        html5 += '<div class="sec-label">🗓 Opening Fixtures — Predicted Outcomes</div>'
        for _, row in df_sched.head(9).iterrows():
            home, away = str(row["home_team"]), str(row["away_team"])
            hf = FLAGS.get(home, "🏳"); af = FLAGS.get(away, "🏳")
            hw, d, aw = predict_match(home, away)
            fav   = home if hw > aw else away
            fav_p = max(hw, aw) * 100
            fav_ab = ABBR.get(fav, fav[:3].upper())
            html5 += f"""
            <div class="sched-row">
            <div class="sched-date">{row['date_fmt']}</div>
            <div class="sched-home">{hf} {home}</div>
            <div class="sched-vs">vs</div>
            <div class="sched-away">{af} {away}</div>
            <div class="sched-fav">
                <div class="sched-fav-pct">{fav_p:.0f}%</div>
                <div class="sched-fav-tag">{fav_ab} fav</div>
            </div>
            </div>"""
        html5 += "</div>"
        st.markdown(html5, unsafe_allow_html=True)
    else:
        st.warning("Run `python src/updater.py` to load live fixtures.")
 