import html

import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Doodle Tunes",
    page_icon="🎧",
    layout="wide",
)


# =========================================================
# CSS
# =========================================================

st.html(
    """
<style>

:root {
    --ink: #433746;
    --cream: #fffaf5;
    --pink: #f6bfd7;
    --pink-light: #ffe5ef;
    --purple: #ded1ff;
    --purple-light: #f0ebff;
    --blue: #dceeff;
    --yellow: #fff0ad;
    --green: #dff3dd;
    --white: #fffefb;
}


/* ========================================================
   PAGE
======================================================== */

.stApp {
    background:
        radial-gradient(circle at 8% 3%, #ffe3ef 0, transparent 270px),
        radial-gradient(circle at 95% 8%, #e9e2ff 0, transparent 310px),
        radial-gradient(circle at 80% 95%, #fff1bd 0, transparent 290px),
        var(--cream);

    color: var(--ink);
    overflow-x: hidden;
}

.block-container {
    max-width: 96vw !important;
    width: 96vw !important;

    padding-left: 1.4rem !important;
    padding-right: 1.4rem !important;
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;

    position: relative;
    z-index: 5;
}


/* ========================================================
   SIDE DOODLES
======================================================== */

.doodle-side {
    position: fixed;
    top: 65px;

    width: 150px;
    height: calc(100vh - 80px);

    z-index: 1;
    pointer-events: none;

    opacity: 0.9;
}

.doodle-left {
    left: 2px;
}

.doodle-right {
    right: 2px;
}

.doodle-side svg {
    width: 100%;
    height: 100%;
    overflow: visible;
}

.doodle-stroke {
    stroke: var(--ink);
    stroke-width: 3;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.doodle-thin {
    stroke: var(--ink);
    stroke-width: 2.2;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}


/* ========================================================
   TOP BAR
======================================================== */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;

    margin-bottom: 25px;

    max-width: 1500px;
    margin-left: auto;
    margin-right: auto;
}

.logo {
    display: flex;
    align-items: center;
    gap: 11px;

    font-size: 1.25rem;
    font-weight: 900;
    color: var(--ink);
}

.logo-disc {
    width: 38px;
    height: 38px;

    border-radius: 50%;
    background: var(--ink);

    position: relative;

    box-shadow: 3px 3px 0 var(--pink);
}

.logo-disc::after {
    content: "";

    width: 10px;
    height: 10px;

    border-radius: 50%;

    background: var(--pink);

    position: absolute;

    top: 14px;
    left: 14px;
}

.nav-pill {
    background: var(--purple-light);

    border: 2px solid var(--ink);
    border-radius: 999px;

    padding: 7px 13px;

    font-weight: 800;
    font-size: .82rem;
}


/* ========================================================
   HERO
======================================================== */

.hero {
    max-width: 1500px;

    margin-left: auto;
    margin-right: auto;
    margin-bottom: 42px;

    background: var(--white);

    border: 2.5px solid var(--ink);
    border-radius: 34px;

    padding: 40px 44px;

    box-shadow:
        8px 8px 0 var(--pink),
        11px 11px 0 var(--ink);

    position: relative;
    overflow: hidden;
}

.hero-kicker {
    display: inline-block;

    background: var(--yellow);

    border: 2px solid var(--ink);
    border-radius: 999px;

    padding: 6px 13px;

    font-size: .8rem;
    font-weight: 900;

    transform: rotate(-2deg);

    margin-bottom: 18px;
}

.hero h1 {
    margin: 0;

    color: var(--ink);

    font-size: 4.5rem;
    line-height: .95;

    letter-spacing: -3px;

    font-weight: 950;
}

.hero p {
    max-width: 680px;

    color: #77677b;

    font-size: 1.08rem;
    line-height: 1.7;

    margin-top: 18px;
}

.hero-headphones {
    position: absolute;

    right: 55px;
    top: 35px;

    font-size: 4.7rem;

    transform: rotate(9deg);
}

.hero-stars {
    position: absolute;

    right: 180px;
    bottom: 32px;

    font-size: 2.4rem;
}


/* ========================================================
   SECTION HEADINGS
======================================================== */

.section-title {
    font-size: 2.2rem;
    font-weight: 950;

    color: var(--ink);

    margin-bottom: 6px;

    text-align: center;

    letter-spacing: -0.5px;
}

.section-subtitle {
    color: #907f93;

    font-size: 1rem;

    margin-bottom: 20px;

    text-align: center;
}


/* ========================================================
   SONG PICKER
======================================================== */

div[data-baseweb="select"] > div {
    background: #fff !important;

    border: 2px solid var(--ink) !important;
    border-radius: 18px !important;

    min-height: 54px !important;

    box-shadow: 5px 5px 0 var(--purple);

    padding-left: 6px !important;
    padding-right: 10px !important;
}

div[data-baseweb="select"] span {
    color: var(--ink) !important;
    font-weight: 650 !important;
}

div[data-baseweb="select"] svg {
    width: 25px !important;
    height: 25px !important;

    color: var(--ink) !important;
    fill: var(--ink) !important;

    opacity: 1 !important;
}

div[data-baseweb="select"] > div > div:last-child {
    padding-left: 8px !important;
    padding-right: 7px !important;
}


/* ========================================================
   BUTTON
======================================================== */

.stButton > button {
    background: var(--pink) !important;

    color: var(--ink) !important;

    border: 2px solid var(--ink) !important;
    border-radius: 999px !important;

    padding: .8rem 1.7rem !important;

    font-weight: 900 !important;
    font-size: 1rem !important;

    box-shadow: 5px 5px 0 var(--ink) !important;

    transition: .15s ease;
}

.stButton > button:hover {
    transform: translate(-2px, -2px);

    box-shadow: 7px 7px 0 var(--ink) !important;

    background: #f5aeca !important;
}


/* ========================================================
   PLAYER
======================================================== */

.player {
    background: var(--ink);

    color: white;

    border-radius: 28px;

    padding: 25px;

    min-height: 300px;

    position: relative;

    box-shadow:
        7px 7px 0 var(--blue),
        10px 10px 0 var(--ink);
}

.album-art {
    width: 145px;
    height: 145px;

    margin: 0 auto 18px auto;

    border-radius: 24px;

    border: 2px solid #fff;

    background:
        radial-gradient(
            circle at center,

            var(--ink) 0 13%,
            var(--pink) 14% 20%,
            var(--ink) 21% 34%,
            var(--purple) 35% 38%,
            var(--ink) 39% 100%
        );

    box-shadow:
        5px 5px 0 var(--pink);

    position: relative;
}

.album-art::after {
    content: "✦";

    position: absolute;

    right: -18px;
    top: -18px;

    color: var(--yellow);

    font-size: 2rem;
}

.now-playing {
    font-size: .72rem;

    letter-spacing: 1px;

    color: #d9cddd;

    font-weight: 800;
}

.player-song {
    font-size: 1.45rem;

    font-weight: 900;

    margin-top: 7px;
}

.player-artist {
    color: #ddd0df;

    margin-top: 4px;
}

.progress {
    height: 5px;

    border-radius: 999px;

    background: #796e7d;

    margin: 22px 0 10px 0;
}

.progress-fill {
    width: 63%;
    height: 100%;

    background: var(--pink);

    border-radius: 999px;
}

.player-controls {
    text-align: center;

    letter-spacing: 13px;

    margin-top: 12px;

    font-size: 1.15rem;
}


/* ========================================================
   TAGS
======================================================== */

.tags {
    margin-top: 14px;
}

.tag {
    display: inline-block;

    color: var(--ink);

    border: 1.5px solid var(--ink);
    border-radius: 999px;

    padding: 5px 10px;

    margin: 3px;

    font-size: .75rem;

    font-weight: 850;
}

.tag-pink {
    background: var(--pink-light);
}

.tag-yellow {
    background: var(--yellow);
}

.tag-purple {
    background: var(--purple-light);
}

.tag-green {
    background: var(--green);
}

.tag-blue {
    background: var(--blue);
}


/* ========================================================
   RECOMMENDATIONS
======================================================== */

.rec-wrap {
    background: rgba(255,255,255,.55);

    border: 2px dashed #c9b9ca;
    border-radius: 30px;

    padding: 18px;

    position: relative;
}

.rec-wrap::before {
    content: "✎";

    position: absolute;

    right: 18px;
    top: -22px;

    font-size: 2rem;

    transform: rotate(12deg);
}

.rec-card {
    background: var(--white);

    border: 2px solid var(--ink);
    border-radius: 22px;

    margin-bottom: 14px;

    padding: 17px 18px;

    box-shadow:
        4px 4px 0 var(--ink);

    display: flex;
    align-items: center;

    gap: 16px;

    transition: .15s ease;
}

.rec-card:hover {
    transform:
        translateY(-2px)
        rotate(-.25deg);

    box-shadow:
        6px 6px 0 var(--ink);
}

.rec-number {
    min-width: 43px;

    width: 43px;
    height: 43px;

    border-radius: 50%;

    border: 2px solid var(--ink);

    background: var(--yellow);

    display: flex;

    align-items: center;
    justify-content: center;

    font-weight: 950;

    transform: rotate(-5deg);
}

.rec-info {
    flex-grow: 1;
}

.rec-song {
    font-size: 1.08rem;

    font-weight: 950;

    color: var(--ink);
}

.rec-artist {
    font-size: .88rem;

    color: #7b6b7e;

    margin-top: 2px;
}

.rec-heart {
    font-size: 1.25rem;

    padding-right: 5px;
}


/* ========================================================
   EMPTY STATE
======================================================== */

.empty-box {
    border: 2px dashed #c8b8cb;
    border-radius: 26px;

    padding: 60px 20px;

    text-align: center;

    color: #9b899d;

    background: rgba(255,255,255,.42);

    position: relative;
}

.empty-box::after {
    content: "♡";

    position: absolute;

    right: 20px;
    bottom: 10px;

    font-size: 2rem;

    transform: rotate(-12deg);
}

.empty-icon {
    font-size: 3rem;

    margin-bottom: 8px;
}


/* ========================================================
   RESPONSIVE
======================================================== */

@media (max-width: 1450px) {

    .doodle-side {
        width: 95px;
        opacity: .65;
    }

}

@media (max-width: 1200px) {

    .doodle-side {
        display: none;
    }

}

@media (max-width: 800px) {

    .block-container {
        max-width: 98vw !important;
        width: 98vw !important;

        padding-left: .8rem !important;
        padding-right: .8rem !important;
    }

    .hero h1 {
        font-size: 3rem;
    }

    .hero-headphones {
        opacity: .35;
    }

}


/* ========================================================
   STREAMLIT CLEANUP
======================================================== */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
"""
)


# =========================================================
# LEFT SIDE DOODLES
# =========================================================

st.html(
    """
<div class="doodle-side doodle-left">

<svg viewBox="0 0 180 900">

    <g transform="translate(38,70) rotate(-12)">

        <circle
            cx="0"
            cy="-18"
            r="14"
            fill="#f6bfd7"
            stroke="#433746"
            stroke-width="2"
        />

        <circle
            cx="18"
            cy="0"
            r="14"
            fill="#ded1ff"
            stroke="#433746"
            stroke-width="2"
        />

        <circle
            cx="0"
            cy="18"
            r="14"
            fill="#fff0ad"
            stroke="#433746"
            stroke-width="2"
        />

        <circle
            cx="-18"
            cy="0"
            r="14"
            fill="#dff3dd"
            stroke="#433746"
            stroke-width="2"
        />

        <circle
            cx="0"
            cy="0"
            r="10"
            fill="#fffefb"
            stroke="#433746"
            stroke-width="2"
        />

    </g>

    <path
        d="
            M15 180
            C65 135, 90 225, 145 170
            S175 210, 120 235
        "
        class="doodle-stroke"
    />

    <path
        d="
            M50 280
            L58 301
            L80 309
            L58 317
            L50 339
            L42 317
            L20 309
            L42 301
            Z
        "
        fill="#fff0ad"
        stroke="#433746"
        stroke-width="2"
    />

    <g transform="translate(20,390) rotate(-6)">

        <rect
            x="0"
            y="0"
            rx="10"
            width="120"
            height="76"
            fill="#ffe5ef"
            stroke="#433746"
            stroke-width="3"
        />

        <rect
            x="16"
            y="14"
            rx="5"
            width="88"
            height="33"
            fill="#fffefb"
            stroke="#433746"
            stroke-width="2"
        />

        <circle
            cx="40"
            cy="30"
            r="9"
            fill="#ded1ff"
            stroke="#433746"
            stroke-width="2"
        />

        <circle
            cx="80"
            cy="30"
            r="9"
            fill="#fff0ad"
            stroke="#433746"
            stroke-width="2"
        />

    </g>

    <path
        d="
            M55 690
            C34 667, 2 687, 17 716
            C28 737, 55 753, 55 753
            C55 753, 82 737, 93 716
            C108 687, 76 667, 55 690
            Z
        "
        fill="#f6bfd7"
        stroke="#433746"
        stroke-width="2.5"
    />

</svg>

</div>
"""
)


# =========================================================
# RIGHT SIDE DOODLES
# =========================================================

st.html(
    """
<div class="doodle-side doodle-right">

<svg viewBox="0 0 180 900">

    <g transform="translate(35,55) rotate(8)">

        <path
            d="M20 55 C20 5, 100 5, 100 55"
            class="doodle-stroke"
        />

        <rect
            x="10"
            y="48"
            width="22"
            height="45"
            rx="10"
            fill="#ded1ff"
            stroke="#433746"
            stroke-width="2.5"
        />

        <rect
            x="88"
            y="48"
            width="22"
            height="45"
            rx="10"
            fill="#f6bfd7"
            stroke="#433746"
            stroke-width="2.5"
        />

    </g>

    <g transform="translate(95,210)">

        <path
            d="
                M0 -28
                L8 -8
                L28 0
                L8 8
                L0 28
                L-8 8
                L-28 0
                L-8 -8
                Z
            "
            fill="#fff0ad"
            stroke="#433746"
            stroke-width="2"
        />

    </g>

    <g transform="translate(37,330) rotate(-7)">

        <circle
            cx="40"
            cy="40"
            r="38"
            fill="#fff0ad"
            stroke="#433746"
            stroke-width="3"
        />

        <circle
            cx="27"
            cy="31"
            r="4"
            fill="#433746"
        />

        <circle
            cx="53"
            cy="31"
            r="4"
            fill="#433746"
        />

        <path
            d="M24 50 C34 62, 48 62, 58 50"
            class="doodle-thin"
        />

    </g>

    <path
        d="
            M65 650
            L115 650
            L91 688
            L126 688
            L65 760
            L81 708
            L45 708
            Z
        "
        fill="#ded1ff"
        stroke="#433746"
        stroke-width="2.5"
    />

</svg>

</div>
"""
)


# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "data/spotify_tracks_clean.csv"
    )

    data["song_label"] = (
        data["track_name"]
        + " — "
        + data["artists"]
    )

    return data


df = load_data()


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


@st.cache_resource
def prepare_features(data):

    scaler = StandardScaler()

    return scaler.fit_transform(
        data[FEATURES]
    )


X_scaled = prepare_features(df)


# =========================================================
# RECOMMENDER
# =========================================================

def recommend_songs(
    song_label,
    n_recommendations=5,
):

    matches = df[
        df["song_label"]
        == song_label
    ]

    if matches.empty:
        return pd.DataFrame()

    song_index = matches.index[0]

    song_genre = df.loc[
        song_index,
        "track_genre"
    ]

    song_vector = (
        X_scaled[song_index]
        .reshape(1, -1)
    )

    similarities = (
        cosine_similarity(
            song_vector,
            X_scaled,
        )
        .flatten()
    )

    candidates = (
        df[
            df["track_genre"]
            == song_genre
        ]
        .copy()
    )

    candidate_indices = (
        candidates.index
    )

    candidates["similarity"] = (
        similarities[
            candidate_indices
        ]
    )

    candidates["popularity_score"] = (
        candidates["popularity"]
        / 100
    )

    candidates["final_score"] = (
        0.85
        * candidates["similarity"]
        +
        0.15
        * candidates["popularity_score"]
    )

    candidates = candidates[
        candidates["song_label"]
        != song_label
    ]

    return (
        candidates
        .sort_values(
            "final_score",
            ascending=False,
        )
        .head(
            n_recommendations
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# TOP BAR
# =========================================================

st.html(
    """
<div class="topbar">

    <div class="logo">
        <div class="logo-disc"></div>
        Doodle Tunes
    </div>

    <div class="nav-pill">
        ♡ Music recommender
    </div>

</div>
"""
)


# =========================================================
# HERO
# =========================================================

st.html(
    """
<div class="hero">

    <div class="hero-headphones">
        🎧
    </div>

    <div class="hero-stars">
        ✦ ♫
    </div>

    <div class="hero-kicker">
        ✿ Your tiny music matchmaker
    </div>

    <h1>
        Find your<br>
        next obsession.
    </h1>

    <p>
        Pick a song you already love and Doodle Tunes
        will dig through thousands of tracks to find
        five songs with suspiciously similar energy.
    </p>

</div>
"""
)


# =========================================================
# SONG PICKER
# =========================================================

st.html(
    """
<div class="section-title">
    What are we listening to? 🎀
</div>

<div class="section-subtitle">
    Search a song or artist below.
</div>
"""
)


song_list = sorted(
    df["song_label"]
    .dropna()
    .unique()
)


selected_song = st.selectbox(
    "Song",
    song_list,
    label_visibility="collapsed",
)


selected_info = (
    df[
        df["song_label"]
        == selected_song
    ]
    .iloc[0]
)


# =========================================================
# TWO-COLUMN LAYOUT
# =========================================================

left, right = st.columns(
    [0.32, 0.68],
    gap="large",
)


# =========================================================
# LEFT: PLAYER
# =========================================================

with left:

    safe_title = html.escape(
        str(
            selected_info[
                "track_name"
            ]
        )
    )

    safe_artist = html.escape(
        str(
            selected_info[
                "artists"
            ]
        )
    )

    safe_genre = html.escape(
        str(
            selected_info[
                "track_genre"
            ]
        )
    )


    st.html(
        f"""
<div class="player">

    <div class="album-art"></div>

    <div class="now-playing">
        Now playing
    </div>

    <div class="player-song">
        {safe_title}
    </div>

    <div class="player-artist">
        {safe_artist}
    </div>

    <div class="progress">
        <div class="progress-fill"></div>
    </div>

    <div class="player-controls">
        ◀  ❚❚  ▶
    </div>

    <div class="tags">

        <span class="tag tag-pink">
            {safe_genre}
        </span>

        <span class="tag tag-yellow">
            ★ Popularity {int(selected_info["popularity"])}
        </span>

        <span class="tag tag-purple">
            ♡ Selected
        </span>

    </div>

</div>
"""
    )


    st.write("")


    clicked = st.button(
        "✧ Find my music twins ✧",
        use_container_width=True,
    )


# =========================================================
# RIGHT: RECOMMENDATIONS
# =========================================================

with right:

    st.html(
        """
<div class="section-title">
    Made for your ears ✨
</div>

<div class="section-subtitle">
    Your five closest musical cousins.
</div>
"""
    )


    if clicked:

        recommendations = recommend_songs(
            selected_song,
            5,
        )


        cards = ""


        backgrounds = [
            "#fff2f7",
            "#f3efff",
            "#fff9df",
            "#eef8f0",
            "#eef7ff",
        ]


        for i, row in recommendations.iterrows():

            safe_track = html.escape(
                str(
                    row[
                        "track_name"
                    ]
                )
            )

            safe_artist = html.escape(
                str(
                    row[
                        "artists"
                    ]
                )
            )

            safe_genre = html.escape(
                str(
                    row[
                        "track_genre"
                    ]
                )
            )


            similarity = (
                float(
                    row[
                        "similarity"
                    ]
                )
                * 100
            )


            bg = backgrounds[
                i
                % len(
                    backgrounds
                )
            ]


            cards += f"""
<div
    class="rec-card"
    style="background:{bg};"
>

    <div class="rec-number">
        {i + 1}
    </div>

    <div class="rec-info">

        <div class="rec-song">
            {safe_track}
        </div>

        <div class="rec-artist">
            {safe_artist}
        </div>

        <div class="tags">

            <span class="tag tag-purple">
                ♫ {safe_genre}
            </span>

            <span class="tag tag-green">
                ★ Popularity {int(row["popularity"])}
            </span>

            <span class="tag tag-pink">
                {similarity:.0f}% Match
            </span>

        </div>

    </div>

    <div class="rec-heart">
        ♡
    </div>

</div>
"""


        st.html(
            f"""
<div class="rec-wrap">
    {cards}
</div>
"""
        )


    else:

        st.html(
            """
<div class="empty-box">

    <div class="empty-icon">
        🎶
    </div>

    Pick a song, press the pink button,
    <br>
    and your recommendations will appear here.

</div>
"""
        )