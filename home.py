# home.py
import streamlit as st
from PIL import Image
import numpy as np
import os, json, joblib, random, time, requests
from datetime import datetime
from collections import Counter
import cv2
from urllib.parse import quote_plus
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Smart Wardrobe", page_icon="👗", layout="wide")

# Directories
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(f"{DATA_DIR}/wardrobe_images", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

USER_PROFILE_PATH = f"{DATA_DIR}/user_profile.json"
WARDROBE_PATH = f"{DATA_DIR}/wardrobe.json"
OOTD_PATH = f"{DATA_DIR}/ootd_logs.json"
RECOMMENDER_PATH = f"{MODEL_DIR}/outfit_recommender.pkl"

# Utility functions
def load_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except:
        return None
    return None

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# State
if 'wardrobe' not in st.session_state:
    st.session_state.wardrobe = load_json(WARDROBE_PATH) or []
if 'ootd_logs' not in st.session_state:
    st.session_state.ootd_logs = load_json(OOTD_PATH) or []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = load_json(USER_PROFILE_PATH) or {}

# Categories
CATEGORIES = [
    'T-Shirt/Top','Trouser','Pullover/Sweater','Dress','Coat/Jacket',
    'Sandal','Shirt/Blouse','Sneaker','Bag','Ankle Boot','Jeans','Shorts','Skirt'
]
TAG_MAP = {
    'T-Shirt/Top': ['casual','comfortable','everyday'],
    'Trouser': ['professional','formal'],
    'Pullover/Sweater': ['cozy','layering'],
    'Dress': ['elegant','party'],
    'Coat/Jacket': ['outer','layering'],
    'Sandal': ['summer','casual'],
    'Shirt/Blouse': ['work','polished'],
    'Sneaker': ['casual','sporty'],
    'Bag': ['accessory'],
    'Ankle Boot': ['stylish'],
    'Jeans': ['denim','casual'],
    'Shorts': ['casual','summer'],
    'Skirt': ['feminine']
}

# Simple color extractor
def extract_colors(image, n=3):
    img = image.convert('RGB').resize((40, 40))
    arr = np.array(img).reshape(-1,3)
    idxs = np.random.choice(len(arr), min(n, len(arr)), replace=False)
    return ['#{:02x}{:02x}{:02x}'.format(*tuple(arr[i])) for i in idxs]

# --------------------- Style Recommender ---------------------
class StyleRecommender:
    def __init__(self, path=RECOMMENDER_PATH):
        self.path = path
        self.model = None
        self.le = LabelEncoder()
        self._ensure()

    def _feature_vec(self, ctx):
        mood_map = ["confident","happy","neutral","anxious","low"]
        loc_map = ["home","office","cafe","public_transport","night_out"]
        event_map = ["work","meeting","casual","date","party"]
        comp_map = ["alone","friends","mixed","strangers"]
        weather_map = ["hot","mild","cold","rainy"]
        return [
            mood_map.index(ctx.get("mood","neutral")),
            loc_map.index(ctx.get("location","home")),
            event_map.index(ctx.get("event","casual")),
            comp_map.index(ctx.get("company","friends")),
            weather_map.index(ctx.get("weather","mild")),
            int(ctx.get("safety_score",5)),
            int(ctx.get("self_expression",5))
        ]

    def _ensure(self):
        if os.path.exists(self.path):
            try:
                obj = joblib.load(self.path)
                self.model = obj['model']
                self.le = obj['le']
                return
            except Exception as e:
                print("Could not load recommender:", e)
        X, y = self._create_synthetic(2000)
        self.le.fit(y)
        y_enc = self.le.transform(y)
        self.model = RandomForestClassifier(n_estimators=80, random_state=42, max_depth=12)
        self.model.fit(X, y_enc)
        joblib.dump({'model': self.model, 'le': self.le}, self.path)

    def _create_synthetic(self, n):
        moods = ["confident","happy","neutral","anxious","low"]
        locs = ["home","office","cafe","public_transport","night_out"]
        events = ["work","meeting","casual","date","party"]
        comps = ["alone","friends","mixed","strangers"]
        weathers = ["hot","mild","cold","rainy"]
        bottoms = ["ripped jeans","skinny jeans","straight jeans","chinos","tailored trousers","pleated skirt","mini skirt","shorts"]
        colors = ["white","black","blue","red","pink","green","yellow","beige","brown","grey","purple"]
        patterns = ["plain","striped","floral","polka","checked","graphic"]
        X, styles = [], []
        for _ in range(n):
            attrs = {
                "category": random.choice(["tshirt","shirt","blouse","dress","pullover","jacket","jeans","trouser","skirt","shorts"]),
                "color": random.choice(colors),
                "pattern": random.choice(patterns),
                "sleeve": random.choice(["short","long","sleeveless","three-quarter"]),
                "fit": random.choice(["slim","regular","oversized"]),
                "bottom_style": random.choice(bottoms),
                "footwear": random.choice(["white sneakers","black boots","sandals","loafers","heels"])
            }
            label = f"{attrs['pattern']} {attrs['color']} {attrs['category']} {attrs['sleeve']}-sleeve {attrs['fit']} with {attrs['bottom_style']} + {attrs['footwear']}"
            ctx = {
                "mood": random.choice(moods),
                "location": random.choice(locs),
                "event": random.choice(events),
                "company": random.choice(comps),
                "weather": random.choice(weathers),
                "safety_score": random.randint(1,10),
                "self_expression": random.randint(1,10)
            }
            X.append(self._feature_vec(ctx))
            styles.append(label)
        return np.array(X), np.array(styles)

    def predict(self, context, topk=3):
        X = np.array(self._feature_vec(context)).reshape(1, -1)
        probs = self.model.predict_proba(X)[0]
        idxs = np.argsort(probs)[::-1][:topk]
        labels = self.le.inverse_transform(idxs)
        return list(zip(labels, probs[idxs]))

recommender = StyleRecommender()

# --------------------- Sidebar Navigation ---------------------
st.sidebar.title("👗 Smart Wardrobe")
page = st.sidebar.radio(
    "Navigate",
    ["👤 Profile", "➕ Add Clothes", "👔 Browse Wardrobe",
     "✨ Get Recommendation", "📸 Log OOTD",
     "🛡️ Safety Insights", "🎨 Style Inspiration", "Wardrobe Colors"]
)

# --------------------- Profile Page ---------------------
if page == "👤 Profile":
    st.title("👤 Profile")
    profile = st.session_state.user_profile or {}
    with st.form("profile_form"):
        name = st.text_input("Name", value=profile.get('name',''))
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 13, 100, value=profile.get('age',25))
        with col2:
            profession = st.text_input("Profession", value=profile.get('profession',''))
        fashion_sense = st.selectbox("Fashion Sense", ['conservative','moderate','bold'])
        submitted = st.form_submit_button("Save")
        if submitted:
            pdata = {'name':name,'age':age,'profession':profession,'fashion_sense':fashion_sense,'created_at':datetime.now().isoformat()}
            st.session_state.user_profile = pdata
            save_json(USER_PROFILE_PATH, pdata)
            st.success("Profile saved.")

# --------------------- Add Clothes ---------------------
elif page == "➕ Add Clothes":
    st.title("➕ Add Clothing Item")
    uploaded = st.file_uploader("Upload image", type=['png','jpg','jpeg'])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)
        colors = extract_colors(image, 3)
        name = st.text_input("Item name", value=f"Item {len(st.session_state.wardrobe)+1}")
        category = st.selectbox("Category", CATEGORIES)
        formality = st.selectbox("Formality", ['casual','business-casual','formal','party'])
        submitted = st.button("Add")
        if submitted:
            filename = f"item_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = f"{DATA_DIR}/wardrobe_images/{filename}"
            image.save(path)
            item = {
                'id': len(st.session_state.wardrobe),
                'name': name,
                'category': category,
                'colors': colors,
                'formality': formality,
                'image_path': path,
                'date_added': datetime.now().isoformat(),
                'times_worn': 0
            }
            st.session_state.wardrobe.append(item)
            save_json(WARDROBE_PATH, st.session_state.wardrobe)
            st.success("Added successfully.")

# --------------------- Browse Wardrobe ---------------------
elif page == "👔 Browse Wardrobe":
    st.title("👔 Your Wardrobe")
    if not st.session_state.wardrobe:
        st.info("No items yet.")
    else:
        cats = ['All'] + sorted(list({it['category'] for it in st.session_state.wardrobe}))
        sel = st.selectbox("Filter by category", cats)
        items = st.session_state.wardrobe if sel == 'All' else [it for it in st.session_state.wardrobe if it['category'] == sel]
        cols = st.columns(3)
        for i, it in enumerate(items):
            with cols[i % 3]:
                if os.path.exists(it['image_path']):
                    st.image(Image.open(it['image_path']), use_column_width=True)
                st.write(f"**{it['name']}** — {it['category']}")
                if st.button("Delete", key=f"del_{i}"):
                    st.session_state.wardrobe.remove(it)
                    save_json(WARDROBE_PATH, st.session_state.wardrobe)
                    st.experimental_rerun()

# --------------------- Recommendation ---------------------
elif page == "✨ Get Recommendation":
    st.title("✨ Get Outfit Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        mood = st.selectbox("Mood", ['confident','happy','neutral','anxious','low'])
        location = st.selectbox("Location", ['home','office','cafe','public_transport','night_out'])
        event = st.selectbox("Event", ['work','meeting','casual','date','party'])
    with col2:
        company = st.selectbox("Company", ['alone','friends','mixed','strangers'])
        weather = st.selectbox("Weather", ['hot','mild','cold','rainy'])
        safety_score = st.slider("Safety", 1, 10, 7)
    self_expression = st.slider("Self Expression", 1, 10, 6)
    if st.button("Recommend"):
        ctx = {'mood':mood,'location':location,'event':event,'company':company,'weather':weather,
               'safety_score':safety_score,'self_expression':self_expression}
        preds = recommender.predict(ctx)
        st.subheader("Suggested Styles:")
        for label, prob in preds:
            st.write(f"- {label} ({prob:.2%})")

# --------------------- Log OOTD ---------------------
elif page == "📸 Log OOTD":
    st.title("📸 Log Outfit of the Day")
    st.info("Feature under integration with personalization module.")

# --------------------- Wardrobe Colors ---------------------
elif page == "Wardrobe Colors":
    st.title("🎨 Wardrobe Color Analysis")
    uploaded_file = st.file_uploader("Upload wardrobe image", type=["jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Wardrobe", use_container_width=True)
        img_array = np.array(image)
        resized_img = cv2.resize(img_array, (200, 200))
        reshaped_img = resized_img.reshape((-1, 3))
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(reshaped_img)
        colors = kmeans.cluster_centers_.astype(int)
        st.subheader("Dominant Colors")
        fig, ax = plt.subplots(1, k, figsize=(12, 3))
        for i in range(k):
            patch = np.zeros((100, 100, 3), dtype=np.uint8)
            patch[:, :] = colors[i]
            ax[i].imshow(patch)
            ax[i].axis("off")
        st.pyplot(fig)
        avg_brightness = np.mean(cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)[:, :, 2])
        st.write("Your wardrobe is **bright/vibrant**." if avg_brightness > 120 else "Your wardrobe is **muted/darker**.")

# --------------------- Style Inspiration ---------------------
elif page == "🎨 Style Inspiration":
    st.title("🎨 Style Inspiration (Pinterest API)")
    st.subheader("Find outfit ideas based on your mood, company, and aesthetic.")

    from urllib.parse import quote_plus
    import requests
    import os
    from dotenv import load_dotenv
    load_dotenv()

    @st.cache_data(ttl=3600)
    def fetch_style_pins(query, limit=9):
        api_key = os.getenv("SCRAPE_CREATORS_API_KEY")
        if not api_key:
            raise RuntimeError("⚠ Missing API key! Add SCRAPE_CREATORS_API_KEY to your .env or st.secrets.")

        url_q = quote_plus(query)
        url = f"https://api.scrapecreators.com/v1/pinterest/search?query={url_q}&limit={limit}"
        headers = {"x-api-key": api_key}

        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        pins = data.get("pins") or data.get("data") or data.get("results") or []
        images = []
        for p in pins:
            if not isinstance(p, dict):
                continue
            img_url = None
            imgs = p.get("images") or {}
            if isinstance(imgs, dict):
                orig = imgs.get("orig") or imgs.get("original") or next(iter(imgs.values()), None)
                if isinstance(orig, dict):
                    img_url = orig.get("url") or orig.get("src")
            img_url = img_url or p.get("image") or p.get("image_url") or p.get("url") or p.get("link")
            title = p.get("title") or p.get("description") or p.get("note") or ""
            if img_url:
                images.append({"image_url": img_url, "title": title})
        return images[:limit]

    # --- User Inputs ---
    mood = st.text_input("🌸 Enter your mood (e.g., happy, chill, bold)")
    company = st.text_input("👭 Who are you with? (e.g., friends, date, work)")
    aesthetic = st.text_input("✨ Aesthetic preference (e.g., minimal, boho, chic)")

    if st.button("Get Style Ideas"):
        query = f"{mood} {company} {aesthetic} women's fashion".strip()
        if not query.strip():
            st.warning("⚠ Please enter some details first.")
        else:
            st.write(f"🔍 Searching Pinterest for: *{query}*")
            with st.spinner("Fetching ideas..."):
                try:
                    pins = fetch_style_pins(query, limit=9)
                except Exception as e:
                    st.error(f"❌ Failed to fetch: {e}")
                    pins = []

            if not pins:
                st.info("No results found. Try different words (e.g., 'casual outfit').")
            else:
                st.success(f"✅ Found {len(pins)} style ideas!")
                cols = st.columns(3)
                for idx, pin in enumerate(pins):
                    with cols[idx % 3]:
                        st.image(pin["image_url"], caption=pin["title"][:80], use_container_width=True)

# --------------------- Safety Insights ---------------------
elif page == "🛡️ Safety Insights":
    st.title("🛡️ Women's Safety Score")
    st.subheader("Get a heuristic safety estimate for your location.")
    location = st.text_input("📍 Enter location", "San Francisco, CA")
    if st.button("Check Safety Score 🚨"):
        st.info(f"Analyzing safety for: {location}")
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {"q": location, "format": "json", "limit": 1}
            resp = requests.get(url, params=params, headers={"User-Agent": "SmartWardrobe/1.0"}, timeout=10)
            if resp.ok and resp.json():
                res = resp.json()[0]
                lat, lon = float(res["lat"]), float(res["lon"])
                st.success(f"📌 {res['display_name']} ({lat:.4f}, {lon:.4f})")
            random.seed(len(location))
            crimes = random.randint(0, 15)
            neg_reviews = random.randint(0, 10)
            pos_reviews = random.randint(5, 20)
            lighting = random.randint(4, 10)
            score = 100 - crimes*3 - neg_reviews*5 + pos_reviews*3 + lighting
            score = max(0, min(100, score))
            st.metric("Safety Score", f"{score}/100")
            if score < 40:
                st.warning("⚠️ Low safety — choose modest coverage.")
            elif score < 70:
                st.info("ℹ️ Moderate safety — dress cautiously.")
            else:
                st.success("✅ High safety — express yourself freely.")
        except Exception as e:
            st.error(f"Error: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Smart Wardrobe — AI-driven style and safety intelligence.")
