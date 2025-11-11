import streamlit as st
import requests, os, json, random, joblib
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from streamlit_oauth import OAuth2Component
from database import get_user_by_provider, create_user, save_user_data, load_user_data

# ----------------------------------------
# 🔧 Setup
# ----------------------------------------
load_dotenv()
st.set_page_config(page_title="Smart Wardrobe", page_icon="👗", layout="wide")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8501/")
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPE = "openid email profile"

oauth2 = OAuth2Component(
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    AUTHORIZE_URL,
    TOKEN_URL,
    None,
    None
)

# ----------------------------------------
# 👤 Login with Google
# ----------------------------------------
if "token" not in st.session_state:
    result = oauth2.authorize_button("🔑 Sign in with Google", REDIRECT_URI, SCOPE)
    if result and "token" in result:
        access_token = result["token"]["access_token"]
        user_info_resp = requests.get(USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
        if user_info_resp.status_code == 200:
            user_info = user_info_resp.json()
            st.session_state["token"] = access_token
            st.session_state["user_info"] = user_info
            st.rerun()
        else:
            st.error("Login failed or user information missing. Please try again.")
    st.stop()

# ----------------------------------------
# 👕 Load or create user
# ----------------------------------------
user_info = st.session_state["user_info"]
user = get_user_by_provider("google", user_info["sub"])
if not user:
    user = create_user(
        "google",
        user_info["sub"],
        user_info.get("name"),
        user_info.get("email"),
        user_info.get("picture")
    )

st.session_state["user_id"] = user.id
st.sidebar.success(f"Welcome, {user_info.get('name', 'User')} 👋")

# Load user's wardrobe & profile data from DB
user_data = load_user_data(user.id) or {}
wardrobe = user_data.get("wardrobe", [])
profile = user_data.get("profile", {})

# ----------------------------------------
# 🧠 Style Recommender
# ----------------------------------------
class StyleRecommender:
    def _init_(self):
        self.model_path = "models/outfit_recommender.pkl"
        self.le = LabelEncoder()
        self.model = None
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
        os.makedirs("models", exist_ok=True)
        if os.path.exists(self.model_path):
            obj = joblib.load(self.model_path)
            self.model, self.le = obj["model"], obj["le"]
            return
        X, y = self._create_synthetic(1500)
        self.le.fit(y)
        y_enc = self.le.transform(y)
        self.model = RandomForestClassifier(n_estimators=80, random_state=42, max_depth=12)
        self.model.fit(X, y_enc)
        joblib.dump({"model": self.model, "le": self.le}, self.model_path)

    def _create_synthetic(self, n):
        moods = ["confident","happy","neutral","anxious","low"]
        locs = ["home","office","cafe","public_transport","night_out"]
        events = ["work","meeting","casual","date","party"]
        comps = ["alone","friends","mixed","strangers"]
        weathers = ["hot","mild","cold","rainy"]
        labels, X = [], []
        for _ in range(n):
            ctx = {
                "mood": random.choice(moods),
                "location": random.choice(locs),
                "event": random.choice(events),
                "company": random.choice(comps),
                "weather": random.choice(weathers),
                "safety_score": random.randint(1,10),
                "self_expression": random.randint(1,10)
            }
            label = f"{ctx['mood']} {ctx['event']} outfit"
            X.append(self._feature_vec(ctx))
            labels.append(label)
        return np.array(X), np.array(labels)

    def predict(self, ctx, topk=3):
        X = np.array(self._feature_vec(ctx)).reshape(1,-1)
        probs = self.model.predict_proba(X)[0]
        idxs = np.argsort(probs)[::-1][:topk]
        labels = self.le.inverse_transform(idxs)
        return list(zip(labels, probs[idxs]))

recommender = StyleRecommender()

# ----------------------------------------
# 🧭 Sidebar Navigation
# ----------------------------------------
st.sidebar.title("👗 Smart Wardrobe")
page = st.sidebar.radio(
    "Navigate",
    ["👤 Profile", "➕ Add Clothes", "👔 Browse Wardrobe", "✨ Get Recommendation"]
)

# ----------------------------------------
# 👤 Profile
# ----------------------------------------
if page == "👤 Profile":
    st.title("👤 Profile")
    with st.form("profile_form"):
        name = st.text_input("Name", value=profile.get("name", user_info.get("name", "")))
        age = st.number_input("Age", 13, 100, value=profile.get("age", 25))
        profession = st.text_input("Profession", value=profile.get("profession", ""))
        fashion_sense = st.selectbox("Fashion Sense", ["conservative","moderate","bold"],
                                     index=["conservative","moderate","bold"].index(profile.get("fashion_sense","moderate")))
        submitted = st.form_submit_button("Save")
        if submitted:
            profile = {
                "name": name, "age": age,
                "profession": profession,
                "fashion_sense": fashion_sense,
                "updated_at": datetime.now().isoformat()
            }
            user_data["profile"] = profile
            save_user_data(user.id, user_data)
            st.success("Profile updated successfully!")

# ----------------------------------------
# ➕ Add Clothes
# ----------------------------------------
elif page == "➕ Add Clothes":
    st.title("➕ Add Clothing Item")
    uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)
        name = st.text_input("Item name")
        category = st.selectbox("Category", ["T-Shirt","Dress","Jeans","Jacket","Shirt","Sweater"])
        formality = st.selectbox("Formality", ["Casual","Formal","Party"])
        if st.button("Add to Wardrobe"):
            item = {
                "id": len(wardrobe)+1,
                "name": name,
                "category": category,
                "formality": formality,
                "date_added": datetime.now().isoformat()
            }
            wardrobe.append(item)
            user_data["wardrobe"] = wardrobe
            save_user_data(user.id, user_data)
            st.success("Added successfully!")

# ----------------------------------------
# 👔 Browse Wardrobe
# ----------------------------------------
elif page == "👔 Browse Wardrobe":
    st.title("👔 Your Wardrobe")
    if not wardrobe:
        st.info("Your wardrobe is empty.")
    else:
        for item in wardrobe:
            st.write(f"👗 {item['name']} — {item['category']} ({item['formality']})")

# ----------------------------------------
# ✨ Recommendation
# ----------------------------------------
elif page == "✨ Get Recommendation":
    st.title("✨ Outfit Recommendation")
    mood = st.selectbox("Mood", ["confident","happy","neutral","anxious","low"])
    location = st.selectbox("Location", ["home","office","cafe","night_out"])
    event = st.selectbox("Event", ["work","meeting","casual","party"])
    company = st.selectbox("Company", ["alone","friends","strangers"])
    weather = st.selectbox("Weather", ["hot","mild","cold"])
    safety_score = st.slider("Safety", 1, 10, 5)
    self_expression = st.slider("Self Expression", 1, 10, 5)

    if st.button("Recommend Outfit"):
        ctx = {
            "mood": mood, "location": location, "event": event,
            "company": company, "weather": weather,
            "safety_score": safety_score, "self_expression": self_expression
        }
        preds = recommender.predict(ctx)
        st.subheader("🎀 Suggested Styles:")
        for label, prob in preds:
            st.write(f"- {label} ({prob:.2%})")

# ----------------------------------------
# 🚪 Logout
# ----------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()