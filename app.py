from flask import Flask, request, render_template, jsonify, url_for, redirect, flash
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai
import numpy as np
import os
import io
import uuid
import json
import logging
import cv2
import shutil
import traceback
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote_plus
from datetime import datetime
import random
from segment2 import segment_analyze_plant
from routes.post_harvest import post_harvest_bp
from routes.schemes import schemes_bp
import signal
import sys
import socket
from datetime import datetime, timedelta
from flask import session


# LOGIN & DATABASE IMPORTS
from flask_login import LoginManager, login_required, current_user, logout_user
from model import db, User, LoginHistory, DiseaseDetection, WeeklyAssessment
from routes.auth import auth_bp

from nutrition_analyzer import (
    analyze_nutrition_deficiency, 
    calculate_fertilizer_dosage, 
    load_nutrition_deficiency_data
)
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

nutrition_deficiency_data = load_nutrition_deficiency_data()
logger.info(f"Loaded {len(nutrition_deficiency_data)} nutrition deficiency types")

SERVER_START_TIME = datetime.now()

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)

# ===== SESSION & DATABASE CONFIGURATION =====
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_COOKIE_NAME'] = 'agripal_session'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///agripal.db')

if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SERVER_START_TIME'] = SERVER_START_TIME.isoformat()

# ===== OTHER CONFIGURATIONS =====
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
app.config['DEBUG'] = (FLASK_ENV != 'production')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info(f"Upload folder configured at: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")

# ===== INITIALIZE EXTENSIONS =====
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# ===== USER LOADER =====
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAL_7MfAGGI8HBpyUhAvyzUl9hPIWJk4bk")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API configured successfully")
except Exception as e:
    logger.error(f"❌ Failed to configure Gemini API: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image):
    try:
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

# ML model is not loaded - running in maintenance/placeholder mode
model = None
logger.info("ℹ️ ML model not loaded - running in maintenance/placeholder mode")

def load_disease_treatments():
    try:
        treatment_path = 'disease_treatments.json'
        if os.path.exists(treatment_path):
            with open(treatment_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded disease treatments from {treatment_path}")
                return data
        else:
            logger.error(f"Disease treatments file not found at: {os.path.abspath(treatment_path)}")
            return {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading disease treatments: {e}")
        return {}

class_names = [
    "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
    "Blueberry_healthy", "Cherry_(including_sour)Powdery_mildew", "Cherry(including_sour)_healthy",
    "Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot", "Corn(maize)_Common_rust",
    "Corn_(maize)Northern_Leaf_Blight", "Corn(maize)_healthy", "Grape_Black_rot",
    "Grape_Esca_(Black_Measles)", "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape_healthy",
    "Orange_Haunglongbing_(Citrus_greening)", "Peach_Bacterial_spot", "Peach_healthy",
    "Pepper_bell_Bacterial_spot", "Pepper_bell_healthy", "Potato_Early_blight",
    "Potato_Late_blight", "Potato_healthy", "Raspberry_healthy", "Soybean_healthy",
    "Squash_Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry_healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two-spotted_spider_mite", "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato_healthy"
]

CONFIDENCE_THRESHOLD = 50.0
SUPPORTED_PLANTS = {
    'Apple': ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy'],
    'Blueberry': ['Blueberry_healthy'],
    'Cherry': ['Cherry_(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy'],
    'Corn (Maize)': ['Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot', 'Corn(maize)_Common_rust', 
                     'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy'],
    'Grape': ['Grape_Black_rot', 'Grape_Esca_(Black_Measles)', 'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape_healthy'],
    'Orange': ['Orange_Haunglongbing_(Citrus_greening)'],
    'Peach': ['Peach_Bacterial_spot', 'Peach_healthy'],
    'Pepper (Bell)': ['Pepper_bell_Bacterial_spot', 'Pepper_bell_healthy'],
    'Potato': ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy'],
    'Raspberry': ['Raspberry_healthy'],
    'Soybean': ['Soybean_healthy'],
    'Squash': ['Squash_Powdery_mildew'],
    'Strawberry': ['Strawberry_Leaf_scorch', 'Strawberry_healthy'],
    'Tomato': ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato_Target_Spot',
               'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy']
}

COMMON_QUESTIONS = {
    "plant_diseases": [
        "What are the most common tomato diseases?",
        "How do I identify powdery mildew?",
        "What causes yellow leaves on plants?",
        "How to prevent fungal diseases in plants?",
        "What are the signs of bacterial infection in crops?",
        "How to identify viral diseases in plants?",
        "What causes leaf spots on vegetables?",
        "How to detect early blight in tomatoes?"
    ],
    "treatment_methods": [
        "What are organic pest control methods?",
        "How to make homemade fungicide?",
        "What is integrated pest management?",
        "How to use neem oil for plant diseases?",
        "What are the best copper-based fungicides?",
        "How to apply systemic pesticides safely?",
        "What is the difference between preventive and curative treatments?",
        "How to rotate pesticides to prevent resistance?"
    ],
    "crop_management": [
        "When is the best time to plant tomatoes?",
        "How much water do vegetables need daily?",
        "What is crop rotation and why is it important?",
        "How to improve soil fertility naturally?",
        "What are companion plants for tomatoes?",
        "How to prepare soil for planting?",
        "What are the signs of nutrient deficiency?",
        "How to manage weeds organically?"
    ],
    "seasonal_advice": [
        "What crops to plant in monsoon season?",
        "How to protect plants from extreme heat?",
        "What are winter crop management tips?",
        "How to prepare garden for rainy season?",
        "What vegetables grow best in summer?",
        "How to manage greenhouse in different seasons?",
        "What are post-harvest handling best practices?",
        "How to store seeds for next season?"
    ],
    "technology_agriculture": [
        "How can AI help in agriculture?",
        "What are smart farming techniques?",
        "How to use drones in agriculture?",
        "What are precision agriculture tools?",
        "How does satellite imagery help farmers?",
        "What are IoT applications in farming?",
        "How to use weather data for crop planning?",
        "What are digital farming platforms?"
    ]
}

disease_treatments = load_disease_treatments()
logger.info(f"Loaded {len(disease_treatments)} disease treatments")


def normalize_disease_info(disease_info):
    """
    Map old JSON field names to standardized template field names.
    This allows backward compatibility with the existing JSON structure.
    """
    if not disease_info or 'pesticide' not in disease_info:
        return disease_info
    
    import copy
    normalized = copy.deepcopy(disease_info)
    
    logger.info("=" * 80)
    logger.info("📄 NORMALIZING DISEASE INFO FIELDS")
    logger.info("=" * 80)
    
    for treatment_type in ['chemical', 'organic']:
        if treatment_type not in normalized['pesticide']:
            logger.warning(f"⚠️ No {treatment_type} treatment found")
            continue
            
        treatment = normalized['pesticide'][treatment_type]
        logger.info(f"📦 Processing {treatment_type.upper()} treatment...")
        
        if 'application_frequency' in treatment and 'frequency' not in treatment:
            treatment['frequency'] = treatment['application_frequency']
            logger.info(f"  ✅ Mapped application_frequency -> frequency")
            logger.info(f"     Value: {treatment['frequency'][:50]}...")
        elif 'frequency' not in treatment or not treatment.get('frequency'):
            treatment['frequency'] = "Apply according to product label recommendations and disease pressure."
            logger.warning(f"  ⚠️ No frequency field found, added fallback")
        
        if 'precautions' in treatment and 'safety' not in treatment:
            treatment['safety'] = treatment['precautions']
            logger.info(f"  ✅ Mapped precautions -> safety")
            logger.info(f"     Value: {treatment['safety'][:50]}...")
        elif 'safety' not in treatment or not treatment.get('safety'):
            if treatment_type == 'chemical':
                treatment['safety'] = "Wear protective equipment. Follow all label precautions. Keep away from water sources."
            else:
                treatment['safety'] = "Safe for beneficial insects when used as directed. Apply during cooler parts of day."
            logger.warning(f"  ⚠️ No safety field found, added fallback")
        
        if 'usage' not in treatment or not treatment.get('usage') or len(treatment.get('usage', '').strip()) < 10:
            treatment['usage'] = f"Apply as directed on product label. Ensure thorough coverage of all affected plant surfaces. Repeat applications as needed based on disease pressure."
            logger.warning(f"  ⚠️ Missing or short usage, added fallback")
        
        required_fields = {
            'name': f"{treatment_type.title()} Treatment",
            'dosage_per_hectare': 0.0,
            'unit': 'L',
            'usage': 'Apply as directed',
            'frequency': 'As needed',
            'safety': 'Follow product label instructions'
        }
        
        for field, default_value in required_fields.items():
            if field not in treatment or not treatment.get(field):
                treatment[field] = default_value
                logger.warning(f"  ⚠️ Missing {field}, added default: {default_value}")
        
        logger.info(f"  📊 Field lengths:")
        logger.info(f"     - Name: {len(treatment.get('name', ''))} chars")
        logger.info(f"     - Usage: {len(treatment.get('usage', ''))} chars")
        logger.info(f"     - Frequency: {len(treatment.get('frequency', ''))} chars")
        logger.info(f"     - Safety: {len(treatment.get('safety', ''))} chars")
    
    logger.info("=" * 80)
    logger.info("✅ NORMALIZATION COMPLETE")
    logger.info("=" * 80)
    
    return normalized


def get_disease_info(disease_name):
    """
    Enhanced function with field normalization and detailed logging.
    """
    try:
        logger.info("=" * 80)
        logger.info(f"🔍 DISEASE LOOKUP: {disease_name}")
        logger.info("=" * 80)
        logger.info(f"📚 Database has {len(disease_treatments)} diseases")
        
        disease_info = disease_treatments.get(disease_name, None)
        
        if not disease_info:
            logger.info(f"⚠️ No exact match, trying variations...")
            cleaned_name = disease_name.replace('_', ' ').replace('(', '').replace(')', '').strip()
            
            for key, value in disease_treatments.items():
                if cleaned_name.lower() in key.lower() or key.lower() in cleaned_name.lower():
                    disease_info = value
                    logger.info(f"✅ Found match with key: {key}")
                    break
        
        if not disease_info:
            logger.error(f"❌ NO DISEASE INFO FOUND for: {disease_name}")
            available = list(disease_treatments.keys())[:5]
            logger.info(f"📝 Available diseases (first 5): {available}")
            return None
        
        logger.info(f"✅ Raw disease info found")
        logger.info(f"📋 Raw keys: {list(disease_info.keys())}")
        
        disease_info = normalize_disease_info(disease_info)
        
        logger.info("=" * 80)
        logger.info("📊 FINAL VALIDATION")
        logger.info("=" * 80)
        logger.info(f"✅ Disease Name: {disease_info.get('name')}")
        logger.info(f"✅ Description: {len(disease_info.get('description', ''))} chars")
        logger.info(f"✅ Treatment Steps: {len(disease_info.get('treatment', []))}")
        logger.info(f"✅ Severity: {disease_info.get('severity')}")
        
        if 'pesticide' in disease_info:
            for treatment_type in ['chemical', 'organic']:
                if treatment_type in disease_info['pesticide']:
                    t = disease_info['pesticide'][treatment_type]
                    logger.info(f"")
                    logger.info(f"📦 {treatment_type.upper()}:")
                    logger.info(f"  Name: {t.get('name')}")
                    logger.info(f"  Usage: {len(t.get('usage', ''))} chars - {bool(t.get('usage'))}")
                    logger.info(f"  Frequency: {len(t.get('frequency', ''))} chars - {bool(t.get('frequency'))}")
                    logger.info(f"  Safety: {len(t.get('safety', ''))} chars - {bool(t.get('safety'))}")
                    logger.info(f"  Dosage: {t.get('dosage_per_hectare')} {t.get('unit')}/hectare")
        
        if 'pesticide' in disease_info:
            for treatment_type in ['chemical', 'organic']:
                if treatment_type not in disease_info['pesticide']:
                    continue
                
                treatment = disease_info['pesticide'][treatment_type]
                
                if 'video_sources' in treatment:
                    video_sources = treatment['video_sources']
                    
                    if 'search_terms' in video_sources:
                        search_urls = []
                        for term in video_sources['search_terms']:
                            search_urls.append({
                                'term': term,
                                'url': f"https://www.youtube.com/results?search_query={quote_plus(term)}"
                            })
                        video_sources['search_urls'] = search_urls
                        logger.info(f"✅ Added {len(search_urls)} YouTube URLs for {treatment_type}")
                    
                    if 'reliable_channels' in video_sources:
                        channel_urls = []
                        for channel in video_sources['reliable_channels']:
                            channel_urls.append({
                                'name': channel,
                                'url': f"https://www.youtube.com/results?search_query={quote_plus(channel + ' ' + disease_name.replace('_', ' '))}"
                            })
                        video_sources['channel_urls'] = channel_urls
                        logger.info(f"✅ Added {len(channel_urls)} channel URLs for {treatment_type}")
        
        logger.info("=" * 80)
        logger.info("✅ DISEASE INFO PROCESSING COMPLETE")
        logger.info("=" * 80)
        
        return disease_info
        
    except Exception as e:
        logger.error(f"❌ ERROR in get_disease_info: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def combine_disease_treatments(unique_diseases):
    """
    Combine treatments from multiple diseases of the same plant.
    Returns a merged treatment plan with intelligent deduplication.
    """
    logger.info("=" * 80)
    logger.info("🔀 COMBINING TREATMENTS FROM MULTIPLE DISEASES")
    logger.info("=" * 80)
    
    combined = {
        'diseases': [],
        'description': '',
        'treatment': [],
        'severity': 'Unknown',
        'pesticide': {
            'chemical': {
                'name': 'Combined Chemical Treatment',
                'usage': [],
                'frequency': [],
                'safety': [],
                'dosage_per_hectare': 0,
                'unit': 'L',
                'video_sources': {
                    'search_terms': [],
                    'reliable_channels': []
                }
            },
            'organic': {
                'name': 'Combined Organic Treatment',
                'usage': [],
                'frequency': [],
                'safety': [],
                'dosage_per_hectare': 0,
                'unit': 'L',
                'video_sources': {
                    'search_terms': [],
                    'reliable_channels': []
                }
            }
        },
        'additional_resources': {
            'step_by_step_guide': [],
            'extension_guides': []
        }
    }
    
    severity_levels = {'Low': 1, 'Moderate': 2, 'Medium': 2, 'High': 3, 'Severe': 4}
    max_severity_score = 0
    
    unique_chemical_names = set()
    unique_organic_names = set()
    unique_treatments = set()
    unique_guides = set()
    
    logger.info(f"📊 Processing {len(unique_diseases)} diseases...")
    
    for disease, data in unique_diseases.items():
        disease_info = data['disease_info']
        if not disease_info:
            logger.warning(f"⚠️ No disease info for {disease}")
            continue
        
        logger.info(f"   Processing: {disease}")
        
        combined['diseases'].append({
            'name': disease,
            'display_name': disease.replace('_', ' '),
            'count': data['count'],
            'avg_confidence': data['total_confidence'] / data['count']
        })
        
        if disease_info.get('description'):
            combined['description'] += f"**{disease.replace('_', ' ')}**: {disease_info['description']}\n\n"
        
        if disease_info.get('treatment'):
            header = f"=== Treatment for {disease.replace('_', ' ')} ==="
            if header not in unique_treatments:
                combined['treatment'].append(header)
                unique_treatments.add(header)
                
                for step in disease_info['treatment']:
                    if step and step not in unique_treatments:
                        combined['treatment'].append(step)
                        unique_treatments.add(step)
                
                combined['treatment'].append("")
        
        disease_severity = disease_info.get('severity', 'Unknown')
        severity_score = severity_levels.get(disease_severity, 0)
        if severity_score > max_severity_score:
            max_severity_score = severity_score
            combined['severity'] = disease_severity
            logger.info(f"   Updated max severity: {disease_severity}")
        
        if 'pesticide' in disease_info:
            for treatment_type in ['chemical', 'organic']:
                if treatment_type not in disease_info['pesticide']:
                    continue
                
                treatment = disease_info['pesticide'][treatment_type]
                unique_set = unique_chemical_names if treatment_type == 'chemical' else unique_organic_names
                
                treatment_name = treatment.get('name', '')
                if treatment_name and treatment_name not in unique_set:
                    unique_set.add(treatment_name)
                    usage_text = f"**{treatment_name}** ({disease.replace('_', ' ')}): {treatment.get('usage', 'Apply as directed')}"
                    combined['pesticide'][treatment_type]['usage'].append(usage_text)
                    
                    logger.info(f"      Added {treatment_type}: {treatment_name}")
                
                if treatment.get('frequency'):
                    freq = treatment['frequency'].strip()
                    if freq not in combined['pesticide'][treatment_type]['frequency']:
                        combined['pesticide'][treatment_type]['frequency'].append(freq)
                
                if treatment.get('safety'):
                    safety = treatment['safety'].strip()
                    if safety not in combined['pesticide'][treatment_type]['safety']:
                        combined['pesticide'][treatment_type]['safety'].append(safety)
                
                dosage = treatment.get('dosage_per_hectare', 0)
                combined['pesticide'][treatment_type]['dosage_per_hectare'] += dosage
                
                if treatment.get('video_sources'):
                    video_sources = treatment['video_sources']
                    
                    if 'search_terms' in video_sources:
                        for term in video_sources['search_terms']:
                            if term not in combined['pesticide'][treatment_type]['video_sources']['search_terms']:
                                combined['pesticide'][treatment_type]['video_sources']['search_terms'].append(term)
                    
                    if 'reliable_channels' in video_sources:
                        for channel in video_sources['reliable_channels']:
                            if channel not in combined['pesticide'][treatment_type]['video_sources']['reliable_channels']:
                                combined['pesticide'][treatment_type]['video_sources']['reliable_channels'].append(channel)
        
        if 'additional_resources' in disease_info:
            resources = disease_info['additional_resources']
            
            if 'step_by_step_guide' in resources:
                for step in resources['step_by_step_guide']:
                    if step not in combined['additional_resources']['step_by_step_guide']:
                        combined['additional_resources']['step_by_step_guide'].append(step)
            
            if 'extension_guides' in resources:
                for guide in resources['extension_guides']:
                    if guide not in unique_guides:
                        combined['additional_resources']['extension_guides'].append(guide)
                        unique_guides.add(guide)
    
    logger.info("📝 Formatting combined treatment data...")
    
    for treatment_type in ['chemical', 'organic']:
        if combined['pesticide'][treatment_type]['usage']:
            combined['pesticide'][treatment_type]['usage'] = "\n\n".join(
                combined['pesticide'][treatment_type]['usage']
            )
        else:
            combined['pesticide'][treatment_type]['usage'] = "Apply treatments according to product labels for each specific disease."
        
        if combined['pesticide'][treatment_type]['frequency']:
            unique_freq = list(set(combined['pesticide'][treatment_type]['frequency']))
            if len(unique_freq) == 1:
                combined['pesticide'][treatment_type]['frequency'] = unique_freq[0]
            else:
                combined['pesticide'][treatment_type]['frequency'] = " OR ".join(unique_freq)
        else:
            combined['pesticide'][treatment_type]['frequency'] = "Follow individual disease treatment schedules"
        
        if combined['pesticide'][treatment_type]['safety']:
            combined['pesticide'][treatment_type]['safety'] = " • ".join(
                list(set(combined['pesticide'][treatment_type]['safety']))
            )
        else:
            combined['pesticide'][treatment_type]['safety'] = "Follow all safety guidelines on product labels. Wear protective equipment."
        
        num_diseases = len(unique_diseases)
        if num_diseases > 0 and combined['pesticide'][treatment_type]['dosage_per_hectare'] > 0:
            combined['pesticide'][treatment_type]['dosage_per_hectare'] /= num_diseases
            logger.info(f"   {treatment_type.title()} avg dosage: {combined['pesticide'][treatment_type]['dosage_per_hectare']:.2f}")
        
        video_sources = combined['pesticide'][treatment_type]['video_sources']
        if video_sources['search_terms']:
            search_urls = []
            for term in video_sources['search_terms']:
                search_urls.append({
                    'term': term,
                    'url': f"https://www.youtube.com/results?search_query={quote_plus(term)}"
                })
            video_sources['search_urls'] = search_urls
        
        if video_sources['reliable_channels']:
            channel_urls = []
            for channel in video_sources['reliable_channels']:
                channel_urls.append({
                    'name': channel,
                    'url': f"https://www.youtube.com/results?search_query={quote_plus(channel + ' multiple plant diseases')}"
                })
            video_sources['channel_urls'] = channel_urls
    
    logger.info("=" * 80)
    logger.info("✅ COMBINED TREATMENT PLAN READY")
    logger.info(f"   Diseases: {len(combined['diseases'])}")
    logger.info(f"   Treatment steps: {len(combined['treatment'])}")
    logger.info(f"   Overall severity: {combined['severity']}")
    logger.info("=" * 80)
    
    return combined


def calculate_dosage(area, area_unit, pesticide_info):
    """Calculate pesticide dosage based on area and unit with enhanced error handling"""
    logger.info("="*60)
    logger.info("🧮 DOSAGE CALCULATION STARTED")
    logger.info("="*60)
    logger.info(f"📏 Input area: {area} {area_unit}")
    logger.info(f"📋 Pesticide info exists: {pesticide_info is not None}")
    
    try:
        chemical_dosage = None
        organic_dosage = None
        hectare_conversion = 0
        
        chemical_info = pesticide_info.get("chemical", {}) if pesticide_info else {}
        organic_info = pesticide_info.get("organic", {}) if pesticide_info else {}
        
        logger.info(f"💊 Chemical info available: {bool(chemical_info)}")
        logger.info(f"💊 Chemical info keys: {list(chemical_info.keys()) if chemical_info else []}")
        logger.info(f"🌿 Organic info available: {bool(organic_info)}")
        logger.info(f"🌿 Organic info keys: {list(organic_info.keys()) if organic_info else []}")
        
        chemical_dosage_per_hectare = float(chemical_info.get("dosage_per_hectare", 0))
        organic_dosage_per_hectare = float(organic_info.get("dosage_per_hectare", 0))
        
        logger.info(f"💊 Chemical dosage per hectare: {chemical_dosage_per_hectare}")
        logger.info(f"🌿 Organic dosage per hectare: {organic_dosage_per_hectare}")
        
        try:
            area_float = float(area) if area else 0
            if area_float <= 0:
                logger.warning(f"⚠️ Invalid or zero area value: {area}")
                logger.warning(f"⚠️ Using default 1 hectare for dosage display")
                area_float = 1.0
                
            conversion_factors = {
                'hectare': 1.0,
                'acre': 0.404686,
                'square_meter': 0.0001,
                'square_feet': 0.0000092903
            }
            
            hectare_conversion = area_float * conversion_factors.get(area_unit, 1.0)
            logger.info(f"📐 Converted {area_float} {area_unit} to {hectare_conversion} hectares")
            
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Error converting area to float: {e}")
            logger.error("❌ Using default 1 hectare")
            hectare_conversion = 1.0
        
        if chemical_dosage_per_hectare > 0:
            chemical_dosage = chemical_dosage_per_hectare * hectare_conversion
            logger.info(f"✅ Calculated chemical dosage: {chemical_dosage}")
        else:
            logger.warning("⚠️ Chemical dosage_per_hectare is 0 or missing")
        
        if organic_dosage_per_hectare > 0:
            organic_dosage = organic_dosage_per_hectare * hectare_conversion
            logger.info(f"✅ Calculated organic dosage: {organic_dosage}")
        else:
            logger.warning("⚠️ Organic dosage_per_hectare is 0 or missing")
        
        logger.info("="*60)
        logger.info(f"🎯 FINAL RESULTS:")
        logger.info(f"   Chemical dosage: {chemical_dosage}")
        logger.info(f"   Organic dosage: {organic_dosage}")
        logger.info(f"   Hectare conversion: {hectare_conversion}")
        logger.info("="*60)
        
        return chemical_dosage, organic_dosage, hectare_conversion
        
    except Exception as e:
        logger.error("="*60)
        logger.error(f"❌ ERROR IN DOSAGE CALCULATION")
        logger.error(f"❌ Error: {e}")
        logger.error(traceback.format_exc())
        logger.error("="*60)
        return None, None, 0


def get_detailed_error_message(error_type, image_analysis=None):
    """Generate detailed error messages for different validation failures"""
    if error_type == "not_plant":
        return {
            "title": "Not a Plant Image",
            "message": "The uploaded image doesn't appear to be a plant photograph.",
            "suggestions": [
                "Upload a clear photo of plant leaves",
                "Ensure the image shows actual plant matter (not drawings or posters)",
                "Make sure leaves are visible with any disease symptoms",
                "Use good lighting and focus on the affected plant parts"
            ],
            "technical_details": image_analysis
        }
    elif error_type == "low_confidence":
        return {
            "title": "Unable to Identify Plant Disease",
            "message": "The image quality or plant type may not be suitable for accurate analysis.",
            "suggestions": [
                "Try uploading a clearer, higher quality image",
                "Ensure the plant is one of our supported types",
                "Focus on leaves showing clear disease symptoms",
                "Check if lighting is adequate"
            ]
        }
    elif error_type == "unsupported_plant":
        return {
            "title": "Unsupported Plant Type",
            "message": "This plant type may not be in our current database.",
            "suggestions": [
                "Check our supported plants list",
                "Try with Apple, Tomato, Potato, Corn, Grape, Peach, Pepper, or Strawberry plants",
                "Ensure the image clearly shows the plant type"
            ]
        }
    else:
        return {
            "title": "Analysis Error",
            "message": "An error occurred during image analysis.",
            "suggestions": [
                "Try uploading the image again",
                "Ensure the image file is not corrupted",
                "Use a different image format (JPG, PNG)"
            ]
        }


def initialize_enhanced_gemini():
    """Enhanced Gemini AI initialization with better error handling"""
    try:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your-api-key-here":
            logger.error("Gemini API key not configured properly")
            return False, "API key not configured"
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        test_model = genai.GenerativeModel('models/gemini-1.5-flash-001')
        test_prompt = "What is the most important factor in plant health?"
        
        test_response = test_model.generate_content(test_prompt)
        
        if test_response and test_response.text:
            logger.info("✅ Gemini AI connected successfully!")
            logger.info(f"Test response: {test_response.text[:100]}...")
            return True, "Connected successfully"
        else:
            logger.error("❌ Gemini AI test failed - no response received")
            return False, "No response from API"
            
    except Exception as e:
        logger.error(f"❌ Gemini AI initialization failed: {str(e)}")
        return False, str(e)


def get_enhanced_chatbot_response(message, detected_disease=None, conversation_history=None):
    """Enhanced chatbot with improved AI integration and common questions"""
    
    original_message = message
    message = message.lower().strip()
    
    logger.info(f"Enhanced chatbot processing: {original_message}")
    
    if message in ["help", "/help", "commands", "/commands"]:
        return generate_help_response()
    
    elif message in ["questions", "/questions", "common questions", "examples"]:
        return generate_common_questions_response()
    
    elif message.startswith("/category "):
        category = message.replace("/category ", "").strip()
        return generate_category_questions(category)
    
    elif any(keyword in message for keyword in ["date", "time", "today", "current date", "current time"]):
        current_datetime = datetime.now()
        if "time" in message:
            return f"🕐 Current time: {current_datetime.strftime('%H:%M:%S')} IST"
        elif "date" in message:
            return f"📅 Today's date: {current_datetime.strftime('%B %d, %Y (%A)')}"
        else:
            return f"📅🕐 Current date and time: {current_datetime.strftime('%B %d, %Y %H:%M:%S (%A)')}"
    
    elif any(greeting in message for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "namaste", "start"]):
        greeting_response = """🌱 **Namaste! Welcome to AgriPal AI!** 

I'm your intelligent agricultural assistant powered by advanced AI. I can help you with:

🔍 **Disease Detection** - Upload images for instant plant disease identification
💊 **Treatment Plans** - Get specific, science-based treatment recommendations  
🧮 **Dosage Calculator** - Calculate exact pesticide amounts for your farm size
🌿 **Organic Solutions** - Eco-friendly pest and disease management
📊 **Crop Management** - Seasonal advice and farming best practices
🤖 **AI-Powered Q&A** - Ask any agricultural question, get expert answers

**Quick Start Commands:**
- Type `questions` to see common agricultural questions
- Type `help` to see available commands
- Ask specific questions like "How to treat tomato blight?"

What would you like to explore today? 🚀"""
        return greeting_response
    
    elif any(farewell in message for farewell in ["bye", "goodbye", "see you", "thanks", "thank you", "dhanyawad"]):
        return """🙏 **Thank you for using AgriPal AI!** 

**Remember these key farming tips:**
- Monitor your crops regularly for early disease detection
- Maintain good field hygiene and crop rotation
- Keep learning about sustainable farming practices

Happy farming! 🌾🚜✨"""
    
    else:
        try:
            ai_prompt = create_agricultural_prompt(original_message, detected_disease, conversation_history)
            
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 500,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-001",
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            response = gemini_model.generate_content(ai_prompt)
            
            if response and response.text:
                ai_response = response.text.strip()
                formatted_response = f"🤖 **AgriPal AI Expert Response:**\n\n{ai_response}\n\n---\n💡 *Need plant disease identification? Upload an image using our detection tool!*"
                return formatted_response
            else:
                return get_fallback_response(original_message, detected_disease)
                
        except Exception as e:
            logger.error(f"❌ Gemini AI error: {str(e)}")
            return get_fallback_response(original_message, detected_disease, str(e))


def get_common_questions_by_category(category=None, limit=5):
    """Get common questions by category or random selection"""
    if category and category in COMMON_QUESTIONS:
        questions = COMMON_QUESTIONS[category]
        return random.sample(questions, min(limit, len(questions)))
    else:
        all_questions = []
        for cat_questions in COMMON_QUESTIONS.values():
            all_questions.extend(cat_questions)
        return random.sample(all_questions, min(limit, len(all_questions)))


def create_agricultural_prompt(user_message, detected_disease=None, conversation_history=None):
    """Create a comprehensive prompt for agricultural AI assistant"""
    
    base_context = """You are AgriPal AI, an expert agricultural assistant specializing in:
- Plant disease identification and treatment
- Crop management and farming techniques
- Pest control and integrated pest management
- Soil health and nutrition management
- Seasonal farming advice
- Sustainable agriculture practices

Your responses should be:
- Scientifically accurate and evidence-based
- Practical and actionable for farmers
- Safe and environmentally conscious
- Appropriate for different farming scales
- Under 400 words but comprehensive"""

    disease_context = ""
    if detected_disease:
        disease_context = f"\n\nCurrent Context: The user recently detected '{detected_disease}'. This should inform your responses."

    history_context = ""
    if conversation_history:
        recent_messages = conversation_history[-3:]
        history_summary = " ".join([msg.get('text', '')[:100] for msg in recent_messages])
        history_context = f"\n\nRecent conversation: {history_summary}"

    user_context = f"\n\nUser's question: {user_message}"
    
    return base_context + disease_context + history_context + user_context


def generate_help_response():
    """Generate help response with available commands"""
    return """🆘 **AgriPal AI Help Center**

**Available Commands:**
- `help` - Show this help menu
- `questions` - View common agricultural questions
- `/category [name]` - Get questions by category

**Categories:**
- `plant_diseases` - Disease identification
- `treatment_methods` - Treatment options
- `crop_management` - Farming practices
- `seasonal_advice` - Season-specific guidance
- `technology_agriculture` - Modern farming tech

**Example Questions:**
- "What causes yellow leaves in tomatoes?"
- "How to make organic pesticide?"
- "Best time to plant vegetables?"

Just type your question naturally! 🌱"""


def generate_common_questions_response():
    """Generate response with common questions"""
    questions = get_common_questions_by_category(limit=8)
    
    response = "❓ **Popular Agricultural Questions**\n\n"
    
    for i, question in enumerate(questions, 1):
        response += f"**{i}.** {question}\n"
    
    response += "\n**More Help:** Type `/category plant_diseases` for specific topics!"
    return response


def generate_category_questions(category):
    """Generate questions for a specific category"""
    if category not in COMMON_QUESTIONS:
        available_categories = ", ".join(COMMON_QUESTIONS.keys())
        return f"❓ Category '{category}' not found.\n\n**Available:** {available_categories}"
    
    questions = COMMON_QUESTIONS[category]
    category_title = category.replace('_', ' ').title()
    
    response = f"📚 **{category_title} - Questions**\n\n"
    
    for i, question in enumerate(questions, 1):
        response += f"**{i}.** {question}\n"
    
    return response


def get_fallback_response(original_message, detected_disease=None, error_msg=None):
    """Enhanced fallback response when AI is unavailable"""
    
    fallback = f"""🤖 **AgriPal AI Assistant** *(Offline Mode)*

**Your question:** "{original_message}"

"""
    
    if detected_disease:
        fallback += f"**Detected disease:** {detected_disease}\n\n"
    
    message_lower = original_message.lower()
    
    if any(word in message_lower for word in ["disease", "fungus", "infection"]):
        fallback += """**For plant diseases:**
🔍 Take clear photos of affected areas
✂️ Remove diseased plant parts
🌿 Apply appropriate treatment
📞 Consult agricultural extension officer"""
    
    elif any(word in message_lower for word in ["treatment", "pesticide", "spray"]):
        fallback += """**Treatment guidelines:**
🧪 Use registered pesticides as per label
🌱 Try organic options (neem oil, copper sulfate)
⏰ Apply during cool hours
⚠️ Always wear protective equipment"""
    
    fallback += "\n\n**Try:** `questions` for common topics or `help` for commands"
    
    return fallback


# ===== WEEKLY ASSESSMENT FUNCTIONS =====

def analyze_weekly_progress(user_id, plant_type, current_detection):
    """
    Analyze weekly progress and generate treatment recommendations.
    Returns dict with assessment, comparison, and recommendations.
    """
    logger.info("=" * 80)
    logger.info("📊 WEEKLY ASSESSMENT ANALYSIS")
    logger.info("=" * 80)
    
    previous_assessments = WeeklyAssessment.query.filter_by(
        user_id=user_id,
        plant_type=plant_type
    ).order_by(WeeklyAssessment.week_number.desc()).limit(4).all()
    
    if not previous_assessments:
        logger.info("🆕 First assessment for this plant")
        return {
            'is_first_assessment': True,
            'week_number': 1,
            'recommendation': 'Start treatment as recommended. Take photos weekly to track progress.',
            'dosage_recommendation': 'maintain',
            'next_assessment_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        }
    
    last_week = previous_assessments[0]
    week_number = last_week.week_number + 1
    
    logger.info(f"📅 Week {week_number} Assessment")
    logger.info(f"📈 Previous Week {last_week.week_number}: {last_week.severity_level}")
    
    severity_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Severe': 4}
    
    current_severity_score = severity_map.get(current_detection['severity'], 0)
    previous_severity_score = last_week.severity_score
    
    severity_change = current_severity_score - previous_severity_score
    
    current_affected = current_detection.get('color_severity', 0)
    previous_affected = last_week.color_severity_percent or 0
    area_change_percent = current_affected - previous_affected
    
    is_improving = severity_change < 0 or area_change_percent < -5
    is_worsening = severity_change > 0 or area_change_percent > 5
    is_stable = abs(severity_change) == 0 and abs(area_change_percent) <= 5
    is_cured = current_detection.get('disease', '').lower().endswith('healthy')
    
    logger.info(f"📊 Progress Analysis:")
    logger.info(f"   - Severity change: {severity_change}")
    logger.info(f"   - Area change: {area_change_percent:+.1f}%")
    logger.info(f"   - Improving: {is_improving}")
    logger.info(f"   - Worsening: {is_worsening}")
    logger.info(f"   - Cured: {is_cured}")
    
    recommendation, dosage_change, switch_treatment = generate_treatment_recommendation(
        is_improving, is_worsening, is_stable, is_cured,
        week_number, last_week, current_detection
    )
    
    assessment_result = {
        'is_first_assessment': False,
        'week_number': week_number,
        'previous_week_severity': last_week.severity_level,
        'current_severity': current_detection['severity'],
        'severity_change': severity_change,
        'area_change_percent': area_change_percent,
        'is_improving': is_improving,
        'is_worsening': is_worsening,
        'is_stable': is_stable,
        'is_cured': is_cured,
        'recommendation': recommendation,
        'dosage_recommendation': dosage_change,
        'treatment_switch': switch_treatment,
        'previous_treatment': last_week.pesticide_used,
        'previous_dosage': last_week.dosage_applied,
        'next_assessment_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
        'assessment_history': [
            {
                'week': a.week_number,
                'date': a.assessment_date.strftime('%Y-%m-%d'),
                'severity': a.severity_level,
                'affected_area': a.affected_area_percent,
                'treatment': a.pesticide_used
            } for a in reversed(previous_assessments)
        ]
    }
    
    logger.info("=" * 80)
    logger.info(f"✅ Assessment Complete: {recommendation[:100]}...")
    logger.info("=" * 80)
    
    return assessment_result


def generate_treatment_recommendation(is_improving, is_worsening, is_stable, 
                                     is_cured, week_number, last_assessment, 
                                     current_detection):
    """Generate AI-powered treatment recommendations based on progress"""
    
    if is_cured:
        return (
            "🎉 Excellent! Your plant has fully recovered! "
            "Continue with preventive care: maintain good hygiene, proper watering, "
            "and monitor weekly. No pesticides needed unless symptoms reappear.",
            "stop",
            None
        )
    
    if is_improving:
        if week_number <= 2:
            recommendation = (
                "✅ Great progress! Disease severity is decreasing. "
                "Continue with current treatment plan. "
                "Keep dosage the same for one more week to ensure effectiveness."
            )
            dosage_change = "maintain"
            switch = None
        else:
            recommendation = (
                f"✅ Continued improvement over {week_number} weeks! "
                f"You can now REDUCE pesticide dosage by 25-30% as the plant is responding well. "
                f"Previous dosage: {last_assessment.dosage_applied:.2f}L - "
                f"Reduce to: {last_assessment.dosage_applied * 0.70:.2f}L. "
                f"This reduces chemical load while maintaining effectiveness."
            )
            dosage_change = "decrease_25"
            switch = None
        
        return recommendation, dosage_change, switch
    
    if is_stable:
        if week_number >= 3:
            current_type = last_assessment.pesticide_type
            switch_to = "organic" if current_type == "chemical" else "stronger chemical"
            
            recommendation = (
                f"⚠️ Disease is stable but not improving after {week_number} weeks. "
                f"Current treatment ({last_assessment.pesticide_used}) may not be fully effective. "
                f"RECOMMENDATION: Switch to {switch_to} treatment alternative. "
                f"Also increase application frequency."
            )
            dosage_change = "maintain_or_increase"
            switch = switch_to
        else:
            recommendation = (
                "📊 Disease severity is stable. Continue current treatment "
                "but monitor closely. If no improvement by next week, "
                "we'll recommend switching treatments."
            )
            dosage_change = "maintain"
            switch = None
        
        return recommendation, dosage_change, switch
    
    if is_worsening:
        if week_number <= 2:
            recommendation = (
                "⚠️ WARNING: Disease is progressing despite treatment! "
                f"IMMEDIATE ACTION NEEDED: "
                f"1. INCREASE dosage by 30-40% "
                f"(from {last_assessment.dosage_applied:.2f}L to "
                f"{last_assessment.dosage_applied * 1.35:.2f}L). "
                f"2. Increase application frequency. "
                f"3. Remove and destroy heavily infected plant parts. "
                f"4. Improve field sanitation."
            )
            dosage_change = "increase_35"
            switch = None
        else:
            current_type = last_assessment.pesticide_type
            
            if current_type == "organic":
                recommendation = (
                    f"🚨 CRITICAL: Disease worsening after {week_number} weeks of organic treatment. "
                    f"URGENT RECOMMENDATION: Switch to CHEMICAL pesticides immediately. "
                    f"Organic methods are not controlling the infection. "
                    f"Suggested: Use systemic fungicide/pesticide for this disease. "
                    f"Consider consulting agricultural extension officer."
                )
                switch = "chemical_systemic"
            else:
                recommendation = (
                    f"🚨 CRITICAL: Disease worsening despite chemical treatment for {week_number} weeks. "
                    f"URGENT ACTIONS: "
                    f"1. Switch to DIFFERENT chemical class (avoid resistance) "
                    f"2. Increase dosage by 40% "
                    f"3. Apply every 5 days instead of weekly "
                    f"4. Consider professional consultation "
                    f"5. Test if disease strain is pesticide-resistant"
                )
                switch = "different_chemical_class"
            
            dosage_change = "increase_40"
        
        return recommendation, dosage_change, switch
    
    return (
        "Continue monitoring. Take clear photos weekly for accurate tracking.",
        "maintain",
        None
    )


def save_weekly_assessment(user_id, plant_type, detection_data, assessment_result):
    """Save the weekly assessment to database"""
    try:
        severity_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Severe': 4}
        
        assessment = WeeklyAssessment(
            user_id=user_id,
            plant_type=plant_type,
            disease_name=detection_data.get('disease', 'Unknown'),
            week_number=assessment_result['week_number'],
            assessment_date=datetime.now(),
            
            severity_level=detection_data.get('severity', 'Unknown'),
            severity_score=severity_map.get(detection_data.get('severity', 'Unknown'), 0),
            color_severity_percent=detection_data.get('color_severity', 0),
            affected_area_percent=detection_data.get('affected_percentage', 0),
            
            pesticide_used=detection_data.get('pesticide_used', 'Not specified'),
            pesticide_type=detection_data.get('pesticide_type', 'chemical'),
            dosage_applied=detection_data.get('dosage_applied', 0),
            application_method=detection_data.get('application_method', 'Spray'),
            
            is_improving=assessment_result.get('is_improving', False),
            is_worsening=assessment_result.get('is_worsening', False),
            is_stable=assessment_result.get('is_stable', False),
            is_cured=assessment_result.get('is_cured', False),
            
            recommendation=assessment_result.get('recommendation', ''),
            recommended_dosage_change=assessment_result.get('dosage_recommendation', 'maintain'),
            recommended_switch=assessment_result.get('treatment_switch'),
            
            image_filename=detection_data.get('image_filename'),
            
            farmer_notes=detection_data.get('farmer_notes', '')
        )
        
        db.session.add(assessment)
        db.session.commit()
        
        logger.info(f"✅ Weekly assessment saved: Week {assessment_result['week_number']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving weekly assessment: {e}")
        db.session.rollback()
        return False


def startup_gemini_check():
    """Check Gemini AI status on startup"""
    logger.info("🚀 Initializing Enhanced AgriPal Chatbot...")
    
    success, message = initialize_enhanced_gemini()
    
    if success:
        logger.info(f"✅ Chatbot Ready: {message}")
    else:
        logger.warning(f"⚠️ AI Limited Mode: {message}")
    
    return success


# ===== REGISTER BLUEPRINTS =====
app.register_blueprint(auth_bp)
app.register_blueprint(post_harvest_bp)
app.register_blueprint(schemes_bp)


# ===== SESSION MANAGEMENT =====
def clear_sessions_on_startup():
    session_dir = './.flask_session/'
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir)
            logger.info("🗑️ Cleared all previous sessions")
        except Exception as e:
            logger.error(f"❌ Error clearing sessions: {e}")
    os.makedirs(session_dir, exist_ok=True)


def init_database():
    with app.app_context():
        db.create_all()
        logger.info("✅ Database tables created successfully")


# ===== SESSION VALIDATION MIDDLEWARE =====
@app.before_request
def validate_session():
    if request.endpoint and (
        request.endpoint.startswith('static') or 
        request.endpoint.startswith('auth.') or
        request.endpoint == 'index' or
        request.endpoint == 'health_check' or
        request.endpoint == 'api_info'
    ):
        return
    
    if current_user.is_authenticated:
        if 'session_start' not in session:
            logout_user()
            session.clear()
            flash('Your session has expired. Please login again.', 'warning')
            return redirect(url_for('auth.login'))
        
        session_server_start = session.get('server_start')
        current_server_start = app.config['SERVER_START_TIME']
        
        if session_server_start != current_server_start:
            logout_user()
            session.clear()
            flash('Server was restarted. Please login again.', 'info')
            return redirect(url_for('auth.login'))


# ===== DISEASE HISTORY COMPARISON =====
def check_previous_detection(user_id, plant_type):
    """Check if user has previous detection of same plant within last 30 days"""
    try:
        one_month_ago = datetime.now() - timedelta(days=30)
        
        previous = DiseaseDetection.query.filter(
            DiseaseDetection.user_id == user_id,
            DiseaseDetection.plant_type == plant_type,
            DiseaseDetection.detection_time >= one_month_ago
        ).order_by(DiseaseDetection.detection_time.desc()).first()
        
        if previous:
            days_ago = (datetime.now() - previous.detection_time).days
            logger.info(f"📊 Found previous {plant_type} detection from {days_ago} days ago")
            return True, previous, days_ago
        
        return False, None, 0
        
    except Exception as e:
        logger.error(f"Error checking previous detection: {e}")
        return False, None, 0


def compare_disease_progress(previous_detection, current_severity, current_disease):
    """Compare current detection with previous and generate feedback"""
    severity_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Severe': 4}
    
    prev_severity_score = severity_map.get(previous_detection.severity, 0)
    curr_severity_score = severity_map.get(current_severity, 0)
    
    comparison = {
        'previous_disease': previous_detection.detected_disease,
        'previous_severity': previous_detection.severity,
        'current_disease': current_disease,
        'current_severity': current_severity,
        'days_since_last': (datetime.now() - previous_detection.detection_time).days,
        'improved': False,
        'worsened': False,
        'same': False,
        'message': '',
        'recommendation': ''
    }
    
    if previous_detection.detected_disease == current_disease:
        if curr_severity_score < prev_severity_score:
            comparison['improved'] = True
            comparison['message'] = f"🎉 Great news! Your {previous_detection.plant_type} is improving! Severity reduced from {previous_detection.severity} to {current_severity}."
            comparison['recommendation'] = "Continue with your current treatment plan. Keep monitoring regularly."
        
        elif curr_severity_score > prev_severity_score:
            comparison['worsened'] = True
            comparison['message'] = f"⚠️ Alert: Disease severity has increased from {previous_detection.severity} to {current_severity}."
            comparison['recommendation'] = "Current treatment may not be effective. Consider switching to stronger alternatives or consult an expert."
        
        else:
            comparison['same'] = True
            comparison['message'] = f"📊 Disease severity remains {current_severity}."
            comparison['recommendation'] = "Continue treatment. If no improvement in next week, consider alternative methods."
    
    else:
        if 'healthy' in current_disease.lower():
            comparison['improved'] = True
            comparison['message'] = f"🌟 Excellent! Your plant has recovered from {previous_detection.detected_disease}!"
            comparison['recommendation'] = "Maintain good crop management practices to prevent future infections."
        else:
            comparison['worsened'] = True
            comparison['message'] = f"⚠️ New disease detected: {current_disease} (previously: {previous_detection.detected_disease})"
            comparison['recommendation'] = "Multiple diseases detected. Implement comprehensive disease management strategy."
    
    return comparison


# ===== HEALTH CHECK ENDPOINT FOR RENDER =====
@app.route('/health')
def health_check():
    """Health check endpoint for Render monitoring"""
    try:
        db_status = 'connected'
        try:
            db.session.execute('SELECT 1')
        except:
            db_status = 'disconnected'
        
        uptime = datetime.now() - SERVER_START_TIME
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': int(uptime.total_seconds()),
            'database': db_status,
            'model': 'offline_maintenance_mode',
            'gemini_api': 'configured' if GEMINI_API_KEY else 'not_configured'
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503


@app.route('/')
def index():
    """Landing page - shows beautiful page for guests, dashboard for logged-in users"""
    if current_user.is_authenticated:
        logger.info(f"Authenticated user {current_user.username} accessing root - redirecting to dashboard")
        return redirect(url_for('dashboard'))
    
    logger.info("Guest user accessing landing page")
    return render_template('index2.html')


@app.route('/chatbot')
@login_required
def chatbot_page():
    logger.info(f"User {current_user.username} accessing chatbot")
    return render_template('chatbot.html')


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with statistics and history"""
    total_detections = DiseaseDetection.query.filter_by(user_id=current_user.id).count()
    recent_detections = DiseaseDetection.query.filter_by(user_id=current_user.id)\
        .order_by(DiseaseDetection.detection_time.desc()).limit(10).all()
    
    total_assessments = WeeklyAssessment.query.filter_by(user_id=current_user.id).count()
    
    disease_stats = db.session.query(
        DiseaseDetection.detected_disease,
        db.func.count(DiseaseDetection.id).label('count')
    ).filter_by(user_id=current_user.id)\
     .group_by(DiseaseDetection.detected_disease)\
     .order_by(db.text('count DESC'))\
     .limit(5).all()
    
    plant_stats = db.session.query(
        DiseaseDetection.plant_type,
        db.func.count(DiseaseDetection.id).label('count')
    ).filter_by(user_id=current_user.id)\
     .group_by(DiseaseDetection.plant_type)\
     .all()
    
    weekly_assessments_raw = WeeklyAssessment.query.filter_by(user_id=current_user.id)\
        .order_by(WeeklyAssessment.plant_type, WeeklyAssessment.week_number.desc())\
        .all()
    
    weekly_assessments = {}
    for assessment in weekly_assessments_raw:
        plant_type = assessment.plant_type
        if plant_type not in weekly_assessments:
            weekly_assessments[plant_type] = []
        weekly_assessments[plant_type].append(assessment)
    
    return render_template('dashboard.html',
        total_detections=total_detections,
        total_assessments=total_assessments,
        recent_detections=recent_detections,
        disease_stats=disease_stats,
        plant_stats=plant_stats,
        weekly_assessments=weekly_assessments,
        timedelta=timedelta
    )


@app.route('/api/chat/enhanced', methods=['POST'])
def enhanced_chat_api():
    """Enhanced API endpoint with better features"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        conversation_history = data.get('history', [])
        detected_disease = data.get('detected_disease')
        
        if not detected_disease:
            try:
                with open('detected_disease.json', 'r') as f:
                    disease_data = json.load(f)
                    detected_disease = disease_data.get('disease')
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        response_text = get_enhanced_chatbot_response(
            user_message, 
            detected_disease, 
            conversation_history
        )
        
        return jsonify({
            'success': True,
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'detected_disease': detected_disease,
            'ai_status': 'online' if gemini_status else 'offline'
        })
        
    except Exception as e:
        logger.error(f"Enhanced Chat API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your message',
            'details': str(e)
        }), 500


@app.route('/api/chat/common-questions')
def get_common_questions_api():
    """API to get common questions by category"""
    try:
        category = request.args.get('category')
        limit = int(request.args.get('limit', 10))
        
        if category:
            if category not in COMMON_QUESTIONS:
                return jsonify({
                    'success': False,
                    'error': f'Category not found: {category}',
                    'available_categories': list(COMMON_QUESTIONS.keys())
                }), 404
            
            questions = get_common_questions_by_category(category, limit)
            return jsonify({
                'success': True,
                'category': category,
                'questions': questions,
                'total': len(questions)
            })
        else:
            all_categories = {}
            for cat, questions in COMMON_QUESTIONS.items():
                all_categories[cat] = {
                    'title': cat.replace('_', ' ').title(),
                    'sample_questions': questions[:3],
                    'total_questions': len(questions)
                }
            
            return jsonify({
                'success': True,
                'categories': all_categories,
                'total_questions': sum(len(q) for q in COMMON_QUESTIONS.values())
            })
            
    except Exception as e:
        logger.error(f"Common questions API error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat/system-status')
def chat_status():
    """Get current chat status and context"""
    try:
        detected_disease = None
        detection_time = None
        
        try:
            with open('detected_disease.json', 'r') as f:
                disease_data = json.load(f)
                detected_disease = disease_data.get('disease')
                detection_time = disease_data.get('timestamp')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return jsonify({
            'success': True,
            'gemini_available': GEMINI_API_KEY is not None,
            'detected_disease': detected_disease,
            'detection_time': detection_time,
            'model_loaded': False,
            'supported_plants': len(SUPPORTED_PLANTS)
        })
    except Exception as e:
        logger.error(f"Chat status error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/user-data', methods=['GET'])
@login_required
def get_user_data():
    """Fetch user's saved location and land size."""
    try:
        user = current_user
        
        user_data = {
            'location': user.location if hasattr(user, 'location') and user.location else '',
            'land_area': user.land_area if hasattr(user, 'land_area') and user.land_area else 0,
            'area_unit': user.area_unit if hasattr(user, 'area_unit') and user.area_unit else 'square_meter'
        }
        
        return jsonify(user_data), 200
        
    except Exception as e:
        return jsonify({
            'location': '',
            'land_area': 0,
            'area_unit': 'square_meter'
        }), 200


@app.route('/detection-tool')
@login_required
def detection_tool():
    logger.info(f"User {current_user.username} accessing detection tool")
    return render_template('detection-tool.html')


@app.route('/detection')
def detection():
    logger.info("Rendering detection page")
    return render_template('detection-tool.html')


@app.route('/about-us')
def about_us():
    logger.info("Rendering about-us page")
    return render_template('about-us.html')


@app.route('/contact')
def contact():
    logger.info("Rendering contact page")
    return render_template('contact.html')


@app.route('/library')
def library():
    logger.info("Rendering library page")
    return render_template('library.html')


@app.route('/post-harvest')
def post_harvest_page():
    """Render post-harvest management page"""
    logger.info("Rendering post-harvest management page")
    return render_template('post-harvest.html')


@app.route('/schemes')
def schemes_page():
    """Render government schemes page"""
    logger.info("Rendering schemes page")
    return render_template('schemes.html')


@app.route('/api/info')
def api_info():
    """Enhanced API information endpoint"""
    return jsonify({
        'message': 'AGRI_PAL Unified API',
        'version': '2.0',
        'status': 'running',
        'endpoints': {
            'disease_detection': {
                'predict': 'POST /predict',
                'supported_plants': 'GET /api/supported-plants',
                'treatment': 'GET /api/treatment/<disease_name>'
            },
            'post_harvest': {
                'agro_shops': 'POST /post-harvest/agro-shops',
                'markets': 'POST /post-harvest/markets',
                'storage': 'POST /post-harvest/storage'
            },
            'schemes': {
                'all_schemes': 'GET /api/schemes',
                'categories': 'GET /api/schemes/categories',
                'by_category': 'GET /api/schemes/category/<category>',
                'by_id': 'GET /api/schemes/<scheme_id>',
                'search': 'GET /api/schemes/search?q=<query>'
            },
            'chatbot': {
                'chat': 'POST /api/chat/enhanced',
                'common_questions': 'GET /api/chat/common-questions',
                'status': 'GET /api/chat/system-status'
            }
        }
    })


@app.route('/plant-library')
def plant_library():
    logger.info("Rendering plant library page")
    return render_template('library.html')


@app.route('/api/supported-plants')
def get_supported_plants():
    """API endpoint to get list of supported plants"""
    return jsonify({
        'supported_plants': SUPPORTED_PLANTS,
        'total_plants': len(SUPPORTED_PLANTS),
        'total_conditions': len(class_names)
    })


@app.route('/upload')
def upload_file():
    """Route for upload page - alias for detection tool"""
    logger.info("Upload file route accessed - redirecting to detection tool")
    return detection_tool()


from flask_login import login_required, current_user


# ===== LIGHTWEIGHT PLACEHOLDER /predict ROUTE =====
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """
    Lightweight placeholder predict route.
    ML model is offline for optimization - returns maintenance info.
    """
    try:
        image = request.files.get('image')
        location = request.form.get('location', '')

        if not image or image.filename == '':
            flash("Error: No image file uploaded.", "error")
            return render_template("error.html", back_link="/detection-tool")

        if not allowed_file(image.filename):
            flash("Error: Invalid file type. Please upload PNG, JPG, or JPEG.", "error")
            return render_template("error.html", back_link="/detection-tool")

        # Save the image so it can still be displayed
        image_filename = str(uuid.uuid4()) + os.path.splitext(image.filename)[1]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(image_path)

        logger.info(f"✅ Image saved (maintenance mode): {image_path}")

        # Placeholder info for stable deployment
        disease_info = {
            "name": "AI Analysis (Maintenance)",
            "description": "Our AI engine is currently offline for optimization. Please use the Chatbot for immediate care instructions.",
            "treatment": [
                "Inspect leaves for pests",
                "Maintain consistent watering",
                "Ensure adequate sunlight"
            ],
            "severity": "N/A"
        }

        return render_template(
            'result.html',
            result=disease_info,
            location=location,
            image_url=url_for('static', filename=f'uploads/{image_filename}')
        )

    except Exception as e:
        logger.error(f"❌ Error in placeholder predict route: {e}")
        logger.error(traceback.format_exc())
        return f"An error occurred: {str(e)}", 500


@app.route('/api/treatment/<disease_name>')
def treatment_api(disease_name):
    try:
        logger.info(f"Treatment API called for disease: {disease_name}")
        disease_info = get_disease_info(disease_name)
        if disease_info:
            return jsonify(disease_info)
        else:
            logger.warning(f"No disease info found for: {disease_name}")
            return jsonify({'error': 'Disease information not found'}), 404
    except Exception as e:
        logger.error(f"Treatment API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/resources/<disease_name>')
def resources_api(disease_name):
    try:
        logger.info(f"Resources API called for disease: {disease_name}")
        disease_info = get_disease_info(disease_name)
        if disease_info and 'additional_resources' in disease_info:
            return jsonify(disease_info['additional_resources'])
        else:
            return jsonify({'error': 'Additional resources not found'}), 404
    except Exception as e:
        logger.error(f"Resources API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/direct-ai', methods=['POST'])
def direct_ai_chat():
    """Direct Gemini AI chat without AgriPal formatting"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        history = data.get('history', [])
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
            
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-001')
        
        conversation_context = ""
        if history:
            recent_messages = history[-3:]
            for msg in recent_messages:
                role = "Human" if msg['role'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['text']}\n"
        
        full_prompt = f"""You are Gemini AI, a helpful and knowledgeable AI assistant. Respond naturally and comprehensively to the user's question.
        {conversation_context}
        Human: {message}
        Assistant:"""
        
        response = gemini_model.generate_content(full_prompt)
        
        if response and response.text:
            return jsonify({
                'success': True,
                'response': response.text.strip(),
                'timestamp': datetime.now().isoformat(),
                'ai_status': 'online',
                'mode': 'direct_ai'
            })
        else:
            return jsonify({'success': False, 'error': 'No response from AI'}), 500
            
    except Exception as e:
        logger.error(f"Direct AI chat error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calculate-dosage', methods=['POST'])
def calculate_dosage_api():
    try:
        data = request.json
        disease_name = data.get('disease_name')
        area = data.get('area')
        area_unit = data.get('area_unit', 'hectare')
        
        disease_info = get_disease_info(disease_name)
        if disease_info and 'pesticide' in disease_info:
            chemical_dosage, organic_dosage, hectare_conversion = calculate_dosage(
                area, area_unit, disease_info['pesticide']
            )
            return jsonify({
                'chemical_dosage': chemical_dosage,
                'organic_dosage': organic_dosage,
                'hectare_conversion': hectare_conversion,
                'area': area,
                'area_unit': area_unit
            })
        else:
            return jsonify({'error': 'Disease or pesticide information not found'}), 404
    except Exception as e:
        logger.error(f"Dosage calculation API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ROUTE: Nutrition Testing Page
# ============================================================================
@app.route('/nutrition-testing')
@login_required
def nutrition_testing():
    """Nutrition deficiency testing tool page"""
    return render_template('nutrition_testing.html', 
                         user=current_user,
                         location=current_user.location if hasattr(current_user, 'location') else '',
                         land_area=current_user.land_area if hasattr(current_user, 'land_area') else 0,
                         area_unit=current_user.area_unit if hasattr(current_user, 'area_unit') else 'square_meter')


# ============================================================================
# ROUTE: Analyze Nutrition Deficiency
# ============================================================================
@app.route('/analyze-nutrition', methods=['POST'])
@login_required
def analyze_nutrition():
    """Analyze uploaded image for nutrition deficiency"""
    
    logger.info("=" * 80)
    logger.info("🔬 NUTRITION DEFICIENCY ANALYSIS ENDPOINT")
    logger.info("=" * 80)
    
    if 'image' not in request.files:
        flash("Error: No image file uploaded.", "error")
        return render_template("error.html", back_link="/nutrition-testing")
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        flash("Error: No image selected.", "error")
        return render_template("error.html", back_link="/nutrition-testing")
    
    if not allowed_file(image_file.filename):
        flash("Error: Invalid file type. Please upload PNG, JPG, or JPEG.", "error")
        return render_template("error.html", back_link="/nutrition-testing")
    
    try:
        location = request.form.get("location", "").strip()
        area = request.form.get("area", "0")
        area_unit = request.form.get("area_unit", "square_meter")
        area_float = float(area) if area else 0.0
        
        try:
            if location:
                current_user.location = location
            if area_float > 0:
                current_user.land_area = area_float
                current_user.area_unit = area_unit
            db.session.commit()
            logger.info("✅ User profile data saved")
        except Exception as e:
            logger.warning(f"⚠️ Could not save user data: {e}")
            db.session.rollback()
        
        image_filename = str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)
        logger.info(f"✅ Image saved to: {image_path}")
        
        analysis_result = analyze_nutrition_deficiency(image_path)
        
        if not analysis_result['success']:
            flash(f"Error during analysis: {analysis_result.get('error', 'Unknown error')}", "error")
            return render_template("error.html", back_link="/nutrition-testing")
        
        diagnoses = analysis_result['diagnoses']
        
        if len(diagnoses) == 0:
            return render_template('nutrition_result.html',
                                 healthy=True,
                                 image_url=url_for('static', filename=f'uploads/{image_filename}'),
                                 location=location,
                                 area=area,
                                 area_unit=area_unit,
                                 color_analysis=analysis_result['color_analysis'])
        
        primary_deficiency = diagnoses[0]
        
        if area_float > 0:
            chemical_dosage, organic_dosage, hectare_conversion = calculate_fertilizer_dosage(
                area_float, 
                area_unit, 
                primary_deficiency['fertilizer']
            )
        else:
            chemical_dosage = None
            organic_dosage = None
            hectare_conversion = 0
        
        return render_template('nutrition_result.html',
                             healthy=False,
                             diagnoses=diagnoses,
                             primary_deficiency=primary_deficiency,
                             image_url=url_for('static', filename=f'uploads/{image_filename}'),
                             location=location,
                             area=area,
                             area_unit=area_unit,
                             chemical_dosage=chemical_dosage,
                             organic_dosage=organic_dosage,
                             hectare_conversion=hectare_conversion,
                             color_analysis=analysis_result['color_analysis'])
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ ERROR IN NUTRITION ANALYSIS")
        logger.error("=" * 80)
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        flash("Unexpected error occurred during nutrition analysis.", "error")
        return render_template("error.html", back_link="/nutrition-testing"), 500


# ============================================================================
# ROUTE: API Endpoint for Nutrition Info
# ============================================================================
@app.route('/api/nutrition/<deficiency_key>')
def nutrition_api(deficiency_key):
    """API endpoint to get nutrition deficiency information"""
    try:
        logger.info(f"Nutrition API called for: {deficiency_key}")
        
        if deficiency_key in nutrition_deficiency_data:
            return jsonify(nutrition_deficiency_data[deficiency_key])
        else:
            logger.warning(f"No nutrition info found for: {deficiency_key}")
            return jsonify({'error': 'Nutrition deficiency information not found'}), 404
            
    except Exception as e:
        logger.error(f"Nutrition API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ROUTE: Calculate Fertilizer Dosage API
# ============================================================================
@app.route('/api/calculate-fertilizer', methods=['POST'])
def calculate_fertilizer_api():
    """API endpoint to calculate fertilizer dosage"""
    try:
        data = request.json
        deficiency_key = data.get('deficiency_key')
        area = data.get('area')
        area_unit = data.get('area_unit', 'hectare')
        
        if deficiency_key in nutrition_deficiency_data:
            deficiency_info = nutrition_deficiency_data[deficiency_key]
            
            chemical_dosage, organic_dosage, hectare_conversion = calculate_fertilizer_dosage(
                float(area), area_unit, deficiency_info['fertilizer']
            )
            
            return jsonify({
                'success': True,
                'chemical_dosage': chemical_dosage,
                'organic_dosage': organic_dosage,
                'hectare_conversion': hectare_conversion,
                'area': area,
                'area_unit': area_unit
            })
        else:
            return jsonify({'success': False, 'error': 'Deficiency information not found'}), 404
            
    except Exception as e:
        logger.error(f"Fertilizer calculation API error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Initialize enhanced system
gemini_status = startup_gemini_check()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\n" + "="*80)
    logger.info("🛑 SHUTDOWN SIGNAL RECEIVED (Ctrl+C)")
    logger.info("="*80)
    logger.info("Cleaning up resources...")

    try:
        temp_files = ['detected_disease.json']
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"✅ Cleaned up: {temp_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    logger.info("👋 AgriPal Server Stopped Successfully!")
    logger.info("="*80)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':
    
    port = int(os.environ.get('PORT', 5000))
    
    clear_sessions_on_startup()
    init_database()
    
    local_ip = get_local_ip()
    
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    IS_PRODUCTION = (FLASK_ENV == 'production')

    logger.info("="*80)
    logger.info("🌱 STARTING AGRIPAL APPLICATION 🌱")
    logger.info("="*80)
    logger.info(f"Environment: {FLASK_ENV}")
    logger.info(f"Port: {port}")
    logger.info(f"Model status: ⚠️ Offline (maintenance mode)")
    logger.info(f"Disease treatments loaded: {len(disease_treatments)}")
    logger.info(f"Gemini AI status: {'✅ Online' if gemini_status else '⚠️ Offline'}")
    logger.info("="*80)
    
    if not IS_PRODUCTION:
        logger.info("📱 ACCESS URLs:")
        logger.info(f"   Local:    http://127.0.0.1:{port}")
        logger.info(f"   Network:  http://{local_ip}:{port}")
        logger.info("="*80)
        logger.info("📱 MOBILE ACCESS:")
        logger.info(f"   1. Connect your phone to the SAME WiFi network")
        logger.info(f"   2. Open browser on phone")
        logger.info(f"   3. Go to: http://{local_ip}:{port}")
        logger.info("="*80)
        logger.info("🛑 SHUTDOWN: Press Ctrl+C to stop the server")
        logger.info("="*80)

    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=not IS_PRODUCTION,
            use_reloader=not IS_PRODUCTION
        )
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)
