"""
Article Screening Web Application
A Streamlit-based UI for screening academic articles for bibliometric research.
Supports English and Turkish languages.
"""

import io
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Article Screening Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Translations
TRANSLATIONS = {
    "en": {
        "app_title": "📚 Article Screening Tool",
        "app_subtitle": "Academic article evaluation tool for bibliometric studies",
        "settings": "⚙️ Settings",
        "language": "Language",
        "analysis_mode": "Analysis Mode",
        "mode_rules": "🔍 Rule-Based",
        "mode_gemini": "🤖 Gemini AI",
        "mode_deepseek": "🧠 DeepSeek AI",
        "mode_help": "Rule-based: Keyword matching\nGemini/DeepSeek: AI-powered analysis",
        "api_key": "API Key",
        "api_key_required": "⚠️ API key required",
        "column_mapping": "📋 Column Mapping",
        "column_mapping_desc": "Select column names from your Excel file",
        "title_column": "Title Column",
        "abstract_column": "Abstract Column",
        "tab_upload": "📁 Upload File",
        "tab_criteria": "⚙️ Criteria",
        "tab_analysis": "📊 Analysis",
        "upload_title": "Upload Excel File",
        "upload_desc": "Drag and drop your Excel file or click to select",
        "upload_help": "Excel file exported from WoS or Scopus",
        "articles_loaded": "articles loaded!",
        "total_articles": "Total Articles",
        "columns": "Columns",
        "evaluated": "Evaluated",
        "data_preview": "📋 Data Preview",
        "file_error": "❌ Error loading file:",
        "criteria_title": "Evaluation Criteria",
        "criteria_desc": "Enter keywords separated by commas (for rule-based mode)",
        "ai_tech": "🤖 AI Technology",
        "ai_keywords": "AI keywords",
        "ai_help": "Article must contain at least one of these",
        "web_env": "🌐 Web/Online Environment",
        "web_keywords": "Web environment keywords",
        "human_factor": "👥 Human Factor",
        "human_keywords": "Human factor keywords",
        "criteria_summary": "📝 Summary",
        "criteria_info": """**An article is accepted if:**
1. ✅ Mentions AI technology
2. ✅ Mentions web/online environment
3. ✅ Involves human users

All criteria must be met!""",
        "analysis_title": "Analysis and Results",
        "upload_first": "⚠️ Please upload an Excel file first",
        "reprocess_all": "Reprocess all articles",
        "reprocess_help": "If checked, previously evaluated articles will be reprocessed",
        "limit": "Limit (0 = all)",
        "limit_help": "Process limited number of articles for testing",
        "selected_mode": "📌 Selected mode:",
        "start_analysis": "🚀 Start Analysis",
        "api_key_error": "❌ API key required! Please enter in sidebar.",
        "gemini_connected": "✅ Gemini AI connected",
        "deepseek_connected": "✅ DeepSeek AI connected",
        "connection_error": "❌ Connection error:",
        "all_evaluated": "✅ All articles already evaluated!",
        "processing_log": "📋 Processing Log",
        "processing": "Processing:",
        "analysis_complete": "🎉 Analysis complete!",
        "accepted": "accepted",
        "rejected": "rejected",
        "total_processed": "Total Processed",
        "accept": "✅ Accept",
        "reject": "❌ Reject",
        "acceptance_rate": "Acceptance Rate",
        "results_preview": "📋 Results Preview",
        "filter": "Filter",
        "filter_all": "All",
        "filter_accepted": "✅ Accepted",
        "filter_rejected": "❌ Rejected",
        "download_title": "📥 Download Results",
        "download_button": "📥 Download as Excel",
        "ai_not_found": "AI technology not found",
        "web_not_found": "Web/online environment not found",
        "human_not_found": "Human factor not found",
        "criteria_not_met": "Criteria not met",
        "api_error": "API Error:",
        "rate_limit": "⏳ Rate limit - waiting",
        "seconds": "s",
        "attempt": "Attempt",
        "quota_exceeded": "API quota exceeded - use DeepSeek or wait 24 hours",
        "max_retries": "Maximum retry count reached",
    },
    "tr": {
        "app_title": "📚 Makale Tarama Uygulaması",
        "app_subtitle": "Bibliyometrik çalışmalar için akademik makale değerlendirme aracı",
        "settings": "⚙️ Ayarlar",
        "language": "Dil",
        "analysis_mode": "Analiz Modu",
        "mode_rules": "🔍 Kural Tabanlı",
        "mode_gemini": "🤖 Gemini AI",
        "mode_deepseek": "🧠 DeepSeek AI",
        "mode_help": "Kural tabanlı: Anahtar kelime eşleştirmesi\nGemini/DeepSeek: Yapay zeka ile akıllı analiz",
        "api_key": "API Anahtarı",
        "api_key_required": "⚠️ API anahtarı gerekli",
        "column_mapping": "📋 Sütun Eşleştirme",
        "column_mapping_desc": "Excel dosyanızdaki sütun isimlerini seçin",
        "title_column": "Başlık Sütunu",
        "abstract_column": "Özet Sütunu",
        "tab_upload": "📁 Dosya Yükle",
        "tab_criteria": "⚙️ Kriterler",
        "tab_analysis": "📊 Analiz",
        "upload_title": "Excel Dosyası Yükle",
        "upload_desc": "Excel dosyanızı sürükleyip bırakın veya seçin",
        "upload_help": "WoS veya Scopus'tan dışa aktarılmış Excel dosyası",
        "articles_loaded": "makale yüklendi!",
        "total_articles": "Toplam Makale",
        "columns": "Sütun Sayısı",
        "evaluated": "Değerlendirilmiş",
        "data_preview": "📋 Veri Önizleme",
        "file_error": "❌ Dosya yüklenirken hata:",
        "criteria_title": "Değerlendirme Kriterleri",
        "criteria_desc": "Her bir kriter için anahtar kelimeleri virgülle ayırarak girin (Kural tabanlı mod için)",
        "ai_tech": "🤖 AI Teknolojisi",
        "ai_keywords": "AI anahtar kelimeleri",
        "ai_help": "Makale bunlardan en az birini içermeli",
        "web_env": "🌐 Web/Online Ortam",
        "web_keywords": "Web ortam anahtar kelimeleri",
        "human_factor": "👥 İnsan Faktörü",
        "human_keywords": "İnsan faktörü anahtar kelimeleri",
        "criteria_summary": "📝 Özet",
        "criteria_info": """**Bir makale kabul edilir eğer:**
1. ✅ AI teknolojisinden bahsediyorsa
2. ✅ Web/online ortamdan bahsediyorsa
3. ✅ İnsan kullanıcıları içeriyorsa

Tüm kriterler sağlanmalıdır!""",
        "analysis_title": "Analiz ve Sonuçlar",
        "upload_first": "⚠️ Lütfen önce bir Excel dosyası yükleyin",
        "reprocess_all": "Tüm makaleleri yeniden değerlendir",
        "reprocess_help": "İşaretlenirse daha önce değerlendirilmiş makaleler de yeniden işlenir",
        "limit": "Limit (0 = tümü)",
        "limit_help": "Test için sınırlı sayıda makale işleyin",
        "selected_mode": "📌 Seçili mod:",
        "start_analysis": "🚀 Analizi Başlat",
        "api_key_error": "❌ API anahtarı gerekli! Lütfen sidebar'dan girin.",
        "gemini_connected": "✅ Gemini AI bağlantısı kuruldu",
        "deepseek_connected": "✅ DeepSeek AI bağlantısı kuruldu",
        "connection_error": "❌ Bağlantı hatası:",
        "all_evaluated": "✅ Tüm makaleler zaten değerlendirilmiş!",
        "processing_log": "📋 İşlem Logu",
        "processing": "İşleniyor:",
        "analysis_complete": "🎉 Analiz tamamlandı!",
        "accepted": "kabul",
        "rejected": "red",
        "total_processed": "Toplam İşlenen",
        "accept": "✅ Kabul",
        "reject": "❌ Red",
        "acceptance_rate": "Kabul Oranı",
        "results_preview": "📋 Sonuç Önizleme",
        "filter": "Filtre",
        "filter_all": "Tümü",
        "filter_accepted": "✅ Kabul Edilenler",
        "filter_rejected": "❌ Red Edilenler",
        "download_title": "📥 Sonuçları İndir",
        "download_button": "📥 Excel Olarak İndir",
        "ai_not_found": "AI teknolojisi bulunamadı",
        "web_not_found": "Web/online ortam bulunamadı",
        "human_not_found": "İnsan faktörü bulunamadı",
        "criteria_not_met": "Kriterler sağlanmadı",
        "api_error": "API Hatası:",
        "rate_limit": "⏳ Rate limit - bekleniyor",
        "seconds": "s",
        "attempt": "Deneme",
        "quota_exceeded": "API kota limiti aşıldı - DeepSeek kullanın veya 24 saat bekleyin",
        "max_retries": "Maksimum deneme sayısına ulaşıldı",
    }
}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Default keywords
DEFAULT_AI_KEYWORDS = """chatgpt, gpt, gpt-4, gpt-3, llm, large language model, conversational ai, 
ai agent, chatbot, chat bot, artificial intelligence, generative ai, genai, openai, 
gemini, claude, copilot, ai assistant, ai tutor, nlp, natural language processing, 
machine learning, deep learning"""

DEFAULT_WEB_KEYWORDS = """online, distance learning, distance education, remote learning, 
remote education, e-learning, elearning, mooc, massive open online, blended learning, 
blended education, hybrid learning, hybrid education, web-based, web based, 
virtual learning, virtual education, virtual classroom, learning management system, 
lms, digital learning, digital education, online course, online education, online learning"""

DEFAULT_HUMAN_KEYWORDS = """teacher, teachers, student, students, instructor, instructors, 
learner, learners, educator, educators, administrator, administrators, faculty, 
tutor, tutors, pupil, pupils, professor, professors, academic, academics, 
undergraduate, graduate, postgraduate, university student, college student, 
high school student, k-12, higher education"""


def get_text(key: str) -> str:
    """Get translated text for current language."""
    lang = st.session_state.get('language', 'en')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)


def parse_keywords(text: str) -> list:
    """Parse comma-separated keywords into regex patterns."""
    keywords = [k.strip().lower() for k in text.split(',') if k.strip()]
    patterns = [rf'\b{re.escape(k)}\b' for k in keywords]
    return patterns


def check_keywords(text: str, patterns: list) -> tuple[bool, list]:
    """Check if text contains any of the keyword patterns."""
    if not text or pd.isna(text):
        return False, []
    
    text_lower = text.lower()
    found = []
    for pattern in patterns:
        if re.search(pattern, text_lower):
            found.append(pattern.replace(r'\b', '').replace('\\', ''))
    
    return len(found) > 0, found


def screen_article_rules(title: str, abstract: str, ai_patterns: list, 
                         web_patterns: list, human_patterns: list) -> tuple[bool, str]:
    """Screen an article using rule-based keyword matching."""
    combined_text = f"{title or ''} {abstract or ''}"
    
    has_ai, _ = check_keywords(combined_text, ai_patterns)
    has_web, _ = check_keywords(combined_text, web_patterns)
    has_human, _ = check_keywords(combined_text, human_patterns)
    
    missing = []
    if not has_ai:
        missing.append(get_text("ai_not_found"))
    if not has_web:
        missing.append(get_text("web_not_found"))
    if not has_human:
        missing.append(get_text("human_not_found"))
    
    if not missing:
        return True, ""
    else:
        return False, "; ".join(missing)


def screen_article_gemini(title: str, abstract: str, client, lang: str, max_retries: int = 3) -> tuple[bool, str]:
    """Screen an article using Gemini AI with retry logic."""
    if lang == "tr":
        prompt = f"""Sen bir bibliyometrik araştırma asistanısın. Aşağıdaki akademik makaleyi değerlendir.

BAŞLIK: {title}

ÖZET: {abstract}

KRİTERLER (Tümü sağlanmalı):
1. AI Teknolojisi: Makale ChatGPT, GPT, LLM, yapay zeka, chatbot gibi AI teknolojilerini içermeli
2. Web/Online Ortam: Makale online eğitim, uzaktan eğitim, e-learning, MOOC gibi web tabanlı ortamı içermeli
3. İnsan Faktörü: Makale öğretmen, öğrenci, eğitimci gibi insan kullanıcıları içermeli

KARAR:
- Eğer 3 kriter de sağlanıyorsa SADECE şunu yaz: KABUL
- Eğer herhangi biri eksikse şunu yaz: RED: [hangi kriter eksik açıkla]

Tek satır yanıt ver:"""
        accept_keyword = "KABUL"
        reject_keyword = "RED"
    else:
        prompt = f"""You are a bibliometric research assistant. Evaluate the following academic article.

TITLE: {title}

ABSTRACT: {abstract}

CRITERIA (All must be met):
1. AI Technology: Article must mention ChatGPT, GPT, LLM, AI, chatbot, etc.
2. Web/Online Environment: Article must mention online education, e-learning, MOOC, etc.
3. Human Factor: Article must involve teachers, students, educators, etc.

DECISION:
- If all 3 criteria are met, write ONLY: ACCEPT
- If any criterion is missing, write: REJECT: [explain which criterion is missing]

Reply in one line:"""
        accept_keyword = "ACCEPT"
        reject_keyword = "REJECT"
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            result = response.text.strip()
            result_upper = result.upper()
            
            if accept_keyword in result_upper and reject_keyword not in result_upper:
                return True, ""
            else:
                reason = result
                if f"{reject_keyword}:" in result.upper():
                    idx = result.upper().find(f"{reject_keyword}:")
                    reason = result[idx+len(reject_keyword)+1:].strip()
                elif reject_keyword in result.upper():
                    idx = result.upper().find(reject_keyword)
                    reason = result[idx+len(reject_keyword):].strip()
                return False, reason if reason else get_text("criteria_not_met")
                
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = (attempt + 1) * 30
                if attempt < max_retries - 1:
                    st.warning(f"{get_text('rate_limit')} {wait_time}{get_text('seconds')}... ({get_text('attempt')} {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, get_text("quota_exceeded")
            return False, f"{get_text('api_error')} {error_str[:100]}"
    
    return False, get_text("max_retries")


def screen_article_deepseek(title: str, abstract: str, client, lang: str) -> tuple[bool, str]:
    """Screen an article using DeepSeek AI."""
    if lang == "tr":
        prompt = f"""Sen bir bibliyometrik araştırma asistanısın. Aşağıdaki akademik makaleyi değerlendir.

BAŞLIK: {title}

ÖZET: {abstract}

KRİTERLER (Tümü sağlanmalı):
1. AI Teknolojisi: Makale ChatGPT, GPT, LLM, yapay zeka, chatbot gibi AI teknolojilerini içermeli
2. Web/Online Ortam: Makale online eğitim, uzaktan eğitim, e-learning, MOOC gibi web tabanlı ortamı içermeli
3. İnsan Faktörü: Makale öğretmen, öğrenci, eğitimci gibi insan kullanıcıları içermeli

KARAR:
- Eğer 3 kriter de sağlanıyorsa SADECE şunu yaz: KABUL
- Eğer herhangi biri eksikse şunu yaz: RED: [hangi kriter eksik açıkla]

Tek satır yanıt ver:"""
        accept_keyword = "KABUL"
        reject_keyword = "RED"
        system_msg = "Sen bir akademik makale değerlendirme asistanısın. Kısa ve net yanıtlar ver."
    else:
        prompt = f"""You are a bibliometric research assistant. Evaluate the following academic article.

TITLE: {title}

ABSTRACT: {abstract}

CRITERIA (All must be met):
1. AI Technology: Article must mention ChatGPT, GPT, LLM, AI, chatbot, etc.
2. Web/Online Environment: Article must mention online education, e-learning, MOOC, etc.
3. Human Factor: Article must involve teachers, students, educators, etc.

DECISION:
- If all 3 criteria are met, write ONLY: ACCEPT
- If any criterion is missing, write: REJECT: [explain which criterion is missing]

Reply in one line:"""
        accept_keyword = "ACCEPT"
        reject_keyword = "REJECT"
        system_msg = "You are an academic article evaluation assistant. Provide short and clear responses."
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        result = response.choices[0].message.content.strip()
        result_upper = result.upper()
        
        if accept_keyword in result_upper and reject_keyword not in result_upper:
            return True, ""
        else:
            reason = result
            if f"{reject_keyword}:" in result.upper():
                idx = result.upper().find(f"{reject_keyword}:")
                reason = result[idx+len(reject_keyword)+1:].strip()
            elif reject_keyword in result.upper():
                idx = result.upper().find(reject_keyword)
                reason = result[idx+len(reject_keyword):].strip()
            return False, reason if reason else get_text("criteria_not_met")
            
    except Exception as e:
        return False, f"{get_text('api_error')} {str(e)}"


def main():
    # Language selector in sidebar (first item)
    with st.sidebar:
        lang = st.selectbox(
            "🌐 Language / Dil",
            ["English", "Türkçe"],
            index=0
        )
        st.session_state['language'] = 'en' if lang == "English" else 'tr'
        
        st.divider()
    
    # Header
    st.markdown(f'<p class="main-header">{get_text("app_title")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{get_text("app_subtitle")}</p>', unsafe_allow_html=True)
    
    # Sidebar settings
    with st.sidebar:
        st.header(get_text("settings"))
        
        # Analysis mode
        mode_options = [get_text("mode_rules"), get_text("mode_gemini"), get_text("mode_deepseek")]
        analysis_mode = st.radio(
            get_text("analysis_mode"),
            mode_options,
            help=get_text("mode_help")
        )
        st.session_state['analysis_mode'] = analysis_mode
        
        # API Key input based on mode
        if get_text("mode_gemini") in analysis_mode:
            api_key = st.text_input(
                f"Gemini {get_text('api_key')}",
                type="password",
                help="https://aistudio.google.com/apikey"
            )
            st.session_state['api_key'] = api_key
            if not api_key:
                st.warning(get_text("api_key_required"))
        
        elif get_text("mode_deepseek") in analysis_mode:
            api_key = st.text_input(
                f"DeepSeek {get_text('api_key')}",
                type="password",
                help="https://platform.deepseek.com"
            )
            st.session_state['api_key'] = api_key
            if not api_key:
                st.warning(get_text("api_key_required"))
        
        st.divider()
        
        # Column mapping
        st.subheader(get_text("column_mapping"))
        st.caption(get_text("column_mapping_desc"))
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([get_text("tab_upload"), get_text("tab_criteria"), get_text("tab_analysis")])
    
    # Tab 1: File Upload
    with tab1:
        st.subheader(get_text("upload_title"))
        
        uploaded_file = st.file_uploader(
            get_text("upload_desc"),
            type=['xlsx', 'xls'],
            help=get_text("upload_help")
        )
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                st.session_state['original_df'] = df.copy()
                st.session_state['df'] = df
                
                # Column mapping in sidebar
                with st.sidebar:
                    columns = df.columns.tolist()
                    
                    title_col = st.selectbox(
                        get_text("title_column"),
                        columns,
                        index=columns.index('Title') if 'Title' in columns else 0
                    )
                    
                    abstract_col = st.selectbox(
                        get_text("abstract_column"),
                        columns,
                        index=columns.index('Abstract Note') if 'Abstract Note' in columns else (
                            columns.index('Abstract') if 'Abstract' in columns else 1
                        )
                    )
                    
                    st.session_state['title_col'] = title_col
                    st.session_state['abstract_col'] = abstract_col
                
                # Preview
                st.success(f"✅ {len(df)} {get_text('articles_loaded')}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(get_text("total_articles"), len(df))
                with col2:
                    st.metric(get_text("columns"), len(df.columns))
                with col3:
                    if 'Acceptance' in df.columns or 'Acceptance ' in df.columns:
                        acc_col = 'Acceptance ' if 'Acceptance ' in df.columns else 'Acceptance'
                        evaluated = df[acc_col].notna().sum()
                        st.metric(get_text("evaluated"), evaluated)
                
                st.subheader(get_text("data_preview"))
                st.dataframe(df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"{get_text('file_error')} {e}")
    
    # Tab 2: Criteria Configuration
    with tab2:
        st.subheader(get_text("criteria_title"))
        st.caption(get_text("criteria_desc"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {get_text('ai_tech')}")
            ai_keywords = st.text_area(
                get_text("ai_keywords"),
                value=DEFAULT_AI_KEYWORDS,
                height=150,
                help=get_text("ai_help")
            )
            
            st.markdown(f"### {get_text('web_env')}")
            web_keywords = st.text_area(
                get_text("web_keywords"),
                value=DEFAULT_WEB_KEYWORDS,
                height=150,
                help=get_text("ai_help")
            )
        
        with col2:
            st.markdown(f"### {get_text('human_factor')}")
            human_keywords = st.text_area(
                get_text("human_keywords"),
                value=DEFAULT_HUMAN_KEYWORDS,
                height=150,
                help=get_text("ai_help")
            )
            
            st.markdown(f"### {get_text('criteria_summary')}")
            st.info(get_text("criteria_info"))
        
        # Store in session
        st.session_state['ai_keywords'] = ai_keywords
        st.session_state['web_keywords'] = web_keywords
        st.session_state['human_keywords'] = human_keywords
    
    # Tab 3: Analysis
    with tab3:
        st.subheader(get_text("analysis_title"))
        
        if 'df' not in st.session_state:
            st.warning(get_text("upload_first"))
            return
        
        df = st.session_state['df']
        title_col = st.session_state.get('title_col', 'Title')
        abstract_col = st.session_state.get('abstract_col', 'Abstract Note')
        
        # Analysis options
        col1, col2 = st.columns([2, 1])
        with col1:
            force_reprocess = st.checkbox(
                get_text("reprocess_all"),
                help=get_text("reprocess_help")
            )
        with col2:
            limit = st.number_input(
                get_text("limit"),
                min_value=0,
                max_value=len(df),
                value=0,
                help=get_text("limit_help")
            )
        
        # Get current mode
        current_mode = st.session_state.get('analysis_mode', get_text("mode_rules"))
        current_api_key = st.session_state.get('api_key', None)
        current_lang = st.session_state.get('language', 'en')
        
        # Show current mode
        st.info(f"{get_text('selected_mode')} **{current_mode}**")
        
        # Start analysis button
        if st.button(get_text("start_analysis"), type="primary", use_container_width=True):
            
            # Validate API key for AI modes
            if get_text("mode_gemini") in current_mode or get_text("mode_deepseek") in current_mode:
                if not current_api_key:
                    st.error(get_text("api_key_error"))
                    return
            
            # Parse keywords for rule-based
            ai_patterns = parse_keywords(st.session_state.get('ai_keywords', DEFAULT_AI_KEYWORDS))
            web_patterns = parse_keywords(st.session_state.get('web_keywords', DEFAULT_WEB_KEYWORDS))
            human_patterns = parse_keywords(st.session_state.get('human_keywords', DEFAULT_HUMAN_KEYWORDS))
            
            # Setup AI client if needed
            gemini_client = None
            deepseek_client = None
            
            if get_text("mode_gemini") in current_mode:
                try:
                    from google import genai
                    gemini_client = genai.Client(api_key=current_api_key)
                    st.success(get_text("gemini_connected"))
                except Exception as e:
                    st.error(f"{get_text('connection_error')} {e}")
                    return
            
            elif get_text("mode_deepseek") in current_mode:
                try:
                    import httpx
                    from openai import OpenAI
                    http_client = httpx.Client(timeout=60.0)
                    deepseek_client = OpenAI(
                        api_key=current_api_key,
                        base_url="https://api.deepseek.com",
                        http_client=http_client
                    )
                    st.success(get_text("deepseek_connected"))
                except Exception as e:
                    st.error(f"{get_text('connection_error')} {e}")
                    return
            
            # Determine columns
            acc_col = 'Acceptance ' if 'Acceptance ' in df.columns else 'Acceptance'
            reason_col = 'Reason'
            
            # Ensure columns exist
            if acc_col not in df.columns:
                df[acc_col] = None
            if reason_col not in df.columns:
                df[reason_col] = None
            
            # Determine rows to process
            if force_reprocess:
                rows_to_process = df.index.tolist()
            else:
                mask = df[acc_col].isna() | (df[acc_col] == '')
                rows_to_process = df[mask].index.tolist()
            
            if limit > 0:
                rows_to_process = rows_to_process[:limit]
            
            if len(rows_to_process) == 0:
                st.info(get_text("all_evaluated"))
                return
            
            # Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.expander(get_text("processing_log"), expanded=True)
            
            accepted = 0
            rejected = 0
            
            with log_container:
                for i, idx in enumerate(rows_to_process):
                    title = str(df.at[idx, title_col]) if pd.notna(df.at[idx, title_col]) else ""
                    abstract = str(df.at[idx, abstract_col]) if pd.notna(df.at[idx, abstract_col]) else ""
                    
                    # Get short title for display
                    short_title = title[:50] + "..." if len(title) > 50 else title
                    
                    # Screen based on mode
                    if get_text("mode_gemini") in current_mode and gemini_client:
                        acceptance, reason = screen_article_gemini(title, abstract, gemini_client, current_lang)
                        time.sleep(0.5)
                    elif get_text("mode_deepseek") in current_mode and deepseek_client:
                        acceptance, reason = screen_article_deepseek(title, abstract, deepseek_client, current_lang)
                        time.sleep(0.3)
                    else:
                        acceptance, reason = screen_article_rules(title, abstract, ai_patterns, web_patterns, human_patterns)
                    
                    # Store results
                    df.at[idx, acc_col] = acceptance
                    df.at[idx, reason_col] = reason
                    
                    if acceptance:
                        accepted += 1
                        st.write(f"✅ #{idx+1}: {short_title}")
                    else:
                        rejected += 1
                        st.write(f"❌ #{idx+1}: {short_title} → {reason[:80]}")
                    
                    # Update progress
                    progress = (i + 1) / len(rows_to_process)
                    progress_bar.progress(progress)
                    status_text.text(f"{get_text('processing')} {i + 1}/{len(rows_to_process)} | ✅ {accepted} | ❌ {rejected}")
            
            st.session_state['df'] = df
            st.session_state['analysis_complete'] = True
            st.session_state['accepted'] = accepted
            st.session_state['rejected'] = rejected
            
            st.success(f"{get_text('analysis_complete')} ✅ {accepted} {get_text('accepted')} | ❌ {rejected} {get_text('rejected')}")
        
        # Show results if analysis is complete
        if st.session_state.get('analysis_complete', False):
            st.divider()
            
            accepted = st.session_state.get('accepted', 0)
            rejected = st.session_state.get('rejected', 0)
            total = accepted + rejected
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(get_text("total_processed"), total)
            with col2:
                st.metric(get_text("accept"), accepted)
            with col3:
                st.metric(get_text("reject"), rejected)
            with col4:
                rate = (accepted / total * 100) if total > 0 else 0
                st.metric(get_text("acceptance_rate"), f"{rate:.1f}%")
            
            # Results preview
            df = st.session_state['df']
            acc_col = 'Acceptance ' if 'Acceptance ' in df.columns else 'Acceptance'
            
            st.subheader(get_text("results_preview"))
            
            filter_option = st.radio(
                get_text("filter"),
                [get_text("filter_all"), get_text("filter_accepted"), get_text("filter_rejected")],
                horizontal=True
            )
            
            if filter_option == get_text("filter_accepted"):
                display_df = df[df[acc_col] == True]
            elif filter_option == get_text("filter_rejected"):
                display_df = df[df[acc_col] == False]
            else:
                display_df = df
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export
            st.divider()
            st.subheader(get_text("download_title"))
            
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
            output.seek(0)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label=get_text("download_button"),
                data=output,
                file_name=f"screening_results_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
