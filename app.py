"""
Article Screening Web Application
A Streamlit-based UI for screening academic articles for bibliometric research.
"""

import io
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Makale Tarama Uygulaması",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        missing.append("AI teknolojisi bulunamadı")
    if not has_web:
        missing.append("Web/online ortam bulunamadı")
    if not has_human:
        missing.append("İnsan faktörü bulunamadı")
    
    if not missing:
        return True, ""
    else:
        return False, "; ".join(missing)


def screen_article_gemini(title: str, abstract: str, client, max_retries: int = 3) -> tuple[bool, str]:
    """Screen an article using Gemini AI with retry logic."""
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
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            result = response.text.strip()
            
            # Parse response
            result_upper = result.upper()
            if "KABUL" in result_upper and "RED" not in result_upper:
                return True, ""
            else:
                # Extract reason
                reason = result
                if "RED:" in result.upper():
                    idx = result.upper().find("RED:")
                    reason = result[idx+4:].strip()
                elif "RED" in result.upper():
                    idx = result.upper().find("RED")
                    reason = result[idx+3:].strip()
                return False, reason if reason else "Kriterler sağlanmadı"
                
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = (attempt + 1) * 30  # 30, 60, 90 seconds
                if attempt < max_retries - 1:
                    st.warning(f"⏳ Rate limit - {wait_time}s bekleniyor... (Deneme {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, "API kota limiti aşıldı - DeepSeek kullanın veya 24 saat bekleyin"
            return False, f"Gemini API Hatası: {error_str[:100]}"
    
    return False, "Maksimum deneme sayısına ulaşıldı"


def screen_article_deepseek(title: str, abstract: str, client) -> tuple[bool, str]:
    """Screen an article using DeepSeek AI."""
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
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Sen bir akademik makale değerlendirme asistanısın. Kısa ve net yanıtlar ver."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        result = response.choices[0].message.content.strip()
        
        # Parse response
        result_upper = result.upper()
        if "KABUL" in result_upper and "RED" not in result_upper:
            return True, ""
        else:
            # Extract reason
            reason = result
            if "RED:" in result.upper():
                idx = result.upper().find("RED:")
                reason = result[idx+4:].strip()
            elif "RED" in result.upper():
                idx = result.upper().find("RED")
                reason = result[idx+3:].strip()
            return False, reason if reason else "Kriterler sağlanmadı"
            
    except Exception as e:
        return False, f"DeepSeek API Hatası: {str(e)}"


def main():
    # Header
    st.markdown('<p class="main-header">📚 Makale Tarama Uygulaması</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Bibliyometrik çalışmalar için akademik makale değerlendirme aracı</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # Analysis mode
        analysis_mode = st.radio(
            "Analiz Modu",
            ["🔍 Kural Tabanlı", "🤖 Gemini AI", "🧠 DeepSeek AI"],
            help="Kural tabanlı: Anahtar kelime eşleştirmesi\nGemini/DeepSeek: Yapay zeka ile akıllı analiz"
        )
        st.session_state['analysis_mode'] = analysis_mode
        
        # API Key input based on mode
        if "Gemini AI" in analysis_mode:
            api_key = st.text_input(
                "Gemini API Anahtarı",
                type="password",
                help="Google AI Studio'dan alabilirsiniz: https://aistudio.google.com/apikey"
            )
            st.session_state['api_key'] = api_key
            if not api_key:
                st.warning("⚠️ API anahtarı gerekli")
        
        elif "DeepSeek AI" in analysis_mode:
            api_key = st.text_input(
                "DeepSeek API Anahtarı",
                type="password",
                help="DeepSeek platformundan alabilirsiniz: https://platform.deepseek.com"
            )
            st.session_state['api_key'] = api_key
            if not api_key:
                st.warning("⚠️ API anahtarı gerekli")
        
        st.divider()
        
        # Column mapping
        st.subheader("📋 Sütun Eşleştirme")
        st.caption("Excel dosyanızdaki sütun isimlerini seçin")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Dosya Yükle", "⚙️ Kriterler", "📊 Analiz"])
    
    # Tab 1: File Upload
    with tab1:
        st.subheader("Excel Dosyası Yükle")
        
        uploaded_file = st.file_uploader(
            "Excel dosyanızı sürükleyip bırakın veya seçin",
            type=['xlsx', 'xls'],
            help="WoS veya Scopus'tan dışa aktarılmış Excel dosyası"
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
                        "Başlık Sütunu",
                        columns,
                        index=columns.index('Title') if 'Title' in columns else 0
                    )
                    
                    abstract_col = st.selectbox(
                        "Özet Sütunu",
                        columns,
                        index=columns.index('Abstract Note') if 'Abstract Note' in columns else (
                            columns.index('Abstract') if 'Abstract' in columns else 1
                        )
                    )
                    
                    st.session_state['title_col'] = title_col
                    st.session_state['abstract_col'] = abstract_col
                
                # Preview
                st.success(f"✅ {len(df)} makale yüklendi!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Toplam Makale", len(df))
                with col2:
                    st.metric("Sütun Sayısı", len(df.columns))
                with col3:
                    if 'Acceptance' in df.columns or 'Acceptance ' in df.columns:
                        acc_col = 'Acceptance ' if 'Acceptance ' in df.columns else 'Acceptance'
                        evaluated = df[acc_col].notna().sum()
                        st.metric("Değerlendirilmiş", evaluated)
                
                st.subheader("📋 Veri Önizleme")
                st.dataframe(df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Dosya yüklenirken hata: {e}")
    
    # Tab 2: Criteria Configuration
    with tab2:
        st.subheader("Değerlendirme Kriterleri")
        st.caption("Her bir kriter için anahtar kelimeleri virgülle ayırarak girin (Kural tabanlı mod için)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 AI Teknolojisi")
            ai_keywords = st.text_area(
                "AI anahtar kelimeleri",
                value=DEFAULT_AI_KEYWORDS,
                height=150,
                help="Makale bunlardan en az birini içermeli"
            )
            
            st.markdown("### 🌐 Web/Online Ortam")
            web_keywords = st.text_area(
                "Web ortam anahtar kelimeleri",
                value=DEFAULT_WEB_KEYWORDS,
                height=150,
                help="Makale bunlardan en az birini içermeli"
            )
        
        with col2:
            st.markdown("### 👥 İnsan Faktörü")
            human_keywords = st.text_area(
                "İnsan faktörü anahtar kelimeleri",
                value=DEFAULT_HUMAN_KEYWORDS,
                height=150,
                help="Makale bunlardan en az birini içermeli"
            )
            
            st.markdown("### 📝 Özet")
            st.info("""
            **Bir makale kabul edilir eğer:**
            1. ✅ AI teknolojisinden bahsediyorsa
            2. ✅ Web/online ortamdan bahsediyorsa
            3. ✅ İnsan kullanıcıları içeriyorsa
            
            Tüm kriterler sağlanmalıdır!
            """)
        
        # Store in session
        st.session_state['ai_keywords'] = ai_keywords
        st.session_state['web_keywords'] = web_keywords
        st.session_state['human_keywords'] = human_keywords
    
    # Tab 3: Analysis
    with tab3:
        st.subheader("Analiz ve Sonuçlar")
        
        if 'df' not in st.session_state:
            st.warning("⚠️ Lütfen önce bir Excel dosyası yükleyin")
            return
        
        df = st.session_state['df']
        title_col = st.session_state.get('title_col', 'Title')
        abstract_col = st.session_state.get('abstract_col', 'Abstract Note')
        
        # Analysis options
        col1, col2 = st.columns([2, 1])
        with col1:
            force_reprocess = st.checkbox(
                "Tüm makaleleri yeniden değerlendir",
                help="İşaretlenirse daha önce değerlendirilmiş makaleler de yeniden işlenir"
            )
        with col2:
            limit = st.number_input(
                "Limit (0 = tümü)",
                min_value=0,
                max_value=len(df),
                value=0,
                help="Test için sınırlı sayıda makale işleyin"
            )
        
        # Get current mode
        current_mode = st.session_state.get('analysis_mode', '🔍 Kural Tabanlı')
        current_api_key = st.session_state.get('api_key', None)
        
        # Show current mode
        st.info(f"📌 Seçili mod: **{current_mode}**")
        
        # Start analysis button
        if st.button("🚀 Analizi Başlat", type="primary", use_container_width=True):
            
            # Validate API key for AI modes
            if "Gemini AI" in current_mode or "DeepSeek AI" in current_mode:
                if not current_api_key:
                    st.error("❌ API anahtarı gerekli! Lütfen sidebar'dan girin.")
                    return
            
            # Parse keywords for rule-based
            ai_patterns = parse_keywords(st.session_state.get('ai_keywords', DEFAULT_AI_KEYWORDS))
            web_patterns = parse_keywords(st.session_state.get('web_keywords', DEFAULT_WEB_KEYWORDS))
            human_patterns = parse_keywords(st.session_state.get('human_keywords', DEFAULT_HUMAN_KEYWORDS))
            
            # Setup AI client if needed
            gemini_client = None
            deepseek_client = None
            
            if "Gemini AI" in current_mode:
                try:
                    from google import genai
                    gemini_client = genai.Client(api_key=current_api_key)
                    st.success("✅ Gemini AI bağlantısı kuruldu")
                except Exception as e:
                    st.error(f"❌ Gemini bağlantı hatası: {e}")
                    return
            
            elif "DeepSeek AI" in current_mode:
                try:
                    import httpx
                    from openai import OpenAI
                    # Create httpx client without proxy to avoid compatibility issues
                    http_client = httpx.Client(timeout=60.0)
                    deepseek_client = OpenAI(
                        api_key=current_api_key,
                        base_url="https://api.deepseek.com",
                        http_client=http_client
                    )
                    st.success("✅ DeepSeek AI bağlantısı kuruldu")
                except Exception as e:
                    st.error(f"❌ DeepSeek bağlantı hatası: {e}")
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
                st.info("✅ Tüm makaleler zaten değerlendirilmiş!")
                return
            
            # Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.expander("📋 İşlem Logu", expanded=True)
            
            accepted = 0
            rejected = 0
            
            with log_container:
                for i, idx in enumerate(rows_to_process):
                    title = str(df.at[idx, title_col]) if pd.notna(df.at[idx, title_col]) else ""
                    abstract = str(df.at[idx, abstract_col]) if pd.notna(df.at[idx, abstract_col]) else ""
                    
                    # Get short title for display
                    short_title = title[:50] + "..." if len(title) > 50 else title
                    
                    # Screen based on mode
                    if "Gemini AI" in current_mode and gemini_client:
                        acceptance, reason = screen_article_gemini(title, abstract, gemini_client)
                        time.sleep(0.5)  # Rate limiting
                    elif "DeepSeek AI" in current_mode and deepseek_client:
                        acceptance, reason = screen_article_deepseek(title, abstract, deepseek_client)
                        time.sleep(0.3)  # Rate limiting
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
                    status_text.text(f"İşleniyor: {i + 1}/{len(rows_to_process)} | ✅ {accepted} | ❌ {rejected}")
            
            st.session_state['df'] = df
            st.session_state['analysis_complete'] = True
            st.session_state['accepted'] = accepted
            st.session_state['rejected'] = rejected
            
            st.success(f"🎉 Analiz tamamlandı! ✅ {accepted} kabul | ❌ {rejected} red")
        
        # Show results if analysis is complete
        if st.session_state.get('analysis_complete', False):
            st.divider()
            
            accepted = st.session_state.get('accepted', 0)
            rejected = st.session_state.get('rejected', 0)
            total = accepted + rejected
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Toplam İşlenen", total)
            with col2:
                st.metric("✅ Kabul", accepted)
            with col3:
                st.metric("❌ Red", rejected)
            with col4:
                rate = (accepted / total * 100) if total > 0 else 0
                st.metric("Kabul Oranı", f"{rate:.1f}%")
            
            # Results preview
            df = st.session_state['df']
            acc_col = 'Acceptance ' if 'Acceptance ' in df.columns else 'Acceptance'
            
            st.subheader("📋 Sonuç Önizleme")
            
            filter_option = st.radio(
                "Filtre",
                ["Tümü", "✅ Kabul Edilenler", "❌ Red Edilenler"],
                horizontal=True
            )
            
            if filter_option == "✅ Kabul Edilenler":
                display_df = df[df[acc_col] == True]
            elif filter_option == "❌ Red Edilenler":
                display_df = df[df[acc_col] == False]
            else:
                display_df = df
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export
            st.divider()
            st.subheader("📥 Sonuçları İndir")
            
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
            output.seek(0)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📥 Excel Olarak İndir",
                data=output,
                file_name=f"screening_results_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
