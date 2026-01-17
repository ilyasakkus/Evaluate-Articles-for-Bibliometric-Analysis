#!/usr/bin/env python3
"""
Article Screening Application for Bibliometric Research

This application screens academic articles based on title and abstract
to determine if they meet the criteria for inclusion in a bibliometric study.

Criteria:
1. AI Technology: Must contain ChatGPT/GPT/LLM/conversational AI/AI agent/chatbot/AI
2. Web/Online Environment: Must contain online/distance/remote/e-learning/MOOC/blended/hybrid/web
3. Human Factor: Must involve teachers/students/educators in the context

Usage:
    python article_screener.py --mode rules --input Book2.xlsx
    python article_screener.py --mode ai --api-key YOUR_KEY --input Book2.xlsx
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Keywords for each criterion (case-insensitive matching)
AI_KEYWORDS = [
    r'\bchatgpt\b', r'\bgpt[-\s]?\d*\b', r'\bllm\b', r'\blarge language model\b',
    r'\bconversational ai\b', r'\bai agent\b', r'\bchatbot\b', r'\bchat bot\b',
    r'\bartificial intelligence\b', r'\bgenerative ai\b', r'\bgenai\b',
    r'\bopenai\b', r'\bgpt-4\b', r'\bgpt-3\b', r'\bgemini\b', r'\bclaude\b',
    r'\bcopilot\b', r'\bai assistant\b', r'\bai tutor\b', r'\bnlp\b',
    r'\bnatural language processing\b', r'\bmachine learning\b', r'\bdeep learning\b'
]

WEB_KEYWORDS = [
    r'\bonline\b', r'\bdistance learning\b', r'\bdistance education\b',
    r'\bremote learning\b', r'\bremote education\b', r'\be-learning\b',
    r'\belearning\b', r'\bmooc\b', r'\bmassive open online\b',
    r'\bblended learning\b', r'\bblended education\b', r'\bhybrid learning\b',
    r'\bhybrid education\b', r'\bweb-based\b', r'\bweb based\b',
    r'\bvirtual learning\b', r'\bvirtual education\b', r'\bvirtual classroom\b',
    r'\blearning management system\b', r'\blms\b', r'\bdigital learning\b',
    r'\bdigital education\b', r'\bonline course\b', r'\bonline education\b',
    r'\bonline learning\b'
]

HUMAN_KEYWORDS = [
    r'\bteacher\b', r'\bteachers\b', r'\bstudent\b', r'\bstudents\b',
    r'\binstructor\b', r'\binstructors\b', r'\blearner\b', r'\blearners\b',
    r'\beducator\b', r'\beducators\b', r'\badministrator\b', r'\badministrators\b',
    r'\bfaculty\b', r'\btutor\b', r'\btutors\b', r'\bpupil\b', r'\bpupils\b',
    r'\bprofessor\b', r'\bprofessors\b', r'\bacademic\b', r'\bacademics\b',
    r'\bundergraduate\b', r'\bgraduate\b', r'\bpostgraduate\b',
    r'\buniversity student\b', r'\bcollege student\b', r'\bhigh school student\b',
    r'\bk-12\b', r'\bhigher education\b'
]


def check_keywords(text: str, keywords: list) -> tuple[bool, list]:
    """Check if text contains any of the keywords."""
    if not text or pd.isna(text):
        return False, []
    
    text_lower = text.lower()
    found = []
    for pattern in keywords:
        if re.search(pattern, text_lower):
            found.append(pattern.replace(r'\b', '').replace('\\', ''))
    
    return len(found) > 0, found


def check_ai_technology(text: str) -> tuple[bool, list]:
    """Check if text contains AI technology keywords."""
    return check_keywords(text, AI_KEYWORDS)


def check_web_environment(text: str) -> tuple[bool, list]:
    """Check if text contains web/online environment keywords."""
    return check_keywords(text, WEB_KEYWORDS)


def check_human_factor(text: str) -> tuple[bool, list]:
    """Check if text contains human factor keywords."""
    return check_keywords(text, HUMAN_KEYWORDS)


def screen_article_rules(title: str, abstract: str) -> tuple[bool, str]:
    """
    Screen an article using rule-based keyword matching.
    
    Returns:
        tuple: (acceptance: bool, reason: str)
    """
    # Combine title and abstract for analysis
    combined_text = f"{title or ''} {abstract or ''}"
    
    # Check each criterion
    has_ai, ai_found = check_ai_technology(combined_text)
    has_web, web_found = check_web_environment(combined_text)
    has_human, human_found = check_human_factor(combined_text)
    
    # Determine acceptance and reason
    missing_criteria = []
    
    if not has_ai:
        missing_criteria.append("AI teknolojisi (ChatGPT/GPT/LLM/chatbot vb.) bulunamadı")
    if not has_web:
        missing_criteria.append("Web/online ortam (online/e-learning/MOOC vb.) bulunamadı")
    if not has_human:
        missing_criteria.append("İnsan faktörü (öğretmen/öğrenci vb.) bulunamadı")
    
    if not missing_criteria:
        return True, ""
    else:
        return False, "; ".join(missing_criteria)


def screen_article_gemini(title: str, abstract: str, model) -> tuple[bool, str]:
    """
    Screen an article using Gemini AI.
    
    Returns:
        tuple: (acceptance: bool, reason: str)
    """
    prompt = f"""Aşağıdaki akademik makaleyi bibliyometrik çalışma için değerlendir.

BAŞLIK: {title}

ÖZET: {abstract}

DEĞERLENDİRME KRİTERLERİ:
1. AI Teknolojisi: Makale ChatGPT, GPT, LLM, conversational AI, AI agent, chatbot veya benzeri AI teknolojilerinden bahsetmeli
2. Web/Online Ortam: Makale online, distance, remote, e-learning, MOOC, blended, hybrid veya web tabanlı eğitimden bahsetmeli
3. İnsan Faktörü: Makale öğretmen, öğrenci, eğitimci veya yönetici gibi insan kullanıcıları içermeli

GÖREV:
- Tüm kriterler sağlanıyorsa: "KABUL" yaz
- Herhangi bir kriter sağlanmıyorsa: "RED: [eksik kriterleri Türkçe açıkla]" yaz

Sadece KABUL veya RED: [sebep] formatında yanıt ver, başka açıklama ekleme.
"""
    
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        if result.upper().startswith("KABUL"):
            return True, ""
        elif result.upper().startswith("RED"):
            reason = result.replace("RED:", "").replace("RED", "").strip()
            return False, reason if reason else "Kriterler sağlanmadı"
        else:
            # Fallback to rule-based if AI response is unclear
            return screen_article_rules(title, abstract)
    except Exception as e:
        print(f"\n⚠️  Gemini API hatası: {e}")
        print("   Kural tabanlı değerlendirmeye geçiliyor...")
        return screen_article_rules(title, abstract)


def main():
    parser = argparse.ArgumentParser(
        description="Bibliyometrik çalışma için makale tarama uygulaması",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python article_screener.py --mode rules --input Book2.xlsx
  python article_screener.py --mode ai --api-key YOUR_KEY --input Book2.xlsx
  python article_screener.py --mode rules --input Book2.xlsx --limit 10
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Giriş Excel dosyası (örn: Book2.xlsx)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["rules", "ai"],
        default="rules",
        help="Tarama modu: 'rules' (kural tabanlı) veya 'ai' (Gemini AI)"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        help="Gemini API anahtarı (sadece AI modu için gerekli)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="İşlenecek maksimum makale sayısı (test için)"
    )
    
    parser.add_argument(
        "--skip-evaluated",
        action="store_true",
        default=True,
        help="Zaten değerlendirilmiş makaleleri atla (varsayılan: True)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tüm makaleleri yeniden değerlendir (mevcut sonuçları sil)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Hata: Dosya bulunamadı: {args.input}")
        sys.exit(1)
    
    # Validate AI mode requirements
    if args.mode == "ai" and not args.api_key:
        print("❌ Hata: AI modu için --api-key gerekli")
        sys.exit(1)
    
    # Setup Gemini if AI mode
    gemini_model = None
    if args.mode == "ai":
        try:
            import google.generativeai as genai
            genai.configure(api_key=args.api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini AI bağlantısı kuruldu")
        except Exception as e:
            print(f"❌ Gemini API hatası: {e}")
            sys.exit(1)
    
    # Load Excel file
    print(f"\n📖 Excel dosyası yükleniyor: {args.input}")
    df = pd.read_excel(input_path)
    total_rows = len(df)
    print(f"   Toplam makale: {total_rows}")
    
    # Determine which rows to process
    if args.force:
        rows_to_process = df.index.tolist()
    else:
        # Skip rows that already have Acceptance value
        mask = df['Acceptance '].isna() | (df['Acceptance '] == '')
        rows_to_process = df[mask].index.tolist()
    
    if args.limit:
        rows_to_process = rows_to_process[:args.limit]
    
    print(f"   İşlenecek makale: {len(rows_to_process)}")
    
    if len(rows_to_process) == 0:
        print("\n✅ Tüm makaleler zaten değerlendirilmiş!")
        print("   Yeniden değerlendirmek için --force kullanın.")
        return
    
    # Process articles
    mode_text = "Kural Tabanlı" if args.mode == "rules" else "Gemini AI"
    print(f"\n🔍 Makaleler taranıyor ({mode_text} mod)...\n")
    
    accepted_count = 0
    rejected_count = 0
    
    for idx in tqdm(rows_to_process, desc="İlerleme", unit="makale"):
        title = df.at[idx, 'Title']
        abstract = df.at[idx, 'Abstract Note']
        
        if args.mode == "rules":
            acceptance, reason = screen_article_rules(title, abstract)
        else:
            acceptance, reason = screen_article_gemini(title, abstract, gemini_model)
        
        df.at[idx, 'Acceptance '] = acceptance
        df.at[idx, 'Reason'] = reason
        
        if acceptance:
            accepted_count += 1
        else:
            rejected_count += 1
    
    # Save results
    print(f"\n💾 Sonuçlar kaydediliyor: {args.input}")
    df.to_excel(input_path, index=False)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 ÖZET")
    print("=" * 50)
    print(f"   Toplam işlenen: {len(rows_to_process)}")
    print(f"   ✅ Kabul edilen: {accepted_count}")
    print(f"   ❌ Reddedilen:   {rejected_count}")
    print(f"   Kabul oranı:    {accepted_count/len(rows_to_process)*100:.1f}%")
    print("=" * 50)
    print(f"\n✨ Tamamlandı! Sonuçlar {args.input} dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
