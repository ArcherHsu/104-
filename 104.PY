#!C:/Users/您的使用者名稱/anaconda3/envs/您的環境名稱/python.exe
from playwright.sync_api import sync_playwright
import json
import requests
import pandas as pd
from datetime import datetime

# 定義技能關鍵字列表
SKILL_KEYWORDS = {
    '程式語言': ['Python', 'R', 'SQL', 'Java', 'C++', 'JavaScript', 'PHP'],
    '資料處理': ['Excel', 'Power BI', 'Tableau', 'Looker', 'pandas', 'numpy'],
    '資料庫': ['MySQL', 'PostgreSQL', 'MongoDB', 'Oracle', 'SQL Server'],
    '統計工具': ['SPSS', 'SAS', 'Minitab', 'Stata'],
    '其他工具': ['Git', 'Docker', 'Linux', 'AWS', 'GCP', 'Azure']
}

def extract_skills_from_description(description):
    """從描述文字中提取技能關鍵字"""
    found_skills = []
    description_upper = description.upper()
    for category, skills in SKILL_KEYWORDS.items():
        for skill in skills:
            if skill.upper() in description_upper:
                found_skills.append(skill)
    return list(set(found_skills))

def get_job_tools(job_no):
    """獲取職缺的工具技能"""
    api_url = f"https://www.104.com.tw/job/ajax/content/{job_no}"
    headers = {
        "Referer": f"https://www.104.com.tw/job/{job_no}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # 獲取工具技能
            tools = data.get('data', {}).get('condition', {}).get('specialty', [])
            tool_list = [tool.get('description', '') for tool in tools if tool.get('description')]
            
            # 獲取工作技能
            skills = data.get('data', {}).get('condition', {}).get('skill', [])
            skill_list = [skill.get('description', '') for skill in skills if skill.get('description')]
            
            if not tool_list:
                description = data.get('data', {}).get('jobDetail', {}).get('jobDescription', '')
                tool_list = extract_skills_from_description(description)
            
            return tool_list if tool_list else ['未提供'], skill_list if skill_list else ['未提供']
            
    except Exception as e:
        print(f"獲取技能資訊時發生錯誤：{str(e)}")
    return ['未提供'], ['未提供']

def get_job_details(job_no):
    """從 API 獲取職缺詳細資訊"""
    api_url = f"https://www.104.com.tw/job/ajax/content/{job_no}"
    headers = {
        "Referer": f"https://www.104.com.tw/job/{job_no}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            job_categories = data.get('data', {}).get('jobDetail', {}).get('jobCategory', [])
            job_category_names = [cat.get('description', '') for cat in job_categories]
            
            return {
                'description': data.get('data', {}).get('jobDetail', {}).get('jobDescription', ''),
                'post_date': data.get('data', {}).get('header', {}).get('appearDate', '未提供'),
                'apply_date': data.get('data', {}).get('header', {}).get('applyDate', '未提供'),
                'job_content': data.get('data', {}).get('jobDetail', {}).get('jobDescription', '未提供'),
                'job_category': '; '.join(job_category_names) if job_category_names else '未提供'
            }
    except Exception as e:
        print(f"獲取職缺詳細資訊時發生錯誤：{str(e)}")
    return {'description': '', 'post_date': '未提供', 'apply_date': '未提供', 'job_content': '未提供', 'job_category': '未提供'}

def scrape_104():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        all_jobs = []
        
        for page_num in range(1, 2):
            url = f"https://www.104.com.tw/jobs/search/?keyword=數據分析師&page={page_num}"
            print(f"\n正在訪問第 {page_num} 頁...")
            page.goto(url, timeout=60000)
            
            page.wait_for_selector(".vue-recycle-scroller__item-view", timeout=60000)
            print(f"第 {page_num} 頁加載完成")
            
            page.wait_for_timeout(3000)
            
            jobs = page.query_selector_all(".vue-recycle-scroller__item-view")
            print(f"本頁找到 {len(jobs)} 個職缺")
            
            for job in jobs:
                try:
                    title_elem = job.query_selector(".info-job__text")
                    company_elem = job.query_selector(".info-company__text")
                    location_elem = job.query_selector(".info-tags__text a")
                    salary_elem = job.query_selector(".info-tags__text a[href*='scmin']")
                    
                    requirements = {
                        "工作經歷": job.query_selector("a[href*='jobexp']").inner_text().strip() if job.query_selector("a[href*='jobexp']") else "未提供",
                        "學歷要求": job.query_selector("a[href*='edu']").inner_text().strip() if job.query_selector("a[href*='edu']") else "未提供",
                    }
                    
                    job_link = title_elem.get_attribute("href")
                    job_id = job_link.split('job/')[1].split('?')[0]
                    
                    job_details = get_job_details(job_id)
                    tools, skills = get_job_tools(job_id)
                    
                    if title_elem and company_elem:
                        job_data = {
                            "職缺名稱": title_elem.inner_text().strip(),
                            "公司名稱": company_elem.inner_text().strip(),
                            "工作地點": location_elem.inner_text().strip() if location_elem else "未提供",
                            "薪資範圍": salary_elem.inner_text().strip() if salary_elem else "未提供",
                            "職缺連結": job_link,
                            "發布日期": job_details['post_date'],
                            "截止日期": job_details['apply_date'],
                            "工作經歷": requirements["工作經歷"],
                            "學歷要求": requirements["學歷要求"],
                            "擅長工具": tools,
                            "工作技能": skills,
                            "工作內容": job_details['job_content'],
                            "職務類別": job_details['job_category']
                        }
                        
                        all_jobs.append(job_data)
                        print(f"成功抓取：{job_data['職缺名稱']} - {job_data['公司名稱']}")
                    
                except Exception as e:
                    print(f"抓取職缺時發生錯誤：{str(e)}")
                    continue
            
            print(f"第 {page_num} 頁抓取完成，共 {len(all_jobs)} 筆資料")
            if page_num < 10:
                page.wait_for_timeout(2000)
        
        browser.close()
        return all_jobs

def save_to_csv(job_data):
    try:
        df = pd.DataFrame(job_data)
        df['擅長工具'] = df['擅長工具'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
        df['工作技能'] = df['工作技能'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
        
        filename = '104_jobs.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig', mode='w')
        print(f"\n資料已成功儲存至 {filename}")
        
        print(f"\n資料統計：")
        print(f"總職缺數：{len(df)}")
        print(f"欄位：{', '.join(df.columns)}")
        
    except Exception as e:
        print(f"儲存 CSV 時發生錯誤：{str(e)}")

def main():
    job_data = scrape_104()
    print("\n=== 抓取結果摘要 ===")
    print(f"總共抓取了 {len(job_data)} 筆職缺資料")
    save_to_csv(job_data)
    
    for i, job in enumerate(job_data, 1):
        print(f"\n第 {i} 筆職缺：")
        for key, value in job.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()

def test_connection():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://www.104.com.tw")
        print("成功訪問104首頁")
        browser.close()

test_connection()

