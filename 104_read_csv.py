import pandas as pd
import re

def convert_salary_to_monthly_avg(salary_str):
    """將薪資範圍轉換為月薪平均值"""
    try:
        if not isinstance(salary_str, str) or salary_str.strip() == '':
            return '未提供'
        
        # 移除所有空格和多餘符號
        salary_str = salary_str.replace(" ", "").replace("，", ",")
        
        # 面議或未提供則返回未提供
        if "面議" in salary_str or "未提供" in salary_str:
            return '未提供'
        
        # 提取所有數字
        numbers = re.findall(r'\d+(?:,\d+)*', salary_str)
        if not numbers:
            return '未提供'
            
        # 轉換數字為整數
        numbers = [int(n.replace(",", "")) for n in numbers]
        
        # 時薪轉換
        if "時薪" in salary_str:
            numbers = [n * 8 * 22 for n in numbers]  # 轉換為月薪
            
        # 年薪轉換
        elif "年薪" in salary_str:
            numbers = [int(n / 12) for n in numbers]  # 轉換為月薪
            
        # 計算平均值
        if len(numbers) == 1:
            return float(numbers[0])
        else:
            return sum(numbers) / len(numbers)
            
    except Exception as e:
        print(f"薪資轉換錯誤 '{salary_str}': {str(e)}")
        return '未提供'

def extract_city(location):
    """從工作地點提取縣市名稱"""
    cities = ['台北市', '新北市', '桃園市', '台中市', '台南市', '高雄市', 
             '基隆市', '新竹市', '新竹縣', '苗栗縣', '彰化縣', '南投縣', 
             '雲林縣', '嘉義市', '嘉義縣', '屏東縣', '宜蘭縣', '花蓮縣', 
             '台東縣', '澎湖縣', '金門縣', '連江縣']
    
    for city in cities:
        if city in location:
            return city
    return location

def extract_skills_from_description(description):
    """從工作內容提取技能關鍵字"""
    skill_keywords = [
        'Python', 'R', 'SQL', 'Java', 'C++', 'JavaScript', 'PHP',
        'pandas', 'numpy', 'scipy', 'scikit-learn', 'tensorflow', 'pytorch',
        'matplotlib', 'seaborn', 'powerbi', 'tableau', 'excel', 'spss',
        'hadoop', 'spark', 'hive', 'mysql', 'postgresql', 'mongodb',
        'docker', 'kubernetes', 'git', 'linux', 'aws', 'gcp', 'azure',
        'Machine Learning'
    ]
    
    found_skills = []
    if isinstance(description, str):
        description_lower = description.lower()
        for skill in skill_keywords:
            if skill.lower() in description_lower:
                found_skills.append(skill)
    return found_skills

def process_jobs():
    try:
        # 讀取原始 CSV 檔案
        df = pd.read_csv('104_jobs.csv', encoding='utf-8-sig')
        
        # 轉換工作地點為城市名稱
        df['工作地點'] = df['工作地點'].apply(extract_city)
        
        # 從工作內容提取技能並更新擅長工具
        def update_skills(row):
            # 初始化技能集合
            current_skills = set()
            
            # 處理現有的擅長工具
            if pd.notna(row['擅長工具']) and row['擅長工具'] != '未提供':
                current_skills.update(skill.strip() for skill in row['擅長工具'].split(';'))
            
            # 從工作內容提取新技能
            if pd.notna(row['工作內容']):
                new_skills = extract_skills_from_description(row['工作內容'])
                current_skills.update(new_skills)
            
            # 移除空值
            current_skills = {s for s in current_skills if s and s != '未提供'}
            
            # 如果沒有任何技能，返回'未提供'
            return ';'.join(sorted(current_skills)) if current_skills else '未提供'
        
        # 更新擅長工具欄位
        df['擅長工具'] = df.apply(update_skills, axis=1)
        
        # 轉換薪資為月薪平均值
        df['薪資範圍'] = df['薪資範圍'].apply(convert_salary_to_monthly_avg)
        
        # 儲存新的 CSV 檔案
        df.to_csv('104_re.csv', encoding='utf-8-sig', index=False)
        print("已成功創建新的 CSV 檔案：104_re.csv")
        
    except Exception as e:
        print(f"處理錯誤：{str(e)}")

if __name__ == "__main__":
    process_jobs()