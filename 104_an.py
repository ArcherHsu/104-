import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
import jieba
import wordcloud
from PIL import Image
warnings.filterwarnings('ignore')

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

def analyze_salary_percentiles(df):
    """分析薪資的十分位數分布"""
    try:
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 過濾有效薪資數據
        salary_data = pd.to_numeric(df[df['薪資範圍'] != '未提供']['薪資範圍'], errors='coerce').dropna()
        
        # 計算十分位數
        percentiles = []
        for i in range(0, 101, 10):
            if i == 0:
                continue
            percentiles.append(salary_data.quantile(i/100))
        
        # 計算平均薪資
        mean_salary = salary_data.mean()
        
        # 創建圖表
        plt.figure(figsize=(15, 8))
        
        # 繪製柱狀圖
        bars = plt.bar(range(1, 11), percentiles)
        
        # 添加平均薪資線
        plt.axhline(y=mean_salary, color='r', linestyle='--', label=f'平均薪資: {mean_salary:,.0f}元/月')
        
        # 在柱子上標示數值
        for i, v in enumerate(percentiles):
            plt.text(i+1, v, f'{v:,.0f}', ha='center', va='bottom')
        
        # 設置標題和標籤
        plt.title('薪資十分位數分布', pad=20, fontsize=14)
        plt.xlabel('十分位數', fontsize=12)
        plt.ylabel('薪資 (元/月)', fontsize=12)
        
        # 設置X軸刻度
        plt.xticks(range(1, 11), [f'P{i*10}' for i in range(1, 11)])
        
        # 添加圖例
        plt.legend()
        
        # 調整布局
        plt.tight_layout()
        
        # 保存圖片
        plt.savefig('薪資十分位數分布.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"薪資分析錯誤：{str(e)}")

def analyze_tool_associations(df):
    """使用Apriori演算法分析工具關聯性並視覺化"""
    try:
        # 建立工具列表
        all_tools = set()
        tool_lists = []
        
        # 收集所有工具
        for tools in df['擅長工具']:
            if isinstance(tools, str) and tools != '未提供':
                tool_list = [t.strip() for t in tools.split(';')]
                tool_lists.append(tool_list)
                all_tools.update(tool_list)
        
        # 創建one-hot編碼矩陣
        tool_matrix = pd.DataFrame(columns=list(all_tools))
        
        # 填充矩陣
        for i, tools in enumerate(tool_lists):
            for tool in tools:
                tool_matrix.loc[i, tool] = 1
        
        # 填充NaN為0
        tool_matrix = tool_matrix.fillna(0).astype(bool)
        
        # 執行Apriori演算法
        frequent_itemsets = apriori(tool_matrix, 
                                  min_support=0.1,    
                                  use_colnames=True)
        
        # 產生關聯規則
        rules = association_rules(frequent_itemsets, 
                                metric="confidence",
                                min_threshold=0.5)
        
        # 排序規則並取前20
        rules = rules.sort_values(['lift'], ascending=[False]).head(20)
        
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 創建圖表
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 隱藏軸線
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # 設置表格數據
        cell_text = []
        for idx, row in rules.iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            cell_text.append([
                antecedents,
                consequents,
                f"{row['support']:.3f}",
                f"{row['confidence']:.3f}",
                f"{row['lift']:.3f}"
            ])

        # 創建表格
        table = ax.table(cellText=cell_text,
                        colLabels=['前項', '後項', '支持度', '信心度', '提升度'],
                        loc='center',
                        cellLoc='center')
        
        # 調整表格樣式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 設置標題
        plt.title('工具關聯規則 Top 20', pad=20, fontsize=14)
        
        # 隱藏座標軸
        plt.axis('off')
        
        # 調整布局
        plt.tight_layout()
        
        # 保存圖片
        plt.savefig('工具關聯規則.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 儲存詳細結果到CSV
        rules.to_csv('工具關聯分析.csv', encoding='utf-8-sig', index=False)
        
    except Exception as e:
        print(f"關聯分析錯誤：{str(e)}")

def plot_skills_and_categories(df):
    """繪製工作技能和職務類別分布圖"""
    try:
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 統計工作技能
        all_skills = []
        for skills in df['工作技能']:
            if isinstance(skills, str) and skills != '未提供':
                cleaned_skills = [skill.strip() for skill in skills.split(';')]
                all_skills.extend(cleaned_skills)
        
        skill_counts = Counter(all_skills).most_common(20)
        
        # 統計職務類別
        all_categories = []
        for cats in df['職務類別']:
            if isinstance(cats, str) and cats != '未提供':
                cleaned_cats = [cat.strip() for cat in cats.split(';')]
                all_categories.extend(cleaned_cats)
        
        category_counts = Counter(all_categories).most_common(20)

        # 繪製工作技能分布圖
        plt.figure(figsize=(15, 10))
        
        # 工作技能
        skills, counts = zip(*skill_counts)
        plt.barh(range(len(counts)), counts)
        plt.yticks(range(len(skills)), skills)
        plt.gca().invert_yaxis()
        plt.title('工作技能需求 Top 20', pad=20, fontsize=14)
        plt.xlabel('數量', fontsize=12)
        
        for i, v in enumerate(counts):
            plt.text(v, i, f' {v}', va='center')

        plt.tight_layout()
        plt.savefig('工作技能需求.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 繪製職務類別分布圖
        plt.figure(figsize=(15, 10))
        cats, cat_counts = zip(*category_counts)
        plt.barh(range(len(cat_counts)), cat_counts)
        plt.yticks(range(len(cats)), cats)
        plt.gca().invert_yaxis()
        plt.title('職務類別分布 Top 20', pad=20, fontsize=14)
        plt.xlabel('數量', fontsize=12)
        
        for i, v in enumerate(cat_counts):
            plt.text(v, i, f' {v}', va='center')

        plt.tight_layout()
        plt.savefig('職務類別分布.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 繪製各縣市需求數量圖
        plt.figure(figsize=(15, 8))
        city_counts = df['工作地點'].value_counts()
        city_counts.plot(kind='bar')
        plt.title('各縣市需求數量', fontsize=16, pad=20)
        plt.xlabel('縣市', fontsize=12)
        plt.ylabel('數量', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        for i, v in enumerate(city_counts.values):
            plt.text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('各縣市需求數量.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 繪製工具需求圖
        plt.figure(figsize=(15, 10))
        tool_counts = Counter()
        for tools in df['擅長工具']:
            if isinstance(tools, str) and tools != '未提供':
                for tool in tools.split(';'):
                    tool_counts[tool.strip()] += 1
        
        top_tools = tool_counts.most_common(20)
        tools, tool_nums = zip(*top_tools)
        plt.barh(range(len(tool_nums)), tool_nums)
        plt.yticks(range(len(tools)), tools)
        plt.gca().invert_yaxis()
        plt.title('工具需求 Top 20', pad=20, fontsize=14)
        plt.xlabel('數量', fontsize=12)
        
        for i, v in enumerate(tool_nums):
            plt.text(v, i, f' {v}', va='center')

        plt.tight_layout()
        plt.savefig('工具需求.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 繪製工作經歷與薪資分布圖
        plt.figure(figsize=(15, 8))
        
        # 準備數據
        exp_salary_data = []
        exp_counts = []
        
        # 對每種工作經歷要求計算平均薪資
        for exp in df['工作經歷'].unique():
            mask = (df['工作經歷'] == exp) & (df['薪資範圍'] != '未提供')
            salaries = pd.to_numeric(df[mask]['薪資範圍'], errors='coerce')
            
            if not salaries.empty:
                avg_salary = salaries.mean()
                count = len(salaries)
                exp_salary_data.append((exp, avg_salary, count))
        
        # 排序數據（按照年資排序）
        def exp_sort_key(item):
            exp = item[0]
            if '不拘' in exp:
                return -1
            elif '年以上' in exp:
                return float(exp.replace('年以上', ''))
            else:
                return float(exp.replace('年', ''))
        
        exp_salary_data.sort(key=exp_sort_key)
        
        # 準備繪圖數據
        exps, salaries, counts = zip(*exp_salary_data)
        
        # 繪製柱狀圖
        bars = plt.bar(range(len(exps)), salaries)
        plt.title('各工作經歷要求之平均薪資分布', fontsize=14, pad=20)
        plt.xlabel('工作經歷要求', fontsize=12)
        plt.ylabel('平均薪資 (元/月)', fontsize=12)
        
        # 設置 X 軸標籤
        plt.xticks(range(len(exps)), exps, rotation=45, ha='right')
        
        # 在每個柱子上標示平均薪資和樣本數
        for i, (bar, salary, count) in enumerate(zip(bars, salaries, counts)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{salary:,.0f}\nn={count}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('工作經歷薪資分布.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"繪圖錯誤：{str(e)}")

def analyze_job_content_wordcloud(df):
    """分析工作內容文字雲"""
    try:
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 載入停用詞和字典
        global stopwords
        stopwords = [k.strip() for k in open('停用詞.txt', encoding='utf-8') if k.strip() != '']
        jieba.set_dictionary('dict.txt.big')
        
        # 將工具的專有名詞加入自定義詞
        tool_counts = Counter()
        for tools in df['擅長工具']:
            if isinstance(tools, str) and tools != '未提供':
                for tool in tools.split(';'):
                    tool_counts[tool.strip()] += 1
                    jieba.add_word(tool.strip())
        
        # 清理文本並斷詞
        job_content = df['工作內容']
        job_content_cut = [
            word for sent in job_content 
            for word in set(text_cut(sent)) 
            if word not in '數據分析'
        ]
        job_content_cut_cnt = Counter(job_content_cut)
        
        # 生成文字雲，使用系統字體
        wc = wordcloud.WordCloud(
            font_path='msyh.ttc',  # 使用微軟雅黑字體，Windows系統通常都有
            max_words=40,
            max_font_size=180,
            background_color='white',
            width=800, height=600,
        )
        wc.generate_from_frequencies(job_content_cut_cnt)
        
        # 創建圖表
        plt.figure(figsize=(10, 8))
        plt.imshow(wc)
        plt.axis('off')
        
        # 保存圖片
        plt.savefig('工作內容文字雲.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"文字雲分析錯誤：{str(e)}")
        # 打印更詳細的錯誤信息
        import traceback
        print(traceback.format_exc())

def clean_Punctuation(text):
    """清除標點符號"""
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '').replace('[', '').replace(']', '')
    return text

def text_cut(sentence):
    """斷詞並去除停用詞"""
    sentence_cut = [
        word for word in jieba.lcut(clean_Punctuation(sentence)) 
        if word not in stopwords
    ]
    return sentence_cut

def process_jobs():
    try:
        # 讀取原始 CSV 檔案
        df = pd.read_csv('104_jobs.csv', encoding='utf-8-sig')
        
        # 轉換工作地點為城市名稱
        df['工作地點'] = df['工作地點'].apply(extract_city)
        
        # 轉換薪資為月薪平均值
        df['薪資範圍'] = df['薪資範圍'].apply(convert_salary_to_monthly_avg)
        
        # 儲存新的 CSV 檔案
        df.to_csv('104_re.csv', encoding='utf-8-sig', index=False)
        print("已成功創建新的 CSV 檔案：104_re.csv")
        
        # 呼叫繪圖函數
        plot_skills_and_categories(df)
        
        # 執行工具關聯性分析
        analyze_tool_associations(df)
        
        # 執行薪資十分位數分析
        analyze_salary_percentiles(df)

        
        # 執行學歷占比分析
        analyze_education_ratio(df)

        # 執行文字雲分析
        analyze_job_content_wordcloud(df)

    except Exception as e:
        print(f"處理錯誤：{str(e)}")

def analyze_education_ratio(df):
    """分析學歷占比分布"""
    try:
        # 設置中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 學歷排序順序
        edu_order = ['不拘', '高中', '專科', '大學', '碩士', '博士']
        
        # 計算各學歷的數量
        edu_counts = df['學歷要求'].value_counts()
        
        # 計算總數
        total = edu_counts.sum()
        
        # 計算比例
        edu_ratio = edu_counts / total * 100
        
        # 按照指定順序排序
        edu_ratio = edu_ratio.reindex(edu_order)
        
        # 創建圖表
        plt.figure(figsize=(12, 6))
        
        # 繪製柱狀圖
        bars = plt.bar(range(len(edu_ratio)), edu_ratio)
        
        # 在柱子上標示百分比和樣本數
        for i, (ratio, count) in enumerate(zip(edu_ratio, edu_counts.reindex(edu_order))):
            plt.text(i, ratio, f'{ratio:.1f}%\nn={count}', 
                    ha='center', va='bottom')
        
        # 設置標題和標籤
        plt.title('學歷占比', pad=20, fontsize=14)
        plt.xlabel('學歷', fontsize=12)
        plt.ylabel('占比 (%)', fontsize=12)
        
        # 設置X軸刻度
        plt.xticks(range(len(edu_ratio)), edu_order)
        
        # 調整布局
        plt.tight_layout()
        
        # 保存圖片
        plt.savefig('學歷占比.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"學歷占比分析錯誤：{str(e)}")


if __name__ == "__main__":
    process_jobs()