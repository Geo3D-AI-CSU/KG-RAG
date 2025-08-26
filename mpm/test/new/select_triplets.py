import pandas as pd
import random
from typing import List, Dict

class TripletsCategorizer:
    def __init__(self, kg_files: List[str]):
        """
        初始化三元组分类器
        """
        self.kg_files = kg_files
        self.triplets = []
        self.triplets_df = None
        self.categories = {}
        
    def load_knowledge_graph(self):
        """
        加载知识图谱数据
        """
        all_triplets = []
        for kg_file in self.kg_files:
            df = pd.read_csv(kg_file)
            # 跳过标题行
            triplets = df.values.tolist()
            all_triplets.extend(triplets)
        
        self.triplets = all_triplets
        self.triplets_df = pd.DataFrame(all_triplets, columns=['Subject', 'Relation', 'Object'])
        print(f"总共加载了 {len(self.triplets)} 个三元组")
        
    def categorize_triplets(self) -> Dict[str, int]:
        """
        对三元组进行分类统计（8类地质本体）
        """
        # 定义分类规则（根据地质本体8要素）
        category_rules = {
            '地理': ['位于', '距离', '经纬度', '范围', '方位', '相邻', '隶属', '面积', '交通', '方向', '东南方向', '最高气温', '最低气温', '年平均气温', '年降雨量', '经度', '纬度', '气候', '属'],
            '地层': ['地层', '岩性', '岩组', '层位', '接触', '覆盖', '夹', '互层', '夹层', '厚度'],
            '岩体': ['岩体', '岩株', '侵入', '侵位于', '火山岩', '岩相', '形式', '形态', '出露面积', '是'],
            '构造': ['褶皱', '断裂', '节理', '劈理', '构造', '岩浆岩', '轴向', '发育于', '走向', '倾向', '倾角', '作用', '控制', '被叠加于', '水文地质条件', '揭露', '包含', '圈定'],
            '地质年代': ['年代', '时代', '纪', '世', '期', '阶段', '属', '古生界', '元古界', '新生界'],
            '成矿要素': ['矿体', '矿化', '矿物', '矿床', '品位', '厚度', '蚀变', '成因', '控矿', '找矿', '成矿', '成矿物质', '包括', '数量', '圈定', '分类', '构成', '产于', '正相关', '相关', '磁异常', '电阻表现', '钨矿信息', '主要矿产', '金属', '金属组合', '找矿标志', '热液活动', '形成'],
            '地球物理': ['重力', '磁场', '地震', '电性', '磁性', '密度', '磁异常', '电阻表现', '重磁异常'],
            '地球化学': ['元素', '成分', '含量', '化学', '微量元素', '稀土', '亏损', '富集', '高含量', '富集', '化学异常调查']
        }
        
        self.categories = {cat: 0 for cat in category_rules.keys()}
        
        # 建立关系到类别的映射表
        relation_to_category = {}
        for category, keywords in category_rules.items():
            for keyword in keywords:
                relation_to_category[keyword] = category
        
        # 统计分类
        for _, row in self.triplets_df.iterrows():
            relation = row['Relation']
            categorized = False
            
            # 精确匹配
            for keyword, category in relation_to_category.items():
                if keyword in relation:
                    self.categories[category] += 1
                    categorized = True
                    break
            
            # 未匹配项的模糊匹配
            if not categorized:
                # 数值型关系处理
                if any(char.isdigit() for char in str(row['Object'])):
                    if 'm' in str(row['Object']) or '厚度' in relation:
                        self.categories['地层'] += 1
                    elif '品位' in relation or '%' in str(row['Object']) or '高' in str(row['Object']):
                        self.categories['成矿要素'] += 1
                    elif '磁异常' in relation or '电阻' in relation:
                        self.categories['成矿要素'] += 1
                    elif '°' in str(row['Object']) or '气温' in relation:
                        self.categories['地理'] += 1
                    else:
                        self.categories['地球化学'] += 1
                # 特殊关系处理
                elif '主要矿产' in relation:
                    self.categories['成矿要素'] += 1
                elif '岩性' in row['Subject'] or '岩性' in row['Object']:
                    self.categories['地层'] += 1
                elif '岩体' in row['Subject'] or '岩体' in row['Object']:
                    self.categories['岩体'] += 1
                # 其他通用关系
                elif relation in ['矿石矿物', '金属组合', '成矿潜力']:
                    self.categories['成矿要素'] += 1
                elif relation in ['重磁异常', '磁性']:
                    self.categories['地球物理'] += 1
                elif relation in ['轻稀土', '重稀土']:
                    self.categories['地球化学'] += 1
                else:
                    # 统计未匹配三元组
                    self.categories['未匹配'] = self.categories.get('未匹配', 0) + 1
                    # print(f"未匹配三元组: {relation} -> {row['Object']}")
        
        return self.categories
    
    def categorize_selected_triplets(self) -> Dict[str, int]:
        """
        对选中的100个三元组进行分类统计
        """
        # 读取选中的三元组
        selected_df = pd.read_csv('selected_triplets.csv')
        
        # 定义分类规则（根据地质本体8要素）
        category_rules = {
            '地理': ['位于', '距离', '经纬度', '范围', '方位', '相邻', '隶属', '面积', '交通', '方向', '东南方向', '最高气温', '最低气温', '年平均气温', '年降雨量', '经度', '纬度', '气候', '属'],
            '地层': ['地层', '岩性', '岩组', '层位', '接触', '覆盖', '夹', '互层', '夹层', '厚度'],
            '岩体': ['岩体', '岩株', '侵入', '侵位于', '火山岩', '岩相', '形式', '形态', '出露面积', '是'],
            '构造': ['褶皱', '断裂', '节理', '劈理', '构造', '岩浆岩', '轴向', '发育于', '走向', '倾向', '倾角', '作用', '控制', '被叠加于', '水文地质条件', '揭露', '包含', '圈定'],
            '地质年代': ['年代', '时代', '纪', '世', '期', '阶段', '属', '古生界', '元古界', '新生界'],
            '成矿要素': ['矿体', '矿化', '矿物', '矿床', '品位', '厚度', '蚀变', '成因', '控矿', '找矿', '成矿', '成矿物质', '包括', '数量', '圈定', '分类', '构成', '产于', '正相关', '相关', '磁异常', '电阻表现', '钨矿信息', '主要矿产', '金属', '金属组合', '找矿标志', '热液活动', '形成'],
            '地球物理': ['重力', '磁场', '地震', '电性', '磁性', '密度', '磁异常', '电阻表现', '重磁异常'],
            '地球化学': ['元素', '成分', '含量', '化学', '微量元素', '稀土', '亏损', '富集', '高含量', '富集', '化学异常调查']
        }
        
        selected_categories = {cat: 0 for cat in category_rules.keys()}
        
        # 建立关系到类别的映射表
        relation_to_category = {}
        for category, keywords in category_rules.items():
            for keyword in keywords:
                relation_to_category[keyword] = category
        
        # 统计分类
        for _, row in selected_df.iterrows():
            relation = row['Relation']
            categorized = False
            
            # 精确匹配
            for keyword, category in relation_to_category.items():
                if keyword in relation:
                    selected_categories[category] += 1
                    categorized = True
                    break
            
            # 未匹配项的模糊匹配
            if not categorized:
                # 数值型关系处理
                if any(char.isdigit() for char in str(row['Object'])):
                    if 'm' in str(row['Object']) or '厚度' in relation:
                        selected_categories['地层'] += 1
                    elif '品位' in relation or '%' in str(row['Object']) or '高' in str(row['Object']):
                        selected_categories['成矿要素'] += 1
                    elif '磁异常' in relation or '电阻' in relation:
                        selected_categories['成矿要素'] += 1
                    elif '°' in str(row['Object']) or '气温' in relation:
                        selected_categories['地理'] += 1
                    else:
                        selected_categories['地球化学'] += 1
                # 特殊关系处理
                elif '主要矿产' in relation:
                    selected_categories['成矿要素'] += 1
                elif '岩性' in row['Subject'] or '岩性' in row['Object']:
                    selected_categories['地层'] += 1
                elif '岩体' in row['Subject'] or '岩体' in row['Object']:
                    selected_categories['岩体'] += 1
                # 其他通用关系
                elif relation in ['矿石矿物', '金属组合', '成矿潜力']:
                    selected_categories['成矿要素'] += 1
                elif relation in ['重磁异常', '磁性']:
                    selected_categories['地球物理'] += 1
                elif relation in ['轻稀土', '重稀土']:
                    selected_categories['地球化学'] += 1
                else:
                    # 统计未匹配三元组
                    selected_categories['未匹配'] = selected_categories.get('未匹配', 0) + 1
                    # print(f"未匹配三元组: {relation} -> {row['Object']}")
        
        return selected_categories

def main():
    """
    主函数，执行核心功能
    """
    kg_files = [
        'mpm/test/new/毛湾矿区钨矿总结知识库.csv',
        'mpm/test/new/川口矿田外围钨成矿规律知识库.csv'
    ]
    
    categorizer = TripletsCategorizer(kg_files)
    categorizer.load_knowledge_graph()
    
    # 执行分类统计
    categories = categorizer.categorize_triplets()
    total_triplets = len(categorizer.triplets)
    matched_triplets = total_triplets - categories.get('未匹配', 0)
    
    # 对选中的100个三元组进行分类统计
    selected_categories = categorizer.categorize_selected_triplets()
    total_selected = sum(selected_categories.values())
    matched_selected = total_selected - selected_categories.get('未匹配', 0)
    
    # 按照用户要求的格式输出统计表
    print("\n三元组分类统计表")
    print("=" * 80)
    print(f"{'序号':<4} {'三元组类型':<12} {'标注数量':<10} {'三元组总数':<12} {'描述'}")
    print("-" * 80)
    
    category_descriptions = {
        '地理': '描述区域、矿区等的地理位置、坐标范围等信息',
        '地层': '描述各地层单位的岩石特征、厚度、组成等',
        '岩体': '描述岩体类型、成分、结构等信息',
        '构造': '描述褶皱、断层、背斜等地质构造信息',
        '地质年代': '描述地质时代、期次等时间尺度信息',
        '成矿要素': '描述成矿系统、控矿因素、矿化特征等信息',
        '地球物理': '描述物探异常、物理场特征等信息',
        '地球化学': '描述化探异常、元素分布等信息'
    }
    
    # 按照固定顺序输出
    ordered_categories = ['地理', '地层', '岩体', '构造', '地质年代', '成矿要素', '地球物理', '地球化学']
    for i, category in enumerate(ordered_categories, 1):
        count = categories.get(category, 0)
        selected_count = selected_categories.get(category, 0)
        description = category_descriptions[category]
        print(f"{i:<4} {category:<12} {selected_count:<10} {count:<12} {description}")
    
    print("=" * 80)
    print(f"总三元组数: {total_triplets}")
    print(f"已分类三元组数: {matched_triplets}")
    print(f"未匹配三元组数: {categories.get('未匹配', 0)}")
    print(f"分类覆盖率: {matched_triplets/total_triplets*100:.2f}%")
    
    print(f"\n选中三元组数: {total_selected}")
    print(f"选中已分类三元组数: {matched_selected}")
    print(f"选中未匹配三元组数: {selected_categories.get('未匹配', 0)}")
    print(f"选中分类覆盖率: {matched_selected/total_selected*100:.2f}%")

if __name__ == "__main__":
    main()