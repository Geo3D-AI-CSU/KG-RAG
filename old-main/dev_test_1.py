#丰度实验测试
import pandas as pd
import networkx as nx

# 1️⃣ 读取两个 CSV 文件
csv_files = ["onto/川口矿田外围钨成矿规律知识库.csv", "onto/new_db.csv"]  # 请替换为你的文件名
dfs = [pd.read_csv(file) for file in csv_files]

# 2️⃣ 合并知识图谱
df = pd.concat(dfs)

# 3️⃣ 构建图
G = nx.DiGraph()  # 有向图
for _, row in df.iterrows():
    G.add_edge(row["Subject"], row["Object"], relation=row["Rel"])

# 4️⃣ 计算平均度数 aveD
degrees = [deg for _, deg in G.degree()]
average_degree = sum(degrees) / len(degrees) if degrees else 0

print(f"📊 知识图谱的平均度数 (aveD): {average_degree:.2f}")

# 5️⃣ 可选：输出度数分布
degree_distribution = pd.Series(degrees).value_counts().sort_index()
print("📈 度数分布:")
print(degree_distribution)
