from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# 1️⃣ 读取两个 CSV 文件
csv_files = ["kg_1.csv", "kg_2.csv"]
dfs = [pd.read_csv(file) for file in csv_files]

# 2️⃣ 合并知识图谱
df = pd.concat(dfs)
triples = df[['Subject', 'Rel', 'Object']].values

# 3️⃣ 训练知识嵌入模型
tf = TriplesFactory.from_labeled_triples(triples)
result = pipeline(
    model='TransE',
    training=tf,
    training_kwargs={'num_epochs': 10}
)

# 4️⃣ 计算三元组可信度得分
df["score"] = [result.model.predict_hrt(triple) for triple in triples]

# 5️⃣ 设定阈值，筛选低可信三元组
threshold = 0.2
low_confidence = df[df["score"] < threshold]

print("⚠️ 低可信度三元组:")
print(low_confidence)

# 6️⃣ 规则匹配（示例）
valid_genesis_types = {"热液型", "沉积型", "岩浆型"}
valid_minerals = {"金", "铜", "铅", "锌"}
invalid_environments = {"未知"}

def rule_based_check(triple):
    head, relation, tail = triple
    if relation == "成因类型" and tail not in valid_genesis_types:
        return False
    if relation == "矿物成分" and tail not in valid_minerals:
        return False
    if relation == "沉积环境" and tail in invalid_environments:
        return False
    return True

df["valid"] = df.apply(lambda row: rule_based_check((row["Subject"], row["Rel"], row["Object"])), axis=1)
invalid_triples = df[df["valid"] == False]

print("⚠️ 违反规则的三元组:")
print(invalid_triples)
