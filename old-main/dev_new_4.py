from dev_new_3 import MPM_model
import json
import re
#加载全局变量
model_config_file = 'config/mpm_model_config_test.json'
rag_config_file = 'config/mpm_rag_config.json'
with open(model_config_file, 'r') as f:
    model_config = json.load(f)
with open(rag_config_file, 'r') as f:
    rag_config = json.load(f)
model = MPM_model(model_config, rag_config)
db=model.rag_init()


input='请告诉我毛湾矿区存在的与地层相关的信息。'
data={"message": input}
message=data.get("message")
print(message)
model.query=[message]
input_vec, meta_vec = model.prompt_init(db)
result = model.mpm_main(input_vec, meta_vec)
print(result)

