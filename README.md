## 代码说明

代码包括大语言模型的应用以及检索增强生成方案

摘要：（目的）当前地质找矿领域的大语言模型应用面临着专业知识不足、数据隐私安全和模型幻觉等问题，同时大语言模型在地质找矿领域应用中仍缺乏高效快捷的知识推荐手段。（方法）本研究提出了知识图谱与检索增强生成相结合的KG-RAG (Knowledge graph Retrieval-Augmented Generation) 框架，以大语言模型为工具，在地质本体约束下实现了地质找矿知识图谱的自动化抽取和结构化表达，同时利用知识图谱的多跳检索算法实现检索内容的深度与广度优化，实现了地质找矿智能知识问答模型。（结果）实验结果表明：KG-RAG在准确率、召回率和可信度(F1-score)上分别取得的0.807、0.833和0.819，在知识图谱构建任务相比大语言基模型GLM4-9B的直接知识抽取，分别取得了约50%、8%和29%的提升；在问答任务中，KG-RAG召回率和准确率上分别取得了0.917和0.88，相比文档向量检索增强生成方法分别取得了约24%和22%的提升。（结论）KG-RAG在知识图谱构建与智能问答两方面均表现出了较好的性能，能够有效从地质资料中进行地质找矿知识收集与表达，支持地质工作者的地质调查与找矿预测工作，本研究为大语言模型与知识图谱的联合应用提供了借鉴。

关键词：地质找矿；知识图谱；大语言模型；检索增强生成；地质本体；思维链

Abstract：Current applications of large language models (LLMs) in geological prospecting face challenges including insufficient domain expertise, data privacy concerns, and model hallucinations. Furthermore, there remains a lack of efficient and rapid knowledge recommendation methods for LLMs in this filed. This study proposes a KG-RAG (knowledge graph-embedded retrieval-augmented generation) framework that automates the extraction and structured representation of geological prospecting knowledge under the constraints of a geological ontology, leveraging large LLMs as tools. It further employs multi-hop retrieval algorithms within the knowledge graph to enhance the depth and breadth of retrieved content, thereby constructing an intelligent question-answering model for geological prospecting. Experimental results demonstrate that KG-RAG achieved scores of 0.807 (Precision), 0.833 (Recall), and 0.819 (F1-score) in knowledge graph construction tasks. Compared to direct knowledge extraction using the baseline LLM (GLM4-9B), KG-RAG delivers improvements of approximately 50% (Precision), 8% (Recall), and 29% (F1-score), respectively. In question-answering tasks, KG-RAG achieved 0.917 (Recall) and 0.88 (Precision), outperforming document vector-embedded retrieval-augmented generation methods by approximately 24% (Recall) and 22% (Precision), respectively. KG-RAG exhibits superior performance in both knowledge graph construction and intelligent question-answering. It effectively collects and represents geological prospecting and mineral exploration knowledge, providing a valuable reference to geologists for the combined application of LLMs and knowledge graphs.

Key words：Geological Prospecting；Knowledge Graph (KG); Large Language Models (LLMs); Retrieval Augmented Generation (RAG); Geological Ontology; Chain of Thought (CoT).

## KG-project

	该部分内容为前期LLM知识抽取以及数据处理脚本

## llm_local
	
	该脚本是大语言模型以及嵌入模型的本地化部署方案

## main_new_1
	
	该脚本是主界面的调用方案
	
## model_ds
	
	该脚本是LLM的输入和输出以及引导方案

## KnowledgeGraphRAG_1
	
	该脚本是基于图的RAG检索方案


## requirements
	
	该代码环境参照Geo3D的flask_glm_env容器，无需再创建，
    如果需要使用公网IP访问，建议添加frp工具。
	如果需要重新配置容器环境，可以参照requirements.txt

                                                                           
