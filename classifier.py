from typing import List, Dict, Optional
from long_term_memory import Experience

def count_op_keywords(text: str) -> int:
    keywords = ['and', 'then', 'next', 'generate', 'save', 'translate', 'analyze', 'report',
                '并', '然后', '接着', '生成', '保存', '翻译', '分析', '报告']
    return sum(text.lower().count(k) for k in keywords)

def complexity_score(subtask_num: int,
                     query_len: int,
                     tool_num: int,
                     cross_class_num: int,
                     api_num: int,
                     experience_hit: bool,
                     traj_steps: int,
                     reuse_count: int,
                     op_keywords_count: int,
                     exp_api: int) -> float:
    score = 0
    score += subtask_num * 1.5
    score += (query_len / 10) * 0.5
    score += tool_num * 0.8
    score += cross_class_num * 1.0
    score += api_num * 0.5
    if experience_hit and traj_steps > 0:
        score += traj_steps * 0.5
    if experience_hit and reuse_count > 0:
        score -= reuse_count * 0.1
    if experience_hit and exp_api > 0:
        score += exp_api * 0.5
    score += op_keywords_count * 1.2
    return max(score, 0)

def select_strategy(query: str,
                    subtasks: List[str],
                    api_pool: List[Dict],
                    experience: Optional[Experience]) -> str:
    
    subtask_num = len(subtasks)
    query_len = len(query.split())
    tool_names = set(t['tool_name'] for t in api_pool)
    tool_num = len(tool_names)
    category_names = set(t['category_name'] for t in api_pool)
    cross_class_num = len(category_names)
    api_num = len(api_pool)
    op_keywords_count = count_op_keywords(query)

    if experience is not None:
        experience_hit = True
    else:
        experience_hit = False
    traj_steps = len(experience['trajectory'].tool_order)-1 if experience_hit and experience.get('trajectory') else 0
    reuse_count = experience.get("reuse_count") if experience_hit and experience.get("reuse_count") else 0
    tool_details = experience.get("tool_details") if experience_hit and experience.get("tool_details") else []
    exp_api = len(tool_details)
        

    score = complexity_score(subtask_num, query_len, tool_num, cross_class_num, api_num,
                             experience_hit, traj_steps, reuse_count, op_keywords_count, exp_api)
    
    print("Score:", score)

    if score < 8:
        return 'react'
    elif score < 15:
        return 'dfs'
    else:
        return 'lats'

