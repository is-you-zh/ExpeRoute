import json
import re
from openai import OpenAI
from config import *
import toolenv
import logging
from arguments import parse_args
from utils import *
from long_term_memory import add_experience_to_db, ExecutionStrategy, Metadata, ToolDetail
import time
from models import call_gpt

client = OpenAI(api_key=api_key, base_url=base_url)


def build_prompt(api_docs_text, scratchpad, user_input):
    prompt = f"""You are a helpful assistant. You can use the following tools:

{api_docs_text}

Use the following format:

Question: the input question you must answer
Thought: think about what to do next
Action: the action to take, should be a tool name
Action Input: a JSON object of tool arguments
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Action: Finish
Action Input: {{
  "return_type": "give_answer",
  "final_answer": "..."
}}

Important:
- When you think you have finished after calling a tool, please call the 'Finish' tool in the next round.
- Only ONE tool call per turn, **`Finish` also is a tool, it cannot be called simultaneously with other tools**.
- You MUST call `Finish` tool to end, using one of three `return_type` values:
  - "give_answer": you've completed the task, return answer
  - "give_up_and_restart": current tools failed, restart
  - "give_up": task cannot be solved
- Do NOT end with free text. Always use Finish.
Reminder to ALWAYS respond with a single action. Use tools if necessary. Respond directly if appropriate. 
Begin!

Question: {user_input}
{scratchpad}"""
    return [{"role": "user", "content": prompt.strip()}]


def add_experience(experience, scratchpad, query_id, env, api_list, query):
    tool_order, used_api = get_action_trajectory_from_text(scratchpad, query)
    experience.execution_strategy = ExecutionStrategy(
        type="sequential",
        merge_logic="merge",
        tool_order=tool_order
    )
    timestamp = time.time()
    readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    experience.metadata = Metadata(
        created_by="admin",
        created_at=str(readable_time),
        reuse_count=0,
        source_task_id=query_id
    )
    selected_api_list = []
    for item in used_api:
        if item in env.functions_names:
            idx = env.functions_names.index(item)
            selected_api_list.append(api_list[idx])
    experience.tool_details = [ToolDetail(**d) for d in selected_api_list]
    print(experience)
    add_experience_to_db(experience)
    logger.info(f"[INFO] The experience was added successfully: {experience}")


def get_action_trajectory_from_text(text, query):
    trajectory = []
    func_names = []

    trajectory.append(f"User Query: {query}")
    pattern = re.compile(
        r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(\{.*?\})",
        re.DOTALL
    )

    for i, match in enumerate(pattern.finditer(text)):
        func_name = match.group(1).strip()
        raw_input = match.group(2).strip()

        try:
            parsed_input = json.loads(raw_input)
        except json.JSONDecodeError:
            try:
                fixed = raw_input.replace("'", '"').replace("True", "true").replace("False", "false")
                parsed_input = json.loads(fixed)
            except:
                parsed_input = raw_input  

        action_str = f"{func_name}({json.dumps(parsed_input, ensure_ascii=False)})"
        trajectory.append(f"Action {i+1}: {action_str}")
        func_names.append(func_name)
    return trajectory, func_names


def react_search(args, query_data, api_list, tool_des, i, experience=None):
    query = query_data['query']
    env = toolenv.ToolEnv(query=query, api_list=api_list, tool_des=tool_des, process_id=i)
    global logger
    logger = logging.getLogger(f'task_{query_data["query_id"]}')
    print(f"Now playing {query}, with {len(api_list)} APIs")
    logger.info(f"Now playing {query}, with {len(api_list)} APIs")
    solve_tokens = 0
    scratchpad = ""

    messages = build_prompt(env.functions, scratchpad, query)
    logger.info(f"Prompt: {messages}")
    for turn in range(10):
        print(f"\n=== the {turn + 1}-th turn ===")
        logger.info(f"the {turn + 1}-th turn")
        messages = build_prompt(env.functions, scratchpad, query)
        # response = call_gpt(model=model_name, messages=messages)
        response = call_gpt(model="gpt-4-0125-preview", messages=messages)
        solve_tokens += response.usage.total_tokens
        # response = client.chat.completions.create(model=model_name, messages=messages, temperature=0)
        output = response.choices[0].message.content.strip()
        print("🔁 LLM 输出:\n", output)
        logger.info(f"output: {output}")

        if "Final Answer:" in output:
            break

        action_match = re.search(r"Action:\s*(\w+)", output)
        input_match = re.search(r"Action Input:\s*({.*})", output, re.DOTALL)

        if not action_match or not input_match:
            print("❌ 无法解析 Action 或 Action Input，结束。")
            logger.info("无法解析 Action 或 Action Input，结束。")
            break

        action = action_match.group(1).strip()
        try:
            action_input = json.loads(input_match.group(1).strip())
        except Exception as e:
            print("❌ JSON 解析失败:", e)
            logger.info(f"JSON 解析失败: {e}")
            break
        if action == "Finish":
            print("===========================================")
            scratchpad += f"{output}"
            return_type = action_input.get("return_type")
            if return_type == "give_answer":
                print("🎯 最终回答:", action_input.get("final_answer", "（无内容）"))
                logger.info(f"Final Answer: {action_input.get('final_answer', '（无内容）')}")
                if(experience is not None):
                    add_experience(experience, scratchpad, query_data['query_id'], env, api_list, query)
            elif return_type == "give_up_and_restart":
                print("⚠️ 放弃并重启：", action_input.get("reason", "（无说明）"))
                logger.info(f"⚠️ 放弃并重启： {action_input.get('reason', '（无说明）')}")
            elif return_type == "give_up":
                print("❌ 放弃任务：", action_input.get("reason", "（无说明）"))
                logger.info(f"❌ 放弃任务： {action_input.get('reason', '（无说明）')}")
            else:
                print("❓ 未知的 return_type:", return_type)
                logger.info(f"❓ 未知的 return_type: {return_type}")
            break
        try:
            result = env.step(action, action_input)
            print(f"🔧 执行工具 {action} 成功，结果: {result}")
            logger.info(f"执行工具 {action} 成功，结果: {result}")
        except Exception as e:
            print("⚠️ 工具调用失败:", e)
            logger.info(f"工具调用失败: {e}")
            break

        scratchpad += f"{output}\nObservation: {result}\n"
    with open("/root/autodl-tmp/ToolCombine/ToolsTree0307/result/memorybench/toolcombinebench.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"step numbers:\t{turn+1}\n")
    return output, solve_tokens


if __name__ == "__main__":
    args = parse_args()
    with open(args.query_path, 'r', encoding='utf-8') as f:
        query_data_all = json.load(f)
    query_data = query_data_all[25]
    api_list = [{'category_name': 'eCommerce', 'tool_name': 'Amazon Pricing and Product Info', 'api_name': 'Main Endpoint'}, {'category_name': 'Financial', 'tool_name': 'Currency Converter_v2', 'api_name': 'Convert'}]
    tools_des, to_prune = get_tools_des(api_list)
    output = react_search(args, query_data, api_list, tools_des, 25, experience=None)
    print("===========================================")
    print(output)