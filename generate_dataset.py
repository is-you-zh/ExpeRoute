import random
import json
from models import gpt_for_data
from prompt import *
import os
import toolenv
from utils import get_tools_des
from tqdm import tqdm
from termcolor import colored
from prompt import CHECK_API_PROMPT
from config import model_name

# 参数配置，True = 用大模型动态生成场景, False = 从静态库中抽取
USE_DYNAMIC = False  
TOOL_COUNT_ONE_CATEGORY = 20
TOOL_COUNT_MULIT_CATEGORY = 10  
API_COUNT_ONE_CATEGORY = 10
API_COUNT_MUTIL_CATEGORY = 5  

with open('data/sence.json', 'r', encoding='utf-8') as f:
    CATEGORY_COMBINATION_LIBRARY = json.load(f)

with open('data/tool_data.json', 'r', encoding='utf-8') as f:
    data_tool = json.load(f)

with open('data/api_details.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

CATEGORIES = ['eCommerce', 'Music', 'Mapping', 'Science', 'Search', 'Email', 'Health_and_Fitness', 'Business_Software', 'Travel', 'Media', 'Data', 'Location', 'Communication', 'Financial', 'Transportation', 'Commerce', 'Weather', 'Artificial_Intelligence_Machine_Learning', 'Other', 'Video_Images', 'Sports', 'Energy', 'Movies', 'Business', 'Entertainment', 'Jobs', 'Visual_Recognition', 'Translation', 'Advertising', 'Finance', 'Database', 'News_Media', 'Events', 'Food', 'Tools', 'Medical', 'Social', 'Gaming', 'Logistics', 'SMS', 'Education', 'Text_Analysis']

def generate_scene_dynamically():
    """
    Generate dynamic scenes and select category combinations using large language models.
    """
    prompt = (
        "You are an assistant tasked with selecting **two categories** from the list below that can be combined and are complementary to each other. "
        "These category pairs should typically be used together in real-world scenarios or jointly solve a certain need. "
        "Focus only on the relationships between the categories — you do not need to construct specific use cases or scenarios. "
        "Output the result in JSON format, for example:\n"
        "{ \"selected_categories\": [\"Category1\", \"Category2\"],\n"
        "\"reason\": \"Your reasoning here\" }\n"
        f"Available categories: {CATEGORIES}"
    )
    response = gpt_for_data(prompt,model=model_name)
    return response.choices[0].message.content

def truncate(text, max_length=200):
    """
    Truncate the text to the specified length. If it exceeds, truncate it and add an ellipsis.
    """
    return text if len(text) <= max_length else text[:max_length] + "..."

def clean_and_load_json(data_str_or_dict):
    """
    If the input is a string with triple quotes, automatically remove the quotes and parse it into a dict; If the input itself is a dict, return it directly.
    """
    if isinstance(data_str_or_dict, dict):
        return data_str_or_dict
    
    if isinstance(data_str_or_dict, str):
        cleaned = data_str_or_dict.strip()
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.splitlines()[1:-1])
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.splitlines()[1:-1])
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("Unable to parse JSON:", e)
            return None
    else:
        print("The input is neither a dict nor a str.")
        return None

def build_dataset_entry(result, selected_categories, tool_details_with_apis):
    """
    Encapsulate dataset_entry and add description information to apis.
    param result: The result returned by gpt (including query, apis)
    param selected_categories: The list of currently selected categories
    param tool_details_with_apis: The brief information of the currently selected tools for saving (excluding the api list)
    param data: The original data structure, including api descriptions, etc.
    return: The complete dataset_entry dictionary
    """

    apis_with_description = []

    for api_entry in result.get("apis"):
        category = api_entry['category']
        tool = api_entry['tool']
        api_name = api_entry['api']

        api_list = data[category][tool]['api_list']
        matching_api = next((api for api in api_list if api['name'] == api_name), None)

        if matching_api:
            api_description = matching_api['description']
        else:
            api_description = ""  

        enriched_api_entry = {
            "category": api_entry['category'],
            "tool": api_entry['tool'],
            "api": api_entry['api'],
            "api_description": api_description,
            "parameters": api_entry['parameters']
        }
        apis_with_description.append(enriched_api_entry)

    dataset_entry = {
        "query": result.get("query"),
        "query_id": None,
        "apis": apis_with_description,  
        "categories": selected_categories,
        "selected_apis": tool_details_with_apis
    }

    return dataset_entry

def check_api(api_list, result) -> list:
    """
    Detect whether the API is available. If it is not available, record it in the file.
    """

    status_code_map = {
        0: "normal_response",
        1: "no_api_name",
        2: "input_error",
        3: "final_answer",
        4: "pruning",
        5: "timeout",
        6: "404_not_found",
        7: "not_subscribed",
        8: "unauthorized",
        9: "too_many_requests",
        10: "rate_limit",
        11: "error_field_in_message",
        12: "send_request_error"
    }

    status_list = []
    tools_des, _ = get_tools_des(api_list)
    env = toolenv.ToolEnv(query=result.get('query'), api_list=api_list, tool_des=tools_des)
    failed_apis = []
    data_dict = env.data_dict
    if (len(data_dict.get("api_list"))<len(api_list)):
        print(colored("There is a non-existent API. Start over.",'cyan'))
        return[]
         
    print(colored("Checking api",'cyan'))
    if not data_dict.get("api_list"):
        print("⚠️ No APIs found in data_dict['api_list']. Skipping check.")
        return status_list

    for i,api_json in enumerate(data_dict["api_list"]): 
        standard_tool_name = tools_des[i][0]
        openai_function_json,_, _ = env.api_json_to_openai_json(api_json,standard_tool_name)
        action_name = openai_function_json['name']
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                return status_list
        action_param = result.get('apis')[i].get('parameters')
        try:
            obs, status_code, _, _, _ = env.step(action_name, action_param)
            if(status_code != 0):
                api_list[i]['status_code'] = status_code
                failed_apis.append(api_list[i])
                return status_list
            if(status_code == 0):
                prompt = CHECK_API_PROMPT.replace('{action_name}', str(action_name)).replace('{action_param}', str(action_param)).replace('{obs}', str(obs))

                response = gpt_for_data(prompt,model=model_name)
                result_of_res = response.choices[0].message.content
                if result_of_res.strip().lower() == 'real_response':
                    print(colored(f"{api_list[i]} is usable.", 'green'))
                    status_desc = status_code_map.get(status_code, f"unknown_status_{status_code}")
                    status_list.append(status_desc)
                elif result_of_res.strip().lower() == 'error_message':
                    status_code = 2
                    api_list[i]['status_code'] = status_code
                    failed_apis.append(api_list[i])
                    print(colored(f"{api_list[i]} has error message.", 'red'))
                    status_desc = status_code_map.get(status_code, f"unknown_status_{status_code}")
                    status_list.append(status_desc)
                    record_failed_api(failed_apis)
                    return status_list
                else:
                    status_code = 11
                    failed_apis.append(api_list[i])
                    api_list[i]['status_code'] = status_code
                    print(colored(f"{api_list[i]} is not usable.", 'red'))
                    status_desc = status_code_map.get(status_code, f"unknown_status_{status_code}")
                    status_list.append(status_desc)
                    record_failed_api(failed_apis)
                    return status_list
        except Exception as e:
            print(f"An unknown error occurred: {e}")
            status_list.append("unknown_error")

    return status_list

def record_failed_api(failed_apis):
    if failed_apis:
        failed_file = "data/failed_apis.json"

        if os.path.exists(failed_file):
            with open(failed_file, "r", encoding="utf-8") as f:
                try:
                    existing_failed = json.load(f)
                    if not isinstance(existing_failed, list):
                        print("⚠️ The format of 'failed_apis.json' is incorrect, reset it to an empty list.")
                        existing_failed = []
                except json.JSONDecodeError:
                    print("⚠️ The 'failed_apis.json' file is corrupted or empty. Reset it to an empty list.")
                    existing_failed = []
        else:
            existing_failed = []

        existing_failed.extend(failed_apis)
        with open(failed_file, "w", encoding="utf-8") as f:
            json.dump(existing_failed, f, indent=2, ensure_ascii=False)

        print("\n", f"❌ {len(failed_apis)} failed APIs have been saved. Total failed records: {len(existing_failed)}")

    else:
        print("✅ There is no failed API.")

def append_abnormal_query(query_data, json_path="data/abnormal_queries.json"):
    """
    Check if any of the current data has an api_status that is not normal_response. If so, consider this query to be abnormal and add it to the list of abnormal queries.
    """
    has_abnormal = any(
        api.get("api_status") != "normal_response"
        for api in query_data.get("apis", [])
    )
    if not has_abnormal:
        return  

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    query_id = query_data.get("query_id")
    if query_id and query_id not in existing_data:
        existing_data.append(query_id)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        print(f"The abnormal query_id has been appended:{query_id}")
    else:
        print("The query_id already exists or is empty, no need to append.")

def topn_apis_by_similarity(tool_desc, apis, topn=5):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tool_emb = model.encode(tool_desc, convert_to_tensor=True)
    api_texts = [f"{api['name']}: {api.get('description', '')}" for api in apis]
    api_embs = model.encode(api_texts, convert_to_tensor=True)

    similarities = util.cos_sim(tool_emb, api_embs)[0]  # shape: [num_apis]
    top_indices = similarities.topk(k=min(topn, len(apis))).indices.tolist()

    return [apis[i] for i in top_indices]

def build_single_tool_dataset():
    """
    Build single-tool dataset items:
    - Use only one tool each time (judged by the large language model for feasibility and availability).
    - If the tool is unavailable, the large language model returns "None" and it is automatically skipped.
    - Select only 2 - 5 of these APIs and generate reasonable natural language queries.
    - The output structure is exactly the same as that of build_dataset.
    """

    while True:
        selected_category = random.choice(CATEGORIES)
        tools = list(data_tool[selected_category].keys())
        selected_tool = random.choice(tools)

        tool_description = truncate(data[selected_category][selected_tool]["description"], 500)
        api_list_names = data_tool[selected_category][selected_tool]["api_list_names"]
        api_full_list = data[selected_category][selected_tool]["api_list"]
        if len(api_full_list) > 20:
            continue

        api_entries = []
        for api in api_full_list:
            api_entry = {
                "api_name": api["name"],
                "api_description": truncate(api["description"])
            }
            if api["required_parameters"]:
                api_entry["required_parameters"] = api["required_parameters"]
            api_entries.append(api_entry)

        api_prompt_input = {
            "category": selected_category,
            "tool": selected_tool,  
            "tool_description": tool_description,
            "apis": api_entries  
        }
        prompt = GENERATE_SINGLE_TOOL_PROMPT_VARIANTS.replace("{USER_INPUT}", json.dumps(api_prompt_input, ensure_ascii=False, indent=2))

        result = gpt_for_data(prompt,model=model_name).choices[0].message.content
        print("\n", "🤖 GPT Response:\n", result, "\n")
        if result.strip().lower() == "none":
            print("❌ The tools are unrealistic or not applicable, skip...")
            continue
        try:
            result = clean_and_load_json(result)
            if not isinstance(result, dict) or "queries" not in result or "apis" not in result:
                print("❌ Result malformed, skipping")
                continue
            if isinstance(result, str):
                result = json.loads(result)
        except Exception as e:
            print(f"⚠️ JSON parsing failed or fields are missing, skip. Error: {e}")
            continue

        tool_details_with_apis = [{
            "category": selected_category,
            "tools": [{
                "tool_name": selected_tool,
                "tool_description": tool_description,
                "apis": api_list_names
            }]
        }]

        api_entries = []
        try:
            for api_item in result["apis"]:
                selected_api_name = api_item["api"]
                selected_api_obj = next(
                    (api for api in api_full_list if api["name"] == selected_api_name),
                    None
                )
                if selected_api_obj is None:
                    raise ValueError(f"API information not found: {selected_api_name}")

                entry = {
                    "api_name": selected_api_obj["name"],
                    "api_description": truncate(selected_api_obj.get("description", "")),
                }
                if selected_api_obj.get("required_parameters"):
                    entry["parameters"] = selected_api_obj["required_parameters"]
                api_entries.append(entry)
        except Exception as e:
            print(f"❌ Failed to build API information, skipping the current tool. Error: {e}")
            continue  

        if not api_entries:
            print("⚠️ No available API, skip")
            continue

        dataset_entry = build_dataset_entry(result, [selected_category], tool_details_with_apis)

        api_to_check = [
            {
                'category_name': item['category'],
                'tool_name': item['tool'],
                'api_name': item['api']
            }
            for item in result.get('apis')
        ]
        status_list = check_api(api_to_check, result)
        if not status_list:
            continue
        for i, api_entry in enumerate(dataset_entry["apis"]):
            if i < len(status_list):
                api_entry["api_status"] = status_list[i]
            else:
                api_entry["api_status"] = None 

        if any(status != 'normal_response' for status in status_list):
            print("⚠️ At least one API is not normal_response. Skipping this query group.")
            continue
        
        dataset_entries = []
        for i, query in enumerate(result["queries"]):
            query_content = {
                "query": query,
                "apis": result["apis"]
            }
            entry = build_dataset_entry(query_content, [selected_category], tool_details_with_apis)

            for j, api_entry in enumerate(entry["apis"]):
                api_entry["api_status"] = status_list[j] if j < len(status_list) else None

            dataset_entries.append(entry)
        
        return dataset_entries

def generate_batch_for_single_tool(num_entries):
    """
    Batch generate multiple "single-tool" query entries (3 entries each time) and save them to a file.
    Functional description:
    - Each time build_single_tool_dataset() is called, generate a list containing 3 queries;
    - Automatically generate an incrementing query_id (starting from 0000);
    - Write each query one by one into single_dataset.json;
    - Record abnormal queries into abnormal_queries.json;
    - Report the total number, the number of successes, and the number of failures.
    Parameters: num_entries (int): The number of "query groups" to generate (3 queries per group).
    """

    file_path = "./query_dataset/single_tool.json"

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    print("❌ Existing file is not a list. Replacing with a new list.")
                    existing_data = []
            except json.JSONDecodeError:
                print("⚠️ File is empty or corrupted. Starting a new list.")
                existing_data = []
    else:
        existing_data = []

    error_count = 0
    total_added = 0

    for i in tqdm(range(num_entries), desc="Generating 3-query sets for single-tool"):
        try:
            entries = build_single_tool_dataset()  
            if not isinstance(entries, list) or len(entries) != 3:
                raise ValueError("Expected list of 3 entries, got something else.")
            for entry in entries:
                new_id = len(existing_data) + 1
                new_id_str = f"{new_id:04d}"
                entry["query_id"] = new_id_str

                existing_data.append(entry)
                # append_abnormal_query(entry)
            total_added += 1

        except Exception as e:
            error_count += 1
            print(f"⚠️ Failed to generate group #{i + 1}. Skipping. Error: {e}")
            continue

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print("✅ Batch generation completed.")
    print(f"🟢 Successfully generated: {total_added} groups")
    print(f"🔴 Failed (skipped groups): {error_count}")
    print(f"📦 Total entries in file now: {len(existing_data)}")

def build_single_category_dataset():
    """
    Build data items for "multi - tool collaboration under the same category".
    - Select multiple tools (2 - 3) from one category;
    - Select a suitable API for each tool;
    - Construct a natural - language query for a unified scenario, requiring collaboration among multiple tools;
    - Return a dataset_entry that conforms to the standard structure.
    """
    while True:
        selected_category = random.choice(CATEGORIES)
        category = selected_category

        tools = list(data_tool.get(category, {}).keys())
        if not tools:
            print(f"⚠️ No tools found for category: {category}")
            continue

        selected_tools = random.sample(tools, min(TOOL_COUNT_ONE_CATEGORY, len(tools)))
        tool_details = []
        tool_details_with_apis = []
        tool_entries_for_prompt = []
        tool_entries_full = []

        filtered_tools = []
        for tool in selected_tools:
            tool_data_tool = data_tool.get(category, {}).get(tool)
            tool_data_desc = data.get(category, {}).get(tool)

            if not tool_data_tool or not tool_data_desc:
                print(f"⚠️ Skipping '{tool}' due to missing data in category '{category}'")
                continue

            api_list = tool_data_tool.get("api_list_names", [])
            if not isinstance(api_list, list):
                print(f"⚠️ Invalid API list format for tool: {tool}")
                continue

            if len(api_list) > 15:
                print(f"⏩ Skipping tool '{tool}' because it has too many APIs: {len(api_list)}")
                continue

            filtered_tools.append(tool)

            description = truncate(tool_data_desc.get("description", "No description"))

            tool_entry_for_prompt = {
                "tool_name": tool,
                "tool_description": description
            }
            tool_entries_for_prompt.append(tool_entry_for_prompt)

            tool_entry_full = {
                "tool_name": tool,
                "tool_description": description,
                "apis": api_list
            }
            tool_entries_full.append(tool_entry_full)

        selected_tools = filtered_tools

        if not tool_entries_for_prompt:
            print(f"⚠️ All tools in category '{category}' were skipped. Retrying...")
            continue

        category_entry_for_prompt = {
            "category": category,
            "tools": tool_entries_for_prompt
        }
        tool_details.append(category_entry_for_prompt)

        category_entry_full = {
            "category": category,
            "tools": tool_entries_full
        }
        tool_details_with_apis.append(category_entry_full)

        prompt = GENERATE_COMBINATION_SINGLE_CATEGORY_PROMPT.replace("{USER_INPUT}", str(tool_details))
        response = gpt_for_data(prompt,model=model_name)
        result = response.choices[0].message.content    
        print("\n", "🤖 GPT Tool Response:\n", result, "\n")
        if result.strip().lower() == "none":
            print("❌ The tool combination is unreasonable, skip...")
            continue
        try:
            result = clean_and_load_json(result)
            if isinstance(result, str):
                result = json.loads(result)
        except Exception as e:
            print(f"⚠️ JSON parsing failed, skipped. Error: {e}")
            continue
        selected_tools = result.get("selected_tools")

        api_details = []
        for item in selected_tools[:]:
            category = category
            tool = item['tool_name']

            try:
                apis = data[category][tool]['api_list']
            except KeyError:
                selected_tools.remove(item)
                print(colored(f"{tool} is not found","red"))
                with open("./data/invalid_tool_name.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(f"{category} - {tool}\n")
                continue  
            # 注意：这里可以决定是否全部带上，或者用相似度筛topN，或用 random.sample(apis, N)
            selected_apis = random.sample(apis, min(len(apis), API_COUNT_ONE_CATEGORY))
            # tool_description = truncate(data[category][tool]["description"])
            # selected_apis = topn_apis_by_similarity(tool_description, apis, topn=2)

            api_entries = []
            for api in selected_apis:
                api_entry = {
                    "api_name": api["name"],
                    "api_description": truncate(api["description"])
                }
                if api["required_parameters"]:
                    api_entry["required_parameters"] = api["required_parameters"]
                api_entries.append(api_entry)

            tool_entry = {
                "tool_name": tool,
                "tool_description": truncate(data[category][tool]["description"]),
                "apis": api_entries
            }
            api_details.append({
                "category": category,
                "tool": tool_entry
            })

        prompt = GENERATE_SINGLE_CATEGORY_PROMPT_VARIANTS.replace("{USER_INPUT}", str(api_details))
        response = gpt_for_data(prompt,model=model_name)
        result = response.choices[0].message.content
        print("\n", "🤖 GPT Query Response:\n", result, "\n")

        try:
            if not result:
                print("result is none")
                continue
            else:
                result = clean_and_load_json(result)
                if isinstance(result, str):
                    result = json.loads(result)
        except Exception as e:
            print(f"⚠️ JSON parsing failed, skipped. Error: {e}")
            continue

        dataset_entry = build_dataset_entry(result, selected_category, tool_details_with_apis)
        api_to_check = [
            {
                'category_name': item['category'],
                'tool_name': item['tool'],
                'api_name': item['api']
            }
            for item in result.get('apis')
        ]

        status_list = check_api(api_to_check, result)
        if not status_list:
            continue
        for i, api_entry in enumerate(dataset_entry["apis"]):
            if i < len(status_list):
                api_entry["api_status"] = status_list[i]
            else:
                api_entry["api_status"] = None 

        if any(status != 'normal_response' for status in status_list):
            print("⚠️ At least one API is not normal_response. Skipping this query group.")
            continue
        
        dataset_entries = []
        for i, query in enumerate(result["queries"]):
            query_content = {
                "query": query,
                "apis": result["apis"]
            }
            entry = build_dataset_entry(query_content, [selected_category], tool_details_with_apis)

            for j, api_entry in enumerate(entry["apis"]):
                api_entry["api_status"] = status_list[j] if j < len(status_list) else None

            dataset_entries.append(entry)

        return dataset_entries

def generate_batch_for_single_category(num_entries):
    """
    Batch generate multiple "multiple tools within a class" query entries and save them to a file.
    Function description:
    - Each time build_single_category_dataset() is called, a set of 3 variants is generated.
    - Automatically generate an incrementing query_id (starting from 10000).
    - The results are accumulated and saved to the ./query_dataset/single_category_dataset.json file.
    - If a query is identified as abnormal, it is saved to abnormal_queries.json.
    - Report the number of successful/failed entries and the total number.
    Parameters:
        num_entries (int): The number of "sets" to generate (each set contains 3 queries).
    """

    file_path = "./query_dataset/single_category.json"

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    print("❌ Existing file is not a list. Replacing with a new list.")
                    existing_data = []
            except json.JSONDecodeError:
                print("⚠️ File is empty or corrupted. Starting a new list.")
                existing_data = []
    else:
        existing_data = []

    error_count = 0
    total_generated = 0

    for i in tqdm(range(num_entries), desc="Generating single-category (multi-tool) queries"):
        
        try:
            entries = build_single_category_dataset()  
            if not entries or not isinstance(entries, list):
                raise ValueError("Returned entries is None or not a list")

            for entry in entries:
                new_id = 1001 + len(existing_data)
                entry["query_id"] = f"{new_id:04d}"
                existing_data.append(entry)
                # append_abnormal_query(entry)
            total_generated += 1

        except Exception as e:
            error_count += 1
            print(f"⚠️ Failed to generate group #{i + 1}. Skipping. Error: {e}")
            continue

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print("✅ Batch generation completed.")
    print(f"🟢 Successfully generated: {total_generated} groups")
    print(f"🔴 Failed groups (skipped): {error_count}")
    print(f"📦 Total entries in file now: {len(existing_data)}")

def build_mutil_category_dataset():
    """
    Build a multi-class and multi-tool dataset.
    """

    while True:
        if USE_DYNAMIC:
            selected_categories = generate_scene_dynamically()
        else:
            selected_item = random.choice(CATEGORY_COMBINATION_LIBRARY)
            selected_categories = selected_item.get("categories")
        tool_details = []  
        tool_details_with_apis = [] 

        for category in selected_categories:
            tools = list(data_tool.get(category).keys())
            selected_tools = random.sample(tools, min(TOOL_COUNT_MULIT_CATEGORY, len(tools)))

            tool_entries_for_prompt = []
            tool_entries_full = []

            for tool in selected_tools:
                tool_entry_for_prompt = {
                    "tool_name": tool,
                    "tool_description": truncate(data[category][tool]["description"])
                }
                tool_entries_for_prompt.append(tool_entry_for_prompt)

                api_list = data_tool[category][tool]["api_list_names"]

                tool_entry_full = {
                    "tool_name": tool,
                    "tool_description": truncate(data[category][tool]["description"]),
                    "apis": api_list
                }
                tool_entries_full.append(tool_entry_full)

            category_entry_for_prompt = {
                "category": category,
                "tools": tool_entries_for_prompt
            }
            tool_details.append(category_entry_for_prompt)

            category_entry_full = {
                "category": category,
                "tools": tool_entries_full
            }
            tool_details_with_apis.append(category_entry_full)

        prompt = GENERATE_COMBINATION_MULTI_CATEGORY_PROMPT.replace("{USER_INPUT}", str(tool_details))
        response = gpt_for_data(prompt,model=model_name)
        result = response.choices[0].message.content
        print("\n", "🤖 GPT Response:\n", result, "\n")
        result = clean_and_load_json(result)
        
        if isinstance(result, str):
            result = json.loads(result)
        selected_tools = result.get("selected_tools")

        api_details = []
        for item in selected_tools:
            category = item['category']
            tool = item['tool']
            try:
                apis = data[category][tool]['api_list']
            except KeyError:
                selected_tools.remove(item)
                print(colored(f"{tool} is not found","red"))
                with open("./data/invalid_tool_name.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(f"{category} - {tool}\n")
                continue  

            # 注意：这里可以决定是否全部带上，或者用相似度筛topN，或用 random.sample(apis, N)
            selected_apis = random.sample(apis, min(len(apis), API_COUNT_MUTIL_CATEGORY))

            api_entries = []
            for api in selected_apis:
                api_entry = {
                    "api_name": api["name"],
                    "api_description": truncate(api["description"])
                }
                if api["required_parameters"]:
                    api_entry["required_parameters"] = api["required_parameters"]
                api_entries.append(api_entry)

            tool_entry = {
                "tool_name": tool,
                "tool_description": truncate(data[category][tool]["description"]),
                "apis": api_entries
            }
            api_details.append({
                "category": category,
                "tool": tool_entry
            })

        prompt = GENERATE_MUTIL_CATEGORY_PROMPT_VARIANTS.replace("{USER_INPUT}", str(api_details))
        response = gpt_for_data(prompt,model=model_name)
        result = response.choices[0].message.content
        print("\n", "🤖 GPT Query Response:\n", result, "\n")

        try:
            if not result:
                print("result is none")
                continue
            else:
                result = clean_and_load_json(result)
                if isinstance(result, str):
                    result = json.loads(result)
        except Exception as e:
            print(f"⚠️ JSON parsing failed, skipped. Error: {e}")
            continue

        dataset_entry = build_dataset_entry(result, selected_categories, tool_details_with_apis)
        api_to_check = [
            {
                'category_name': item['category'],
                'tool_name': item['tool'],
                'api_name': item['api']
            }
            for item in result.get('apis')
        ]
        status_list = check_api(api_to_check, result)
        if not status_list:
            continue
        for i, api_entry in enumerate(dataset_entry["apis"]):
            if i < len(status_list):
                api_entry["api_status"] = status_list[i]
            else:
                api_entry["api_status"] = None 

        if any(status != 'normal_response' for status in status_list):
            print("⚠️ At least one API is not normal_response. Skipping this query group.")
            continue
        
        dataset_entries = []
        for i, query in enumerate(result["queries"]):
            query_content = {
                "query": query,
                "apis": result["apis"]
            }
            entry = build_dataset_entry(query_content, [selected_categories], tool_details_with_apis)

            for j, api_entry in enumerate(entry["apis"]):
                api_entry["api_status"] = status_list[j] if j < len(status_list) else None

            dataset_entries.append(entry)
            
        return dataset_entries

def generate_batch_for_mutil_category(num_entries):
    """
    Batch generate multiple random queries and save them to a file.

    Each time the build_dataset() is called, it generates a set of three queries with the same semantics but different expressions;
    - Automatically generate an incrementing query_id (starting from 00001);
    - Cumulatively write to ./query_dataset/generated_dataset.json;
    - If the query is abnormal, write it to abnormal_queries.json;
    - Finally, print the number of successes/failures/total.
    """

    file_path = "./query_dataset/mutil_category.json"

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    print("❌ Existing file is not a list. Replacing with new list.")
                    existing_data = []
            except json.JSONDecodeError:
                print("⚠️ File is empty or corrupted. Starting a new list.")
                existing_data = []
    else:
        existing_data = []

    error_count = 0
    total_generated = 0

    for i in tqdm(range(num_entries), desc="Generating semantic variant query sets"):  
        try:
            entries = build_mutil_category_dataset()             
            if not isinstance(entries, list) or len(entries) != 3:
                raise ValueError("Returned entry is not a list of 3 items.")

            for entry in entries:
                new_id = 2001 + len(existing_data)
                new_id_str = f"{new_id:04d}"
                entry["query_id"] = new_id_str
                existing_data.append(entry)
                # append_abnormal_query(entry, json_path="./data/abnormal_queries.json")
            total_generated += 1
                
        except Exception as e:
            error_count += 1
            print(f"⚠️ Failed to generate entry group #{i + 1}. Skipping. Error: {e}")
            continue

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print("✅ Batch generation completed.")
    print(f"🟢 Successfully generated: {total_generated} groups")
    print(f"🔴 Failed groups (skipped): {error_count}")
    print(f"📦 Total entries in file now: {len(existing_data)}")

def sample_and_extract():
    """
    For each input JSON file:
    - Group data every 3 items.
    - Randomly select 1 item from each group.
    - Extract 'query', 'query_id', and 'apis' fields from the selected item.
    - Save all extracted items to a single output file.

    Parameters:
    - input_files: List[str], input JSON file paths.
    - final_output_file: str, path to save the final extracted JSON output.
    """
    input_files = [
        "/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/single_tool.json",
        "/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/single_category.json",
        "/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/mutil_category.json"
    ]
    final_output_file = "./query_dataset/toolcombinebench.json"
    extracted_data = []

    for path in input_files:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if len(data) % 3 != 0:
            print(f"⚠️ Warning: File {path} does not contain a multiple of 3 items. The last incomplete group will be skipped.")

        group_count = len(data) // 3
        for i in range(group_count):
            group = data[i*3 : i*3 + 3]
            sample = random.choice(group)
            filtered_item = {
                "query": sample.get("query"),
                "query_id": sample.get("query_id"),
                "apis": sample.get("apis")
            }
            extracted_data.append(filtered_item)

        print(f"✅ Processed file {path}: {group_count} groups sampled.")

    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 All done! Total {len(extracted_data)} records saved to: {final_output_file}")

def extract_top_queries(top_n: int):
    """
    Extract the first `top_n` items from each input JSON file,
    keeping only the 'query', 'query_id', and 'apis' fields for each item.

    Args:
        input_files (List[str]): List of input file paths (JSON files containing lists of dicts).
        top_n (int): Number of items to extract from each file.
        output_file (str): Path to save the combined output JSON file.
    """
    input_files = [
        "/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/single_tool.json",
        "/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/single_category.json",
        "/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/mutil_category.json"
    ]
    output_file = "./query_dataset/ablation.json"
    all_filtered = []

    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Loaded {len(data)} items from {file_path}")
                selected = data[:top_n]
                for item in selected:
                    filtered = {
                        "query": item.get("query"),
                        "query_id": item.get("query_id"),
                        "apis": item.get("apis")
                    }
                    all_filtered.append(filtered)
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_filtered, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 Saved {len(all_filtered)} total records to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save output file: {e}")

def remove_failed_apis():
    original_file = "/root/autodl-tmp/ToolCombine/ToolsTree0307/data/tool_data.json"
    with open(original_file, 'r') as f:
        original_data = json.load(f)
    
    failed_apis_file = "/root/autodl-tmp/ToolCombine/ToolsTree0307/data/failed_apis.json"
    with open(failed_apis_file, 'r') as f:
        failed_apis = json.load(f)
    
    # Create a set of APIs to remove (only those with status_code 11)
    apis_to_remove = set()
    for entry in failed_apis:
        if entry['status_code'] == 11:
            key = (entry['category_name'], entry['tool_name'], entry['api_name'])
            apis_to_remove.add(key)
    
    for category in list(original_data.keys()):
        for tool in list(original_data[category].keys()):
            if 'api_list_names' in original_data[category][tool]:
                original_data[category][tool]['api_list_names'] = [
                    api for api in original_data[category][tool]['api_list_names']
                    if (category, tool, api) not in apis_to_remove
                ]

                if not original_data[category][tool]['api_list_names']:
                    del original_data[category][tool]
        
        if not original_data[category]:
            del original_data[category]
    
    output_file = "./data/tool_data_new"
    with open(output_file, 'w') as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # generate_batch_for_single_tool(30)
    # generate_batch_for_single_category(3)
    # generate_batch_for_mutil_category(50)
    # sample_and_extract()
    extract_top_queries(60)
    # remove_failed_apis()