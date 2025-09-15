import json
from tqdm import tqdm
import os
from prompt import FILTER_DATASET_PROMPT
from models import gpt_for_data
TO_CLEAN_FILE = 'tool_data_new.json' 
# 以下函数都可用，只需修改文件名称，将输入和输出改成上一级处理好的文件名即可

def get_all_tools_list_from_query():
    """
    从查询数据集中获得所有工具列表
    """
    with open('/root/autodl-tmp/AnyTool/data/instruction/G1_query.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = {}

    for item in data:
        for api in item['api_list']:
            category = api['category_name']
            tool = api['tool_name']
            api_name = api['api_name']

            if category not in new_data:
                new_data[category] = {}

            if tool not in new_data[category]:
                new_data[category][tool] = {'api_list_names': []}

            if api_name not in new_data[category][tool]['api_list_names']:
                new_data[category][tool]['api_list_names'].append(api_name)

    with open('tool_data_new.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print("✅ 新文件已保存为 tool_data_new.json")

def get_all_tools_list_from_query_update():
    """
    更新刚生成的工具列表
    """
    with open('tool_data_new.json', 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    # 加载新的 input 文件
    # with open('/root/autodl-tmp/AnyTool/data/test_instruction/G2_category.json', 'r', encoding='utf-8') as f:
    with open('/root/autodl-tmp/AnyTool/data/instruction/G3_query.json', 'r', encoding='utf-8') as f:
        new_data = json.load(f)

    for item in new_data:
        for api in item['api_list']:
            category = api['category_name']
            tool = api['tool_name']
            api_name = api['api_name']

            if category not in existing_data:
                existing_data[category] = {}

            if tool not in existing_data[category]:
                existing_data[category][tool] = {'api_list_names': []}

            if api_name not in existing_data[category][tool]['api_list_names']:
                existing_data[category][tool]['api_list_names'].append(api_name)

    with open('tool_data_new.json', 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print("✅ 新 API 已追加到 tool_data_new.json")

def get_complete_tools_list():
    """
    将提取的工具的api列表替换为完全的api列表(可选)
    一个文件是从数据集中提取的，另一个是完整的工具列表
    """
    with open('tool_data_new.json', 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    with open('tool_data.json', 'r', encoding='utf-8') as f:
        tool_data = json.load(f)

    for category in output_data:
        if category in tool_data:
            for tool in output_data[category]:
                if tool in tool_data[category]:
                    # 用 tool_data 中的 api_list_names 覆盖掉 output
                    output_data[category][tool]['api_list_names'] = tool_data[category][tool]['api_list_names']
                    print(f"✅ 替换了 {category} → {tool} 的 api_list_names")

    with open('tool_data_updated.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("🎉 已生成 tool_data_updated.json，替换完成！")

def count_query_number():
    """
    计算三个query文件中共有多少个元素，即多少个查询
    """
    # 加载 input.json
    with open('/root/autodl-tmp/AnyTool/data/instruction/G1_query.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('/root/autodl-tmp/AnyTool/data/instruction/G2_query.json', 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    with open('/root/autodl-tmp/AnyTool/data/instruction/G3_query.json', 'r', encoding='utf-8') as f:
        data3 = json.load(f)
    # 统计元素数
    count1 = len(data)
    count2 = len(data2)
    count3 = len(data3)
    # ✅ input.json 中共有 201774 个元素
    print(f"✅ input.json 中共有 {count1+count2+count3} 个元素")

def remove_specified_keys(json_file_path, keys_to_remove, output_file_path=None):
    """
    删除不要的类。
    从 JSON 文件中删除指定的 key 并保存。
    :param json_file_path: 输入的 JSON 文件路径
    :param keys_to_remove: 要删除的 key 列表
    :param output_file_path: 输出文件路径（如果为 None，则覆盖原文件）
    """

    # 共有50个种类，其中有一个是Customized，自建类别，除此外删除7个，还剩42个类别的工具
    keys_to_remove = ["Monitoring", "Storage", "Devices", "Cybersecurity", "Payments", "Cryptography", "Reward"]
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for key in keys_to_remove:
        if key in data:
            del data[key]
            print(f"Removed key: {key}")
        else:
            print(f"Key not found, skipped: {key}")
    
    output_path = output_file_path if output_file_path else json_file_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Finished saving updated JSON to {output_path}")

def get_all_tools_list():
    """
    将所有工具列出来放到一个txt文件中
    """
    with open('tool_data_new_cleaned_api.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('all_tools_list_hhh_last.txt', 'w', encoding='utf-8') as out:
        for category, tools in data.items():
            out.write(f"\n=== Category: {category} ===\n")
            for tool_name in tools.keys():
                out.write(f"- {tool_name}\n")

def get_all_tools_list_no():
    """
    将所有工具列出来放到一个txt文件中，没有换行
    """
    with open('tool_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('tool_list_cleaned.txt', 'w', encoding='utf-8') as out:
        for category, tools in data.items():
            tool_names = [f'"{tool}"' for tool in tools.keys()]
            joined_tools = ', '.join(tool_names)
            out.write(f"=== Category: {category} ===\n")
            out.write(f"{joined_tools}\n\n")

def clean_category_tools(input_json_path, category, tools_to_remove):
    """
    从指定类别中删除工具，并覆盖保存原文件。

    参数：
    - input_json_path: 输入 JSON 文件路径（覆盖保存）
    - category: 要处理的类别名（区分大小写，如 'Email'）
    - tools_to_remove: 要删除的工具名集合（不区分大小写）
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if category in data:
        category_tools = data[category]
        filtered_tools = {
            tool_name: info for tool_name, info in category_tools.items()
            if tool_name.lower() not in tools_to_remove
        }
        data[category] = filtered_tools
        print(f"✅ {len(category_tools) - len(filtered_tools)} tools have been removed from the category '{category}'.")
    else:
        print(f"⚠ The category '{category}' does not exist, skipping processing.")

    with open(input_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=2)

def merge_info():
    """
    将api_details.json和category_tool_details.json进行合并
    第一个文件有api的描述，第二个文件有工具描述
    """
    with open('api_details.json', 'r', encoding='utf-8') as f:
        api_details = json.load(f)

    with open('category_tool_details.json', 'r', encoding='utf-8') as f:
        tool_details = json.load(f)

    # 合并工具描述
    merged_details = {}
    for category, tools in api_details.items():
        merged_details[category] = {}
        for tool_name, tool_info in tools.items():
            merged_details[category][tool_name] = {
                'description': tool_details.get(category, {}).get(tool_name, {}).get('tool_description', ''),
                'api_list': tool_info.get('api_list', [])
            }

    with open('merged_details.json', 'w', encoding='utf-8') as f:
        json.dump(merged_details, f, indent=4, ensure_ascii=False)

    print("合并完成，结果已保存为 merged_details.json")

def clean_null_tool():
    """
    从已经合并好的json文件中，清理无描述的工具
    """
    with open('merged_details.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    deleted_tools = {}
    total_deleted_count = 0

    for category, tools in data.items():
        cleaned_tools = {}
        deleted_in_category = []

        for tool_name, tool_info in tools.items():
            print(tool_name)
            desc_raw = tool_info.get('description')
            description = desc_raw if isinstance(desc_raw, str) else ''
            description = description.strip()

            if description != '':
                cleaned_tools[tool_name] = tool_info
            else:
                deleted_in_category.append(tool_name)
                total_deleted_count += 1  

        data[category] = cleaned_tools
        if deleted_in_category:
            deleted_tools[category] = deleted_in_category

    with open('merged_output_cleaned.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("✅ 清理完成！")
    print(f"保留的工具已保存到：output_cleaned.json")
    print(f"被删除的工具名字已保存到：output_deleted.json")
    print(f"总共删除了 {total_deleted_count} 个工具。")

def remove_deleted_tools_from_second_file():
    """
    从第二个文件(tool_data_new.json)中也删除刚才因为没有工具描述而被删除的工具
    tool_data_new.json保存的是工具列表及api名称，简略信息
    merged_output_cleaned.json保存的是工具描述及api文档的详细信息
    有时为了不加载大文件，所以需要从第二个文件对应的删除被删除的工具
    """
    with open('merged_output_deleted.json', 'r', encoding='utf-8') as f:
        deleted_tools = json.load(f)

    with open('tool_data_new.json', 'r', encoding='utf-8') as f:
        second_data = json.load(f)

    total_removed = 0
    removed_count_by_category = {}

    for category, tools in deleted_tools.items():
        if category in second_data:
            removed_count_by_category[category] = 0 

            for tool in tools:
                if tool in second_data[category]:
                    del second_data[category][tool]
                    total_removed += 1
                    removed_count_by_category[category] += 1

    with open('tool_data_new_cleaned.json', 'w', encoding='utf-8') as f:
        json.dump(second_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 删除完成，总共删除了 {total_removed} 个工具。")
    for category, count in removed_count_by_category.items():
        print(f"  - {category}: 删除了 {count} 个工具。")

    print("新文件保存为：second_file_cleaned.json")

def remove_null_api():
    """
    从全部的merged_output_cleaned.json中删除description为空的api，如果删完api之后tool没有api了，也删除tool
    并保存删掉了哪些tool和api
    """

    with open('merged_output_cleaned.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    deleted_api_count = 0
    deleted_tool_count = 0

    deleted_apis = {}  
    deleted_tools = {}   

    for category, tools in list(data.items()):
        tools_to_delete = []

        for tool_name, tool_info in list(tools.items()):
            api_list = tool_info.get('api_list', [])
            kept_apis = []
            removed_api_names = []

            for api in api_list:
                desc = api.get('description')
                print(desc)
                if desc is None or str(desc).strip() == '':
                    removed_api_names.append(api.get('name', 'unknown'))
                    deleted_api_count += 1
                else:
                    kept_apis.append(api)

            if removed_api_names:
                if category not in deleted_apis:
                    deleted_apis[category] = {}
                deleted_apis[category][tool_name] = removed_api_names

            if kept_apis:
                tool_info['api_list'] = kept_apis
            else:
                tools_to_delete.append(tool_name)

        for tool_name in tools_to_delete:
            del tools[tool_name]
            deleted_tool_count += 1
            if category not in deleted_tools:
                deleted_tools[category] = []
            deleted_tools[category].append(tool_name)

    print(f"✅ 总共删除了 {deleted_api_count} 个 API")
    print(f"✅ 总共删除了 {deleted_tool_count} 个工具")

    with open('merged_output_cleaned_api.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open('deleted_apis.json', 'w', encoding='utf-8') as f:
        json.dump(deleted_apis, f, ensure_ascii=False, indent=2)

    with open('deleted_tools.json', 'w', encoding='utf-8') as f:
        json.dump(deleted_tools, f, ensure_ascii=False, indent=2)

def clean_null_apis_from_second_file():
    """
    从tool_data_new_cleaned.json中删除description为空的api并生成新文件
    """
    with open('tool_data_new_cleaned.json', 'r') as f:
        data = json.load(f)

    with open('deleted_tools.json', 'r') as f:
        delete_tool = json.load(f)

    with open('deleted_apis.json', 'r') as f:
        delete_api = json.load(f)

    deleted_api_count = 0
    deleted_tool_count = 0

    for category, tools in list(data.items()):
        for tool_name, tool_info in list(tools.items()):
            apis_to_delete = delete_api.get(category, {}).get(tool_name, [])
            if apis_to_delete:
                original_len = len(tool_info['api_list_names'])
                tool_info['api_list_names'] = [api for api in tool_info['api_list_names'] if api not in apis_to_delete]
                deleted_count = original_len - len(tool_info['api_list_names'])
                deleted_api_count += deleted_count

            if not tool_info['api_list_names']:
                del data[category][tool_name]
                deleted_tool_count += 1

    for category, tools_to_remove in delete_tool.items():
        for tool_name in tools_to_remove:
            if tool_name in data.get(category, {}):
                del data[category][tool_name]
                deleted_tool_count += 1

    for category in list(data.keys()):
        if not data[category]:
            del data[category]

    print(f"✅ 删除的 API 总数: {deleted_api_count}")
    print(f"✅ 删除的工具总数: {deleted_tool_count}")

    with open('tool_data_new_cleaned_api.json', 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def count_api_number():
    """
    统计每个类中的工具和API数量
    """
    with open('./data/tool_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_classes = len(data)
    total_tools = 0
    total_apis = 0

    class_stats = []
    for class_name, tools in data.items():
        num_tools = len(tools)
        num_apis = sum(len(tool_info.get('api_list_names', [])) for tool_info in tools.values())
        
        total_tools += num_tools
        total_apis += num_apis
        
        class_stats.append({
            'class_name': class_name,
            'num_tools': num_tools,
            'num_apis': num_apis
        })

    class_stats.sort(key=lambda x: x['num_tools'], reverse=True)

    print(f"总共有 {total_classes} 个类\n")

    for stats in class_stats:
        print(f"类: {stats['class_name']}")
        print(f"  工具数量: {stats['num_tools']}")
        print(f"  API 总数: {stats['num_apis']}\n")

    print(f"综合统计：")
    print(f"  总工具数量: {total_tools}")
    print(f"  总 API 数量: {total_apis}")

def count_apis_per_tool_from_file(file_path):
    """"
    查看每个工具有多少api并保存在一个json文件中
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []

    for category, tools in data.items():
        for tool_name, tool_info in tools.items():
            api_list = tool_info.get("api_list_names", [])
            result.append({
                "category": category,
                "tool": tool_name,
                "api_count": len(api_list)
            })

    result.sort(key=lambda x: x["api_count"], reverse=True)

    with open('tool_api_count_sorted.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def filter_api_by_reference():
    """
    减少文件的大小，删除没用的信息（已经删掉的api或工具）
    读取两个 JSON 文件：
    - 'merged_output_cleaned_api.json' 包含完整的类别、工具和 API 信息；
    - 'tool_data_new_v2.json' 包含需要保留的类别、工具及 API 名称。

    函数将根据第二个文件中的 API 名称筛选第一个文件中的内容，
    只保留出现在第二个文件中的 API，最后将结果保存为 'filtered_api_output.json'。

    返回：
        dict，筛选后的数据结构。
    """

    with open('/root/autodl-tmp/ToolCombine/ToolsTree0307/data/api_details.json', 'r', encoding='utf-8') as f1:
        full_data = json.load(f1)

    with open('/root/autodl-tmp/ToolCombine/ToolsTree0307/data/tool_data.json', 'r', encoding='utf-8') as f2:
        filter_data = json.load(f2)

    filtered_result = {}

    for category, tools in filter_data.items():
        if category not in full_data:
            continue
        for tool_name, tool_info in tools.items():
            if tool_name not in full_data[category]:
                continue

            allowed_api_names = set(tool_info.get("api_list_names", []))
            original_api_list = full_data[category][tool_name].get("api_list", [])
            new_api_list = [
                api for api in original_api_list
                if api["name"] in allowed_api_names
            ]

            if new_api_list:
                if category not in filtered_result:
                    filtered_result[category] = {}
                filtered_result[category][tool_name] = {
                    "description": full_data[category][tool_name].get("description", ""),
                    "api_list": new_api_list
                }

    # 保存为新文件
    with open('api_details.json', 'w', encoding='utf-8') as f_out:
        json.dump(filtered_result, f_out, indent=2, ensure_ascii=False)

    print("The new file has been saved to filtered_api_output.json")

def clean_and_overwrite_tool_file():
    """
    去除api_details.json中tool及api的名称前后的空格
    """
    file_path = "api_details.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for category, tools in data.items():
        cleaned_tools = {}
        for tool_name, tool_info in tools.items():
            clean_tool_name = tool_name.strip()

            cleaned_apis = []
            for api in tool_info.get("api_list", []):
                api["name"] = api["name"].strip()
                cleaned_apis.append(api)

            tool_info["api_list"] = cleaned_apis
            cleaned_tools[clean_tool_name] = tool_info

        data[category] = cleaned_tools

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_api_list_names():
    """
    同上
    去除tool_data中工具及api名称前后的空格
    """
    file_path = "tool_data.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for category, tools in data.items():
        # 记录需要重命名的工具
        rename_map = {}

        for tool_name, tool_info in tools.items():
            clean_tool_name = tool_name.strip()

            # 清洗 API 名称前后空格
            if "api_list_names" in tool_info:
                cleaned_api_list = [api.strip() for api in tool_info["api_list_names"]]
                tool_info["api_list_names"] = cleaned_api_list

            # 如果工具名需要重命名，暂存
            if clean_tool_name != tool_name:
                rename_map[tool_name] = clean_tool_name

        # 在循环外执行键的重命名
        for old_name, new_name in rename_map.items():
            tools[new_name] = tools.pop(old_name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def count_json_elements(file_path):
    """
    统计JSON文件的根结构元素数量
    参数:
    file_path (str): JSON文件路径
    返回:
    str: 包含JSON结构类型和元素数量的字符串
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 根据结构类型统计元素数量
        if isinstance(data, dict):
            # 对象类型：统计键值对数量
            element_count = len(data)
            structure_type = "对象（键值对）"
        elif isinstance(data, list):
            # 数组类型：统计数组项数
            element_count = len(data)
            structure_type = "数组"
        else:
            # 其他类型（如单个值）
            element_count = 1
            structure_type = "单个值"
        
        print(f"JSON结构类型：{structure_type}\n元素数量：{element_count}")

    
    except FileNotFoundError:
        print("错误：文件 '{}' 不存在".format(file_path))
    except json.JSONDecodeError:
        print("错误：文件 '{}' 不是有效的JSON格式".format(file_path))
    except Exception as e:
        print("错误：发生未知异常 - {}".format(str(e)))

def clean_bad_tools():
    """
    制定一些标准，使用ChatGPT协助删除一些无用的工具，并制定标准，从要删除列表中进行误删找回
    共整理了37个类别
    """
    target_category = "Data"
    # 删除125个工具
    data_tools_to_remove = [
        "RaastaAPI",              # 名字模糊
        "frrefe",                 # 拼写乱，不知啥
        "scout",                  # 名字模糊
        "France 2D",              # 小众奇怪，不知啥
        "Thai Lotto New API",     # 偏离，彩票专用，非通用数据
        "Thai Lottery Result",    # 同上
        "Holy Bible",             # 偏内容
        "Complete Study Bible",   # 偏内容
        "Motivational Content",   # 偏内容
        "Todo Lsit",              # 拼错，占位
        "Reqres",                 # 测试用 API
        "10000+ Anime Quotes With Pagination Support",  # 偏内容
        "Unicode Codepoints",     # 偏工具类
        "Dog breeds",             # 偏内容
        "Fish species",           # 偏内容
        "House Plants",           # 偏内容
        "Matrimony Profiles",     # 偏内容，非通用数据
        "Kick.com API | Kick API",# 小众，偏平台工具
        "Lotto Draw Results - Global", # 彩票偏离
        "Trinidad Covid 19 Statistics",# 小众地区，时效性弱
        "Tesla VIN Identifier",   # 偏车辆工具类
        "VRM STR Tools",          # 拼写模糊，查不到
        "Liquidation Report",     # 拼写模糊，查不到
        "Random Chunk API",       # 模糊，不知啥
        "StopModReposts Blocklist",# 偏内容，非通用数据
        "Pocket Cube Solver",                 # 小众、偏娱乐
        "Flowers",                            # 偏内容
        "Cat breeds",                         # 偏内容
        "Quotes_v2",                          # 偏内容
        "Semantic Quotes",                    # 偏内容
        "Bible Memory Verse Flashcard",       # 偏内容
        "Cigars",                             # 偏内容
        "Feku Json",                          # 拼写奇怪
        "Africa-Api ",                        # 拼写奇怪
        "FastAPI Project",                    # 测试项目
        "Lorem Ipsum Api",                    # 占位、测试
        "Fake Data Generator",                # 测试、生成器
        "Seeding Data",                       # 测试、生成器
        "Diablo4 Smartable",                  # 小众、偏娱乐
        "Api plaque immatriculation SIV",     # 小众、拼写混乱
        "Sign Hexagram",                      # 偏内容、模糊
        "🔥 All-In-One Crypto Swiss Knife 🚀", # 花哨、稳定性差
        "Most expensive NFT artworks",        # 偏内容、非通用
        "Top NFT Collections",                # 偏内容、非通用
        "Rich NFT API + Metadata",            # 偏内容、非通用
        "Blur",                               # 小众 NFT 平台
        "Chain49",                            # 小众、模糊
        "Crypto Gem Finder",                  # 小众、模糊
        "Yandex Video API",                   # 小众平台
        "Yoonit",                             # 小众、模糊
        "Mobile-Phones",                      # 偏工具、模糊
        "Scraper's Proxy",                    # 偏工具、模糊
        "Open Brewery DB",                    # 小众、模糊
        "Leetcode Compensation",              # 小众专用
        "Awesome RSS",                        # 偏内容、模糊
        "Lowest Bins Api",                     # 小众、模糊
        "Indian Names",                             # 偏内容
        "dummyData",                                # 测试生成器
        "Payment card numbers generator",           # 生成器
        "Fake Identity Generator",                  # 生成器
        "AI Random User Generator",                 # 生成器
        "Randomizer",                               # 测试、生成器
        "Random User by API-Ninjas",                # 生成器
        "Emotional Poem",                           # 偏内容
        "Cat Facts",                                # 偏娱乐
        "Historical Figures by API-Ninjas",         # 偏内容
        "Railway Periods",                          # 小众、模糊
        "CHAT GPT AI BOT",                          # 工具拼接、偏离数据
        "BlockIt",                                  # 拼写模糊
        "LBC Shark",                                # 拼写模糊
        "PMI Jateng",                               # 小众、拼写模糊
        "BreachDirectory",                          # 小众、模糊
        # "Wayback machine domain archived lookup",   # 小众、模糊
        "Book",                                     # 小众、偏内容
        "NIV Bible",                                # 宗教、偏内容
        "Barcodes",                                 # 偏工具、模糊
        "Refactor numbers in human readable form like 1K or 1M", # 小工具
        # "Genderizeio",                              # 小工具、模糊
        "Railway Periods",                          # 小众、模糊
        # "Business email compromise (BEC) API",      # 小众、模糊
        # "Cloudflare bypass",                        # 偏工具、非通用
        "IP to Income",                             # 小众、模糊
        "Uers API",                                 # 拼写错误（应该是 Users？）
        "SA Rego Check",                            # 小众、地区专用
        "Indonesian National Identity Card Validator", # 小众、地区专用
        "Cek ID PLN PASCA DAN PRA BAYAR",          # 小众、地区专用
        "Indonesia Hotspot and Fire Location Data", # 小众、地区专用
        # "Opencage geocoder",                        # 小众、模糊
        "AI detection",                             # 小工具、模糊
        "Webit QR Code",                            # 小工具
        "QR Code Generator API",                     # 小工具
        "PlantWise",                               # 小众、用途模糊
        "Animals by API-Ninjas",                   # 偏娱乐
        "Dogs by API-Ninjas",                      # 偏娱乐
        "Cats by API-Ninjas",                      # 偏娱乐
        "Quotes by API-Ninjas",                    # 偏内容
        "Thesaurus by API-Ninjas",                 # 偏内容
        "Gloppo Fake API",                         # 测试、伪数据
        "Ultimate password generator",             # 生成器
        "Unique Username Generator By Pizza API",  # 生成器
        "EAN13 Code Generator API",                # 生成器
        "GS1-Code128 Generator",                   # 生成器
        "BaseConverterAPI",                        # 小工具、生成器
        "Lorem Ipsum by API-Ninjas",               # 生成器、伪数据
        "4D Dream Dictionary",                     # 偏内容、娱乐
        "MetaAPI Mindfulness Quotes",              # 偏内容
        "Lootero",                                 # 拼写模糊
        "Sample API",                              # 测试、伪数据
        "JoJ Web Search",                          # 拼写模糊
        "Messages",                                # 小众、模糊
        "Person App",                              # 小众、模糊
        "Jiosaavn",                                # 小众、地区专用
        "Italian Pharmacy",                        # 小众、地区专用
        "Mzansi Loadshedding Api",                 # 小众、地区专用
        "WA Rego Check",                           # 小众、地区专用
        # "Car Verification Nigeria",                # 小众、地区专用
        "Indian Mobile info",                      # 小众、地区专用
        # "Vietnamese News",                         # 小众、地区专用
        # "MagicEden",                               # 小众、模糊
        "SnapChat Story",                          # 小众、模糊
        "Diagnostics Code List",                   # 小众、模糊
        "Crops",                                   # 小众、模糊
        "Books",                                   # 偏内容
        "Hedonometer",                             # 小众、模糊
        "Tokenizer",                               # 小工具、非数据
        "File Extension",                          # 小工具、非数据
        "Scrape for me",                           # 模糊拼写
        "Request Boomerang"                        # 小众、用途模糊
    ]
    tools_to_remove_lower = set(t.lower() for t in data_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)

    target_category = "Movies"
    # 删除9个工具
    # 删除原因总结：
    # 1️⃣ 名字模糊、拼写混乱或看不出用途（像 DAILY OVRLL 系列，编号一堆，看不懂）
    # 2️⃣ 明显是测试/样本工具（如 Abir82 Bollywood Recommendations，看起来是个人或内部测试项目）
    # 3️⃣ 与 Movies 类别关系弱或不明确（如 Playphrase.me，虽和影视有关但偏语言片段，不是核心影视工具）
    movies_tools_to_remove = [
        "DAILY OVRLL 0822202130334",  # 名字混乱，看不出具体用途
        "DAILY OVRLL 0822202141203",  # 名字混乱，看不出具体用途
        "DAILY OVRLL 0822202130418",  # 名字混乱，看不出具体用途
        "DAILY OVRLL 0822202130837",  # 名字混乱，看不出具体用途
        "DAILY OVRLL 0822202143542",  # 名字混乱，看不出具体用途
        "DAILY OVRLL 0822202140642",  # 名字混乱，看不出具体用途
        "DAILY OVRLL 0822202124848",  # 名字混乱，看不出具体用途
        "Abir82 Bollywood Recommendations",  # 测试性质，疑似个人项目，不是正式API
        "Playphrase.me"  # 偏语言学习片段，和电影数据或流媒体关系弱
    ]
    tools_to_remove_lower = set(t.lower() for t in movies_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Video_Images"
    # 删除15个工具
    video_images_tools_to_remove = [
        "tes",  # 名字模糊，看不出用途
        "nowyAPI",  # 疑似测试或个人项目，名字模糊
        "Quality Porn",  # 色情内容，低优先，可靠性差
        "Alt Bichinhos",  # 名字怪异，查不到具体用途
        "james",  # 名字模糊，看不出用途
        "amir1",  # 测试性质，个人项目
        "ykapi",  # 测试性质，名字看不出用途
        "Mission Creation",  # 测试性质或个人项目
        "Pattern Monster",  # 图案生成，偏设计，不是视频/图片API
        "Thai Lottery Result Image",  # 彩票图像，和视频图片处理关系弱
        "Porn gallery",  # 色情内容，低优先，可靠性差
        "Unofficial Icons8 Search",  # 非官方接口，可能不稳定
        "Random anime img",  # 随机动漫图像，娱乐性高但用处模糊
        "MlemAPI",  # 名字怪异，看不出用途
        "Astro Gallery"  # 占星图像，偏娱乐，不是主流视频图片工具
    ]
    tools_to_remove_lower = set(t.lower() for t in video_images_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)



    target_category = "Financial"
    # 删除12个工具
    financial_tools_to_remove = [
        "Smile",  # 名字模糊，看不出用途
        "ClearDil",  # 名字模糊，查不到用途
        "Short",  # 名字太模糊，看不出用途
        "I am rich",  # 噱头性质工具，非金融实用工具
        "1p Challenge",  # 玩具性质，非金融实用工具
        "Luhn algorithm",  # 信用卡校验算法，和金融API主线关系弱
        "Number2Words",  # 数字转文字，泛用工具，不是金融专属
        "Number2Text",  # 数字转文字，泛用工具，不是金融专属
        "Consulta de boleto",  # 小众票据查询，区域性太强
        "RedStone",  # 名字模糊，查不到用途
        "Crypto Update Live",  # 名字像信息流，不是API或工具
        "CryptoInfo"  # 名字模糊，和已保留加密工具重复
    ]
    tools_to_remove_lower = set(t.lower() for t in financial_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)



    target_category = "Media"
    # 删除10个工具
    media_tools_to_remove = [
        "NewApi",  # 名字模糊，看不出用途
        "convm",  # 名字模糊，看不出用途
        "Colorful",  # 名字模糊，看不出用途
        "public-url-share",  # 名字模糊，看不出用途
        "riordanverse-api",  # 小众粉丝向API，主线关联弱
        "Baby Pig Pictures",  # 玩具性质，可爱图片，非媒体主线工具
        "Music Trivia",  # 小型娱乐工具，非主线媒体工具
        "Easy QR Code Generator",  # 泛用二维码工具，与媒体无关
        "getQRcode",  # 泛用二维码工具，与媒体无关
        "AOL On Network"  # 查询不到可靠用途
    ]
    tools_to_remove_lower = set(t.lower() for t in media_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)



    target_category = "eCommerce"
    # 删除18个工具，最终删除17个
    ecommerce_tools_to_remove = [
        "rttrt",  # 名字毫无意义，像随手敲的测试
        "Swagger PetStore",  # 明显是API示例，不是实际电商工具
        "Simple TaxRate Retrieval",  # 只是简单税率查找，和电商直接无关
        "Comany Details Search Online",  # 公司详情搜索，偏工商注册类
        "GST Number Search by Name and PAN",  # 税号查询，与购物无关
        "Leo Github API Scraper",  # Github工具，与电商完全不相关
        "Github API Scraper",  # Github工具，与电商完全不相关
        "Dungy Amazon Data Scraper",  # 拼写奇怪，来源不明
        "PPOB",  # 名字含糊不清，看不出具体用途
        "natural milk",  # 听起来像是产品词，不像是API工具
        "Product",  # 名字太模糊，毫无可用性
        "DigiXpress",  # 拼写混乱，看不出用途
        "sellytics",  # 名字拼写奇怪，查不到具体用途
        "E-commerce Delivery Status",  # 工具名字模糊，缺乏明确作用描述
        "Barcode",  # 工具太简单、用途模糊
        "Makeup",  # 单一产品类，缺乏通用性
        "Appibase",  # 名字不清楚用途，也查不到详细资料
        "Get and Compate products allover the web"  # 拼写错误（Compate），工具质量存疑
    ]
    tools_to_remove_lower = set(t.lower() for t in ecommerce_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)



    target_category = "Sports"
    # 删除22个工具
    sports_tools_to_remove = [
        "test opta",  # 明显是测试工具
        "football test",  # 明显是测试工具
        "Satellite API",  # 与体育无关，卫星工具
        "elvar",  # 名字混乱，看不出用途，已测试，不可用
        "adere",  # 名字混乱，看不出用途
        "All data",  # 名字毫无意义
        "score",  # 名字太模糊，无法判断用途
        "AllScores",  # 名字太模糊，无法判断用途
        "SportifyAPI",  # 拼写类似 Spotify，但内容混乱，已测试，不可用
        "Gold Standard Sports",  # 空泛名字，看不出具体功能
        "Decathlon Sport Places",  # 偏健身场地，非体育数据类
        "2PEAK.com Dynamic TRAINING PLANS for cycling running and Triathlon",  # 偏个人训练计划，不是体育数据API，需要单独的key
        "Metrx Factory",  # 名字混乱，查不到具体用途
        # "SportScore",  # 名字模糊，不明确
        "90 Mins",  # 名字太口语化，无法判断用途
        "Zeus API",  # 名字模糊，看不出与体育关系
        "TransferMarkt DB",  # 与 TransferMarket 重复
        "sport_v2",  # 版本号混乱，不明确
        # "sportapi",  # 名字笼统，不具体
        "msport",  # 名字模糊，看不出用途，已测试，不可用
        # "IceHockeyApi",  # 与 Ice Hockey Data / NHL API 等工具重复
        "Match APi"  # 名字拼写错误，工具质量存疑
    ]
    tools_to_remove_lower = set(t.lower() for t in sports_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)



    target_category = "Email"
    # 删除18个工具，最终删除17个
    email_tools_to_remove = [
        "MatinApi",  # 名字混乱、无法定位用途
        "apimail10",  # 测试工具，名字混乱
        "account verifyer",  # 拼写错误，已测试，连接超时
        "Email Validator_v2",  # 重复版本，保留主工具
        "Email Validator_v3",  # 重复版本，保留主工具
        "Email validator_v5",  # 重复版本，保留主工具
        "Email Validator_v9",  # 重复版本，保留主工具
        "Email Validation_v3",  # 重复功能
        "Verify Email",  # 重复功能
        "Email Verifier/Validator",  # 重复功能
        "Email Address Validator",  # 重复功能
        "Email Existence Validator",  # 重复功能
        "Emails Validator",  # 重复功能
        "Emails Verifier",  # 重复功能
        "Disposable Email Validation",  # 重复功能
        "Email Validator_v3",  # 重复列出
        "Email API",  # 笼统名字，无明确功能
        # "E-mail Check Invalid"  # 名字笼统，质量存疑
    ]
    tools_to_remove_lower = set(t.lower() for t in email_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)



    target_category = "Mapping"
    # 删除13个工具
    mapping_tools_to_remove = [
        "Dargan",  # 名字混乱、用途不明，没有该工具
        "peta",  # 名字混乱、用途不明
        "Magical Taske",  # 名字模糊、无背景
        "Verify PAN Aadhaar link_v2",  # 与地理无关，是印度身份验证工具
        # "Geocode - Forward and Reverse",  # 重复功能，保留成熟工具
        # "Forward & Reverse Geocoding",  # 重复功能，保留成熟工具
        "Reverse Geocode Locator (U.S)",  # 区域性太窄，保留全球工具，没有该工具
        "Compare Route Names",  # 功能狭窄，质量存疑
        # "smart locations",  # 模糊名字、无背景
        # "City List",  # 模糊功能，质量存疑
        "Be Zips",  # 模糊功能，质量存疑，该工具收费
        # "MapToolkit",  # 模糊功能，质量存疑
        # "Offline MapTiles"  # 模糊用途，重复 MapTiles 工具
    ]
    tools_to_remove_lower = set(t.lower() for t in mapping_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)



    target_category = "Finance"
    # 删除34个工具
    finance_tools_to_remove = [
        "Hryvna Today",  # 地区货币，仅乌克兰，不具普遍价值
        "Optimism",  # 太模糊，名字像区块链项目，不像API工具
        "Kiann_Options_SABR",  # 看似个人项目或测试
        "Kiann_Options_Project",  # 看似个人项目或测试
        "JoJ Finance",  # 拼写怪异、用途不明
        "sundayfinance",  # 看似个人侧项目，信息模糊
        "Crypto grana",  # 拼写怪异，像个人小项目
        "walletapi.cloud",  # 太泛，来源和用途模糊
        "Palmy Investing API",  # 查不到来源，可信度低
        "Date Calculator",  # 完全与金融无关
        "Global Flight Data",  # 与 Finance 类别不匹配（航班数据）
        "Currency Quake",  # 名字太奇怪，像游戏或个人项目
        "investing financial stocks",  # 名字像关键词堆砌，拼写奇怪
        "G - Finance",  # 名字太简略模糊
        "Qvantana",  # 拼写奇怪、用途不明
        "Quotient",  # 名字泛用，金融关系模糊
        "Is This Coin A Scam",  # 说法模糊，像诈骗检测，不是标准API
        "Litecoin Wallet",  # 钱包客户端，不是API，且偏技术
        "360MiQ",  # 名字模糊，查不到公开金融API信息
        "Fake Credit Card Number Generator API",  # 明显测试/生成伪造数据，不合规
        "MathAAS",  # 纯数学计算服务，与Finance无关
        "Credit Card Number Validator",  # 有专门用途，但名字太宽泛，可能不专注Finance
        "ISLAMICOIN",  # 不明API，名字像项目名，可信度低
        "Test_v3",  # 明显测试工具，版本号暗示
        "Fake Credit Card Generator ",  # 与前面重复，伪造数据工具
        "Ethereum random address generator. ETH key pairs generator",  # 纯密钥生成工具，不是财经API
        "tokenlist",  # 名字过于泛，无法确定金融API
        "Crypto Asset Cold Wallet Create",  # 纯钱包创建工具，非数据接口
        "Direct Debit Managed Service",  # 看似业务系统，不像API工具
        "RetrieveUSTaxRate",  # 名字笼统，没有公开详细资料
        "MS Finance",  # 名字太简单模糊，无法确认来源
        "Armatic",  # 个人小项目或不明API
        "Involve Thailand FX Rates",  # 区域性小众API，资料不全
        "Routing Number Bank Lookup"  # 工具性质过于专一，且有类似更优选项
    ]
    tools_to_remove_lower = set(t.lower() for t in finance_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Health_and_Fitness"
    # 删除4个工具
    health_tools_to_remove = [
        "Scoring Tables API",  # 名称太泛，无法判断具体健康应用
        "selector-tipo-consultas",  # 拼写混乱，非英语，模糊用途
        "fastingcenters",  # 名称过于模糊，不确定是否健康相关API，已测试，不可用
        "Horostory"  # 名称与健康无关，可能是星座类内容
    ]
    tools_to_remove_lower = set(t.lower() for t in health_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Travel"
    # 删除18个工具，删除了17个
    travel_tools_to_remove = [
        "52 In Kicks",  # 拼写混乱，无意义
        "Tsaboin Cams",  # 拼写混乱，疑似测试
        "ASR Hub",  # 名字模糊，不明确
        "funtrip",  # 名字无意义，无法判断用途
        # "borders",  # 太泛，无旅游服务属性
        "VOO",  # 模糊无意义
        "StreetNarrator",  # 不明确用途
        "Airports Info (α)",  # alpha测试版，已测试，连接超时   
        "Flight _v2",  # 拼写不规范，重复含糊
        "Flight , Airline Consolidator, Flight Aggregator",  # 名称重复且表达混乱
        "flight | flight aggregator",  # 名称重复且不规范
        "Turkey Public Holidays",  # 虽属假期但偏非旅游服务，没有该工具
        "Travelopro",  # 名字不明，无公开资料
        "Travelo Pro",  # 名字不明，无公开资料
        "Get_Ticket_Information",  # 名称不专业，且无详细说明
        "BiggestCities",  # 与“Biggest Cities”重复，且无区别
        "world cities by homicide rate",  # 负面属性，不适合旅游工具
        "Ranked Crime Cities"  # 同上
    ]
    tools_to_remove_lower = set(t.lower() for t in travel_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Database"
    # 删除42个工具
    database_tools_to_remove = [
        "aaaa",  # 名字模糊无意义
        "Summery",  # 拼写错误，意义不明
        "Roman Gods By Pizza API",  # 不相关，内容偏离数据库主题
        "Portfolio",  # 太模糊，不明确用途
        "SuggestUse",  # 不明确用途
        "TEAS",  # 名字无说明
        "HSN TSN",  # 拼写混乱，内容不明
        "expense data",  # 名字过于模糊
        "Quadro de sócios CPF/CNPJ",  # 非英文且无明确用途说明
        "veiculos-api",  # 拼写不规范，缺说明
        "Mocking Rock ",  # 无关且名字奇怪
        "Taekwondo_Athlete_World_Ranking",  # 不相关，体育类非数据库
        "Watch Database",  # 名字模糊
        "Lista de empresas por segmento",  # 非英文且无说明
        "siteDomain",  # 不明确
        "Domain Reputation",  # 功能偏网络安全，非数据库核心
        "fwd-api",  # 模糊无意义
        "Python Libraries tst",  # 明显测试
        "get Mood",  # 内容不明，疑似测试
        "Data Breach Checker",  # 偏安全类，不核心数据库
        "Indian RTO's Names Search ",  # 内容不明且拼写混乱
        "Complete Criminal Checks Offender Data",  # 内容偏法律，非数据库API
        "Customer",  # 过于模糊
        "TEST",  # 测试
        "Women in Tech",  # 内容无关
        "Real_Estate_Heroku",  # 名字模糊，不明确
        "gcfen",  # 拼写无意义
        "hhside",  # 拼写无意义
        "Voter Card Verifications",  # 非核心数据库
        "car code",  # 内容不明
        "gsaauction",  # 名字不明，疑似非数据库
        "Winget API",  # 名字与数据库无关
        "Dati Comuni",  # 非英文且内容不明
        "capitall_fitness",  # 名字奇怪，与数据库无关
        "Response WebHook",  # 功能偏通信非数据库
        "Test1",  # 测试
        "Legend of Takada Kenshi",  # 名字无意义，不相关
        "Student",  # 过于模糊
        "ICMAI Verification",  # 可能是验证非数据库
        "Card Databse",  # 拼写错误，且内容不清
        "testapi2",  # 测试
        "Chattydata",  # 名字无明确用途
        "Joke Test",  # 测试
        "testGetApi",  # 测试
        "Geo_Coder_Dev",  # 开发版本疑似测试
        "classes",  # 过于模糊
        "GetTempMail"  # 邮件临时工具，非数据库类
    ]

    tools_to_remove_lower = set(t.lower() for t in database_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Education"
    # 删除28个工具
    education_tools_to_remove = [
        "paultest",  # 测试工具
        "apiDeveloper",  # 测试工具
        "TestAPI",  # 测试工具
        "Test_Crypto_Api",  # 测试工具
        "APIGabin",  # 测试工具
        "aftab",  # 名字无意义
        "mony",  # 名字无意义
        "yosi",  # 名字无意义
        "kittichai",  # 名字无意义
        "nguyenthanhduy178.tk",  # 拼写奇怪，无说明
        "SevenTraderAPI",  # 交易类，非教育
        "weather",  # 气象类
        "weatherJS",  # 气象类
        "weather_v3",  # 气象类
        "message-api",  # 偏通信，非教育
        "futboleta",  # 足球票务，不相关
        "COVID19PH",  # 公共卫生，非教育
        "democracia",  # 内容偏政治，非教育
        "todo",  # 名字模糊，非教育
        "hellonext",  # 内容模糊，非教育
        "nail",  # 与教育无关
        "lm_API",  # 名字模糊
        "Safe Exam",  # 安全类，非学习内容，没有该工具
        "Samyutam Eduction",  # 拼写错误且无说明
        "sekolah",  # 非英文名，缺少上下文，需要单独的key
        "fachaApi",  # 名字无意义
        "tapzulecountry",  # 名字拼写混乱
        "ask-ai"  # 内容模糊，非明确教育用途，已测试，连接超时
    ]

    tools_to_remove_lower = set(t.lower() for t in education_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Science"
    # 删除7个工具
    science_tools_to_remove = [
        "teste",  # 测试工具
        "test",  # 测试工具
        "Irene",  # 名字无意义
        "manatee jokes",  # 笑话，不是科学
        "Al-Quran",  # 宗教内容
        "Yawin Indian Astrology",  # 占星，非科学
        "Astrologer"  # 占星，非科学
    ]

    tools_to_remove_lower = set(t.lower() for t in science_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Monitoring"
    # 删除8个工具
    monitoring_tools_to_remove = [
        "Counter",  # 名字太模糊
        "OTP",  # 不算监测工具
        "Patient",  # 名字模糊，不知所指
        "Certficate",  # 拼写错误
        "SearchingWebRequest",  # 测试或模糊用途
        "Screenshot Maker",  # 截图工具，不是监控
        # "Youtube classification api",  # 分类分析，不是监控
        "Plerdy"  # 偏向分析工具，不是直接监控
    ]

    tools_to_remove_lower = set(t.lower() for t in monitoring_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Translation"
    # 删除8个工具
    translation_tools_to_remove = [
        "01",  # 数字代号，模糊
        "13f918yf19o1t1f1of1t9",  # 混乱字符串
        "navicon1",  # 名字模糊
        "Nitro",  # 名字模糊，看不出用途
        "English synonyms",  # 同义词查询，不是翻译
        "Translator",  # 过于泛化，无法区分，没有该工具
        "Translate it",  # 过于泛化，无法区分，没有该工具
        "Translate Language"  # 过于泛化，无法区分，没有该工具
    ]

    tools_to_remove_lower = set(t.lower() for t in translation_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Tools"
    # 删除21个工具
    tools_tools_to_remove = [
        "dimondevosint",  # 拼写混乱，不知用途
        "teamriverbubbles random utilities",  # 名字混乱，泛化工具集合
        "👋 Demo Project_v12",  # 演示项目
        "Placeholder text API for your application",  # 占位工具
        "echo-api",  # 占位或测试
        "SimpleEcho",  # 测试或回声工具
        "ProbablyThese",  # 名字模糊
        "Plus One",  # 名字模糊
        "core_js",  # 模块名，不是独立工具
        "QR Code API_v33",  # QR工具重复，留清晰版本
        "QR Code_v18",  # 同上
        "QR Code API_v67",  # 同上
        "QR Code API_v92",  # 同上
        "Custom QR Code with Logo_v2",  # 太细分、低版本，留主工具
        "Variable Size QR Code API",  # 名字不清晰
        "Quick QR Code Generator",  # 泛化名字
        "TVB QR Code",  # 指向特定用途，泛化场景用不上
        "ProbablyThese",  # 模糊、不知用途
        # "Anchor Data Scrapper",  # 不知具体用途，名字模糊
        "UptoSite Link Shortener",  # 不清晰、不常用
        "ExplorArc's Password Generation API",  # 非主流、名字不清晰
        "story",  # 名字模糊，不知用途，没有该工具
        "Todo",  # 占位或测试工具
        "Proof of concept",  # 测试工具
        "QuickMocker",  # 测试工具
        "Starline Telematics",  # 不相关，用途模糊
        "Joe Rogan Quote Generator",  # 不相关，搞笑用途
        "Shakespeare Translator",  # 趣味工具，用处不大
        "Words World",  # 模糊，不知用途
        "QR code_v8",  # 重复、低版本
        "QR Code API_v6",  # 重复、低版本
        "QR Code API_v119",  # 重复、版本乱
        "QRLink API",  # 名字模糊，重复，没有该工具
        "QR-Generator-Api",  # 重复，已测试，连接超时
        "QR Code Generator API_v6",  # 重复、低版本
        # "QRickit QR Code QReator",  # 名字混乱
        "Simple & Cheap QR CODE GENERATOR",  # 名字混乱
        "Bar QR Code Generator",  # 重复
        "QR code generator with multiple datatypes .",  # 名字模糊
        "qrcode-generator-base64",  # 细分重复工具
        "QR Code Wizard"  # 重复、名字模糊
    ]
    tools_to_remove_lower = set(t.lower() for t in tools_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Text_Analysis"
    # 删除18个工具
    text_analysis_tools_to_remove = [
        "gruite",  # 名字模糊
        "SpeakEasy",  # 名字模糊
        "Fast Reading",  # 模糊用途
        "Hello world_v2",  # 测试工具
        "testingsunlife",  # 测试工具
        # "Google Translate",  # 重复
        "Google Translate_v2",  # 重复
        # "Profanity Filter",  # 重复
        "Profanity Filter_v2",  # 重复
        "Sentiment Analysis Service",  # 重复
        "Sentimental Analysis_v2",  # 重复
        "Sentiment Analysis_v12",  # 重复
        "Sentiment analysis_v13",  # 重复
        "PAN Card OCR",  # OCR类，剔除
        "Philippines Voter Card OCR",  # OCR类
        "Philippines Passport OCR",  # OCR类
        "Philippines Driving License OCR",  # OCR类
        "Philippines TIN OCR",  # OCR类
        "National ID Vietnam OCR",  # OCR类
        "Voter Card OCR",  # OCR类
        "Philippines Social Security OCR",  # OCR类
        "Driving License OCR"  # OCR类
    ]
    tools_to_remove_lower = set(t.lower() for t in text_analysis_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)



    target_category = "Advertising"
    # 删除37个工具
    advertising_tools_to_remove = [
        "20211230 testing upload swagger",  # 测试工具
        "Putreq",  # 测试/占位
        "route-precedence-test-1",  # 测试
        "testing",  # 测试
        "PublicAPITestingInbox",  # 测试
        "pe-demo",  # 测试
        "PetstoreRateLimit",  # 测试
        "Test_v2",  # 测试
        "MultipleTeamsCallingTest",  # 测试
        "ThisshouldbeFREE",  # 测试
        "petstore blitz",  # 测试
        "PrivatePublicAPI",  # 测试
        "FreePlanwithHardLimit",  # 测试
        "versioning-free",  # 测试
        "FOOTBALL_API_KEY",  # 测试
        "March4",  # 测试/模糊
        "httpbin",  # 测试工具
        "underscore test",  # 测试
        "Hello World",  # 占位
        "test",  # 测试
        "hello_v2",  # 测试
        "Test1",  # 测试
        "ssssss",  # 测试/乱码
        "Test_v5",  # 测试
        "test_v6",  # 测试
        "versions-paid",  # 测试
        "ddd",  # 模糊
        "a",  # 单字母
        "asd",  # 模糊
        "lets",  # 模糊
        "16e0e5f7a076c2f62a2e41f6b5b99d37e5b9b25e",  # 乱码
        "Frederick",  # 模糊
        "Free IP Geolocation",  # 非广告类，需要单独设置key
        "Abstract IP geolocation",  # 非广告类，需要单独设置key
        "JobsApi",  # 非广告类，已废弃
        # "Autosub"  # 非广告类
    ]
    tools_to_remove_lower = set(t.lower() for t in advertising_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Weather"
    # 删除17个工具
    weather_tools_to_remove = [
        "Testing for My Use",  # 测试
        "WeatherTest",  # 测试
        "Test",  # 测试
        "Test_v2",  # 测试
        "daily limit check",  # 测试
        "weather_v14",  # 测试
        "weather_v13",  # 测试
        "Weather_v6",  # 测试
        "123",  # 测试/模糊
        "History",  # 名字太宽泛，找不到
        "Forecast",  # 名字太宽泛，找不到
        "MagicMirror",  # 模糊
        "OikoWeather",  # 模糊
        "havo",  # 模糊
        "Monitoring Syatem",  # 拼写混乱
        "CurrencyConverter"  # 跑题工具
    ]
    tools_to_remove_lower = set(t.lower() for t in weather_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "News_Media"
    # 删除15个工具
    news_media_tools_to_remove = [
        "Test_v2",  # 测试
        "OneLike",  # 模糊
        "papercliff",  # 模糊
        "Green Gold",  # 模糊
        "Goverlytics",  # 模糊，已测试，不可用
        "depuertoplata",  # 模糊
        "getMedia",  # 模糊
        "Book Cover API",  # 跑题
        # "Flixster",  # 跑题
        "Movies details",  # 跑题，找不到
        "Online Movie Database",  # 跑题
        "SB Weather",  # 跑题
        "Football-API",  # 跑题
        "PAC API"  # 跑题
    ]
    tools_to_remove_lower = set(t.lower() for t in news_media_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Business_Software"
    # 删除19个工具
    business_software_tools_to_remove = [
        "test2",  # 测试工具
        "testapi_v2",  # 测试工具
        "OPA-test",  # 测试工具
        "demo",  # 测试/demo工具
        "BogieApis",  # 测试/demo工具
        "fffvfv",  # 名字混乱，用途不明
        "dathoang",  # 名字混乱，用途不明
        "aug13",  # 名字混乱，用途不明
        "sdfsdf_v2",  # 名字混乱，用途不明
        "HelloRold",  # 名字混乱，用途不明
        "newnew",  # 名字混乱，用途不明
        "My API 12345",  # 名字混乱，用途不明
        "ucinema",  # 与商务软件无关（影院）
        "Slot and Betting Games",  # 与商务软件无关（博彩）
        "Find any IP address or Domain Location world wide",  # 与商务软件无关（纯地理查询）
        "Logo",  # 小工具，功能太单一，不算商务软件
        "ShortLink",  # 小工具，功能太单一，不算商务软件
        "QuizApp",  # 小工具，功能太单一，不算商务软件
        # "B2BHint"  # 数据不足，无法判断可靠性，偏向删除
    ]
    tools_to_remove_lower = set(t.lower() for t in business_software_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Gaming"
    # 删除20个工具
    gaming_tools_to_remove = [
        "big89",  # 名字混乱，用途不明
        "HleBy6eK",  # 名字混乱，用途不明
        "a56219609685dd9033d060cdbb60093c",  # 名字混乱，用途不明
        "Fodase",  # 名字混乱，用途不明
        "Scott",  # 名字混乱，用途不明
        "StonxzyAPI",  # 名字混乱，用途不明
        "game",  # 占位或测试名
        "Plugin.proto",  # 占位或测试名
        "hitmen2",  # 用途不明，查不到信息
        "GoTW",  # 用途不明，查不到信息
        "Weeby",  # 用途不明，查不到信息，已测试，不可用
        "Aposta Ganha Aviator API",  # 博彩下注类，非主流游戏工具
        "Vai de Bob Aviator API",  # 博彩下注类，非主流游戏工具
        "Estrelabet Aviator API",  # 博彩下注类，非主流游戏工具
        "Bet7k Aviator API",  # 博彩下注类，非主流游戏工具
        "Simbrief - Get latest OFP",  # 航空飞行计划工具，偏离游戏主题，找不到
        "League Of Legends Champion Generator",  # LoL 同类工具已保留多个，去掉重复
        "CricketLiveApi",  # 体育比分类，偏离主流视频/电子游戏，已测试，连接超时
        # "Play to Earn Blockchain Games",  # 区块链游戏小众工具，质量存疑
        # "Trivia by API-Ninjas"  # 小型工具，质量存疑
    ]
    tools_to_remove_lower = set(t.lower() for t in gaming_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Location"
    # 删除25个工具
    location_tools_to_remove = [
        "BPS",  # 名字模糊，看不出用途
        "Services",  # 名字模糊，看不出用途
        "Location",  # 名字模糊，太宽泛
        "Location_v2",  # 测试或占位，名字不清楚
        "Get IP Info_v2",  # 测试或占位，名字模糊
        "get cities",  # 名字模糊，功能范围太宽
        "https://ipfinder.io/",  # 是网址，不是工具名
        "IsItWater.com",  # 与位置关系弱，更像水域信息
        "Income by Zipcode",  # 偏收入统计，不是核心定位
        # "Feroeg - Reverse Geocoding",  # 拼写混乱，不优质
        "Schweizer Postleitzahlen",  # 过于小众地区，工具重复
        "KFC locations",  # 品牌特定，不通用
        "Nearby Tesla Superchargers",  # 品牌特定，不通用
        "Find By UDPRN",  # 小众，英国邮政专用编码，通用性差
        # "Vessels",  # 船只，不是纯地理位置工具
        "IP Directory",  # 含义模糊，不明确是查什么
        "Wyre Data",  # 名字模糊，查不到明确用途
        "Partenaires Mobilis",  # 本地化、拼写怪异，不通用
        "Referential",  # 太模糊，看不出定位功能
        "BDapi",  # 拼写模糊，缺乏明确功能描述
        "GeoWide",  # 模糊，不明确用途
        "CatchLoc",  # 名字模糊，不清楚具体用途
        "MapReflex",  # 名字模糊，不常见或查不到
        # "Bng2latlong",  # 小众英国坐标转换，重复度高
        "Itcooking.eu - IP Geolocation"  # 名字混乱，不优质来源
    ]
    tools_to_remove_lower = set(t.lower() for t in location_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Social"
    # 删除25个工具
    social_tools_to_remove = [
        "Socie",  # 名字模糊
        "ABCR",  # 名字模糊
        "IDD",  # 名字模糊
        "MESCALC",  # 名字模糊
        "Chuck Norris",  # 笑话类
        "Tronald Dump",  # 笑话类
        "Fortune Cookie",  # 趣味类
        "Flirty words",  # 趣味类
        "Real Love Calculator",  # 趣味类
        "Marryme",  # 趣味类
        "QUIZ",  # 趣味类
        "Instagram_v2",  # 版本重复，保留最新
        "Instagram_v3",  # 版本重复，保留最新
        "Instagram_v6",  # 版本重复，保留最新
        "Instagram_v7",  # 版本重复，保留最新
        "Instagram_v9",  # 版本重复，保留最新
        "Instagram_v10",  # 版本重复，保留稳定版
        "Olato Quotes",  # 拼写怪异，已测试，不可用
        "Tweesky",  # 拼写怪异，已测试，不可用
        "PeerReach",  # 拼写怪异，已测试，连接超时
        "Popular languages",  # 偏语言统计
        # "Jobs from remoteok",  # 偏招聘
        "Instagram #1",  # 奇怪命名
        "Instagram Fast",  # 奇怪命名
        "Instagram Cheapest"  # 奇怪命名
    ]
    tools_to_remove_lower = set(t.lower() for t in social_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Search"
    # 删除19个工具
    search_tools_to_remove = [
        "Fiverr Pro services",           # 非搜索，服务市场
        "TorrentHunt",                   # 主题偏离，BT种子搜索，范围偏窄且不稳定
        "VIN decoder",                  # 车辆VIN解码，与搜索类别不符
        "Vehicle Ownership Cost",       # 车辆成本估算，非搜索工具
        "OPT-NC public docker images",  # 拼写混乱，不相关，貌似测试或专业镜像库
        "NFT Explorer",                 # 非搜索类别，区块链NFT
        "Emplois OPT-NC",               # 名称不明确，且非通用搜索类
        "Postali",                     # 名字模糊，难以确定用途
        "DBU_API",                     # 拼写模糊，无具体说明，疑似测试或专业内部工具
        "barcode.monster",             # 条码相关，非通用搜索
        "Front Page search engine",    # 名称过于通用，难以判定，且无资料，已测试，不可用
        "Netlas All-in-One Host",      # 名称模糊，偏向主机或网络管理，不是搜索
        "ExplorArc's Link Finder",     # 模糊工具名，缺乏清晰用途说明，已测试，不可用
        "UfU",                        # 拼写混乱，无法理解含义
        "Vehicle Market Value",        # 车辆市场估值，非搜索工具
        "HotelApi",                   # 酒店API，非搜索类
        "Question-Answered",          # 名称模糊，不清楚是否搜索
        "Postleitzahl zu Adresse",    # 德语邮编转地址工具，专业性强且模糊
        "Searchhook"                  # 拼写混乱，无法明确判断用途，已测试，不可用
    ]
    tools_to_remove_lower = set(t.lower() for t in search_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Visual_Recognition"
    # 删除6个工具
    visual_recognition_tools_to_remove = [
        "Parking places",              # 语义模糊，且不明显是视觉识别工具
        "Voltox OCR",                 # 名称不常见，缺乏公开资料支持，模糊不明，错误工具已测试
        "Fast Hcaptcha Solver",       # CAPTCHA解码器，不是视觉识别范畴
        "Fast Recaptcha V2 Solver",   # 同上，验证码破解，不属于视觉识别
        "Auther Check",               # 名称模糊，无法判定具体用途
        "OCR - Separate Text From Images"  # 语义重复且不明确，已有OCR工具覆盖
    ]
    tools_to_remove_lower = set(t.lower() for t in visual_recognition_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Transportation"
    # 删除6个工具
    transportation_tools_to_remove = [
        "Datamo",                     # 名称模糊，功能不明
        "AP sample",                  # 明显测试/示例工具
        "FachaAPI",                   # 名称模糊，缺乏明确用途
        "OpenNWI",                   # 名称不明，难以判断是否相关
        "TrackingPackage",            # 名称不符合交通运输核心范畴，且功能不明，应属于物流
        # "TimeTable Lookup "           # 名称不规范，末尾有空格且不够明确
    ]
    tools_to_remove_lower = set(t.lower() for t in transportation_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Other"
    # 删除52个工具
    other_tools_to_remove = [
        "pl1box1",                   # 名称模糊，无明显用途
        "13",                       # 无意义数字，无法判断
        "4Bro  - 1337X",            # 名称模糊，可能与搜索站或盗版相关，不合适
        "Demo1",                    # 明显测试/演示工具
        "sample-app-config.json",   # 配置文件，不是工具
        "team petstore",            # 名称模糊，不明用途
        "quotsy",                   # 拼写不明，模糊用途
        "JAK_API",                  # 名称不明确
        "URLTEST",                  # 测试性质
        "erictestpet",              # 测试性质
        "Node Express API Tutorial",# 教程性质，不是API工具
        "Most Exclusive API",       # 名称模糊，缺乏明确用途
        "myapi",                    # 名称模糊
        "TestAPI_v4",               # 测试工具
        "Darko Androcec Example",   # 示例性质
        "Fake users_v6",            # 测试/演示数据
        "Hello World",              # 测试/示例性质
        "5970b9b3dcc757e61e53daec3e720974",  # 无意义字符串
        "S-YTD",                    # 名称模糊，不明用途
        "Lab2",                     # 测试/演示性质
        "csa_v2",                   # 名称模糊
        "DAd",                      # 名称模糊
        "PracticaUipath",           # 名称模糊，且Uipath通常非API工具
        "toptalOnlineTest",         # 测试性质
        "katzion test",             # 测试性质
        "AI endpoint",              # 名称模糊
        "Ignition",                 # 名称模糊
        "Echo",                     # 名称模糊
        "Heck Yes Markdown",        # 与工具无关
        "Uros Kovcic",              # 个人名，非工具
        "getAssessments",           # 名称模糊
        "RakutenSupportDefaultTeam",# 组织/团队名，非工具
        "backend",                  # 通用术语，模糊
        "TestApi_v2",               # 测试工具
        "Testing",                  # 测试性质
        # "Calulator",                # 拼写错误，模糊用途
        "Evaluate expression",      # 过于模糊
        # "Cors Proxy",               # 代理相关，偏非工具性质
        "Test_v4",                  # 测试工具
        "Test_v2",                  # 测试工具
        # "Random Username Generate", # 用途可，但因其他测试工具太多，可保留也可删（这里视严格程度决定）
        "testApi",                  # 测试工具
        "MyPersonal",               # 个人名或模糊用途
        "Testing Demo",             # 测试性质
        "TestAPI_v3",               # 测试工具
        "haime-python-api",         # 名称拼写错误，模糊用途
        "PragmavantApi",            # 名称拼写模糊，不明确
        "colegiosantaana",          # 地名或组织名，不明确
        "pl1box1",                  # 重复，模糊用途
        "Quran Com"                 # 宗教相关内容，和Other类别工具用途不明确
    ]
    tools_to_remove_lower = set(t.lower() for t in other_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Food"
    # 删除20个工具
    food_tools_to_remove = [
        "Testing_v2",               # 测试工具，名字不明确
        "Auth",                     # 认证类工具，与食品无关
        "Viva City Documentation",  # 文档，非API工具，且用途不明
        "postcap",                  # 名字模糊，无法判断用途
        "Payment",                  # 支付类，与食品无关
        "CamRest676",               # 名字模糊，无明显食品关联
        "MyNewTestApi",             # 测试工具
        "test1",                    # 测试工具
        "Testing_v3",               # 测试工具，名字不明确
        "betaRecipes",              # 测试/测试版
        "Generic Food_v2",          # 名称泛泛，可能重复或无明显差异
        "Recipe_v2",                # 版本老旧，保留最新版本
        "Recipe_v3",                # 版本老旧，保留最新版本
        "Recipe_v4",                # 重复版本，考虑合并，删旧
        "Food Nutrional Data",      # 拼写错误（Nutritional），且可能重复
        "favoriteFoodApi",          # 名字模糊，不够标准
        "ComfyFood",                # 名字不够明确，且无知名度
        "pizzaallapala",            # 拼写混乱，无法确定用途
        "Store Groceries",          # 不够明确，可能与食品库存相关但信息不足
        "Bespoke Diet Generator"    # 名字较长且模糊，非主流或不确定功能
    ]
    tools_to_remove_lower = set(t.lower() for t in food_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Entertainment"
    # 删除42个工具
    entertainment_tools_to_remove = [
        "4D Results",                   # 名字含糊，难确定用途
        "Minecraft-Forge-Optifine",     # 游戏Mod工具，不是API娱乐内容
        "mbar",                        # 名字无意义，模糊
        "kast",                        # 名字无意义，模糊
        "Dung Bui",                    # 人名或无意义，模糊
        "Pipotronic",                  # 名字无意义，模糊
        "yurna",                      # 名字无意义，模糊
        "Sholltna",                   # 名字无意义，模糊
        "Testing",                    # 测试工具
        "Interesting Facts API",      # 偏知识类，不是纯娱乐
        "Text similarity calculator",# 工具类，非娱乐内容
        "Direct Porn_v2",             # 成人内容，且版本不规范
        "rapid-porn",                 # 成人内容，非正规API
        "Porn Movies",                # 成人内容
        "Ten Secrets About Mega888 Ios Download That Has Never Been Revealed For The Past Years", # 垃圾推广标题
        "69bd0c7193msh3c4abb39db3a82fp18c336jsn8470910ae9f0", # 无意义字符串，疑似垃圾
        "DaddyJokes",                 # 重复笑话工具，且名不规范
        "Joke1",                     # 名称模糊，低质量
        "jokes ",                    # 空格拼写，低质量重复
        "cubiculus - managing LEGO set collection", # LEGO收藏工具，不属于娱乐API
        "Twitter video downloader mp4", # 工具类，非娱乐内容
        "Facebook video downloader MP4",# 工具类，非娱乐内容
        "Img to ASCII",              # 图像转ASCII，工具类非娱乐API
        "MagicMirrorAPI",            # 不相关
        "Allbet Online Casino  in Singapore", # 明显博彩广告，不正规
        "Follow Youtube Channel",    # 运营工具，不是API娱乐内容
        "mydailyinspiration",        # 灵感内容，偏知识类
        # "Would You Rather",          # 游戏类重复，名气小
        # "Comedy(KK)",                # 名字不专业
        "Random Yes/No",             # 随机工具，不是娱乐API
        "Vadym-rock-paper-scissors-api", # 游戏API太小众且拼写乱
        "Cash4Life",                 # 彩票类，非主流娱乐API
        "Euro Millions",             # 彩票类，非主流娱乐API
        "Lotto America",             # 彩票类，非主流娱乐API
        "Fantasy 5",                 # 彩票类，非主流娱乐API
        "New York Lottery",          # 彩票类，非主流娱乐API
        "Casino related"             # 其他博彩相关工具一律剔除，保持健康向
    ]
    tools_to_remove_lower = set(t.lower() for t in entertainment_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)

    target_category = "Commerce"
    # 删除26个工具
    commerce_tools_to_remove = [
        "sandbox mktplace eu  01 products",      # 名字模糊，含空格且不规范
        "sandbox ecombr com - 04 orders",        # 名字混乱，sandbox测试环境
        "sandbox mktplace eu - 04 orders",       # 同上，测试环境
        "sandbox ecombr com - 01 products",      # 同上，测试环境
        "dd",                                   # 无意义，名字模糊
        "Movives",                              # 电影相关，不属于商务
        "test pg prod",                         # 测试工具
        "test",                                # 测试工具
        "togo420",                             # 名字无意义
        "Test API",                            # 测试工具
        "test_v3",                            # 测试工具
        "JSMon",                              # 名字无意义
        "Test_v2",                            # 测试工具
        "API1",                               # 泛泛无信息
        "BrowserObject",                      # 开发相关，非商务
        "IP2Proxy",                          # 代理工具，非商务
        "ClickbankUniversity",               # 教育培训，不是工具
        "resumeAPI",                        # 简历相关，非商务
        "Face Compare",                    # 人脸识别，非商务
        "Inventory and eCommerce hosted and self-hosted solution",  # 描述泛泛，不是具体API
        "Cartify",                        # 名字不清晰，信息少
        # "Flance AliExpress",             # 拼写疑似错误，可信度低
        "HaloBiru.store",               # 店铺名，非API工具
        "Azaprime",                    # 信息不明，名字不清晰
        "789club New",                 # 名字无意义，杂乱
        "Amazon Live Data"             # 可能重复Amazon_API_v2，冗余
    ]
    tools_to_remove_lower = set(t.lower() for t in commerce_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Music"
    # 删除28个工具
    music_tools_to_remove = [
        "kotak7",                               # 名字无意义
        "Miza",                                # 名字无意义
        "Latest Spotify Downloader",           # 版权风险工具
        "MusiclinkssApi",                      # 拼写错误，名字不清晰
        "MyAPI",                              # 泛泛无信息
        # "L-yrics",                           # 名字不清晰
        "station",                           # 泛泛且无意义
        "getSongs",                         # 泛泛无意义
        # "MusicZone",                        # 不明确
        "shoxbps",                         # 名字无意义
        "baixar musicas mp3 completas",    # 版权风险，且名字拼写杂乱
        "deepsound",                       # 不明确，模糊
        "xmusic",                         # 不明确
        # "melodyn",                        # 不明确
        "RojMusic",                      # 不明确
        "online-music",                  # 名字模糊泛泛
        "Youtube MP3 Converter",          # 版权风险工具
        "Bandamp Downloader API",         # 版权风险
        "ReverbNation Song Downloader",  # 版权风险
        "MP3 Downloader",                 # 版权风险
        "VkoorSound",                    # 不明确
        "myPlayvv",                     # 不明确
        "GG",                          # 无意义缩写
        "Spotify _v2",                  # 命名不规范，带空格
        "Billboard-API",                # 冗余，保留其他版
        "Billboard API_v2",             # 冗余，保留主版
        "MusicAPI",                    # 泛泛无具体信息
        "247NaijaBuzz"                 # 不明确，名字非主流音乐工具
    ]
    tools_to_remove_lower = set(t.lower() for t in music_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Business"
    # 删除69个工具
    business_tools_to_remove = [
        "apfd",  # 拼写模糊，不明用途
        "Interceptor Sample",  # 明显测试工具
        "token2go",  # 名字模糊
        "DOMAINE nc",  # 名字模糊
        "Ziff",  # 名字模糊
        "Woo-temp",  # 明显测试工具
        "acopaer",  # 拼写模糊
        "PetStoreAPI2.0",  # 类似测试/demo工具
        "TEXTKING Translation",  # 不属于商务核心
        "SOTI Sync",  # 名字不清晰，业务无关
        "Shwe 2D Live Api",  # 不相关娱乐工具
        "mgs",  # 拼写模糊
        "DaysAPI",  # 作用不清晰
        "Fake Brightcove",  # 明显测试或伪造
        "Testing out Sharing w/ Rachael",  # 明显测试工具
        "5ka_Porocila",  # 名字模糊，难以判断
        "000",  # 无意义名称
        "print",  # 无意义名称
        "news",  # 业务不相关
        "hyhryh",  # 拼写混乱无意义
        "KarenRecommends",  # 不相关推荐工具
        "WRAWS Load Test",  # 明显测试
        "test_API_v2",  # 明显测试
        "test_v16",  # 明显测试
        "XTREAM",  # 业务不相关
        # "'"/>><img src=x onerror=prompt(document.domain);>",  # 恶意代码或无效输入
        "constructioness",  # 拼写错误且无意义
        "test_v20",  # 明显测试
        "partydiva",  # 不明业务工具
        "test",  # 明显测试
        "567 Live app 2022",  # 业务无关
        "TestPOCApi",  # 明显测试
        "Healthcaremailing",  # 名字不清晰，疑似非核心业务
        "Test_API",  # 明显测试
        "hfghfgh",  # 拼写混乱
        "Ken",  # 拼写不明确
        "prueba",  # 西班牙语测试
        "InsafEl",  # 名字不明
        "enrich",  # 名字模糊
        "11BET",  # 赌博相关，业务不相关
        "denemeapisi",  # 土耳其语测试API
        "test apizzz",  # 明显测试
        "Test_v18",  # 明显测试
        "Rotating Proxies",  # 技术支持，不是业务工具
        "0MMO",  # 名字模糊
        "b4c6577cb2msh7c15fca215f2c30p1f1717jsn998498c6865e",  # 非业务工具，杂乱字符
        "My Bot",  # 测试或无关工具
        "New Client",  # 无意义
        "Test_v14",  # 明显测试
        "Personas",  # 不清晰工具
        # "eSignly",  # 名字不清晰，非核心业务
        "KUBET-",  # 赌博相关
        "kassbet",  # 赌博相关
        "789bettnet",  # 赌博相关
        "ExplorArc's Internel Links Crawler",  # 爬虫非业务核心
        "FM_API_RELEASE v0.1",  # 版本号测试
        "testpk",  # 明显测试
        "dxjewof",  # 拼写无意义
        "pss",  # 无意义
        "LD",  # 名字模糊
        "My public interface",  # 无明确业务含义
        "Test New Team",  # 测试工具
        "123CACA",  # 无意义
        "오류",  # 韩文错误，非业务工具
        "Test_v3",  # 明显测试
        "1800 Printing Test",  # 明显测试
        "test1",  # 明显测试
        "anttdev 2",  # 测试或开发用
        "qwe",  # 无意义
        "111",  # 无意义
        "Test Plan",  # 明显测试
        "czkjlj",  # 无意义
        "AE6888 Link vao nha cai AE688  2022"  # 赌博相关广告
    ]
    tools_to_remove_lower = set(t.lower() for t in business_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Artificial_Intelligence_Machine_Learning"
    # 删除6个工具
    ai_ml_tools_to_remove = [
        "test",  # 明显测试工具，无实际用途
        "teste",  # 拼写错误，测试工具
        "Carbon management",  # 与AI/ML无关，环保管理
        # "Web Scraping API",  # 爬虫工具，不属于AI/ML
        "Midjourney best experience",  # 重复，保留“MidJourney”
        # "Explor-Arc's Colors API"  # 名称不明，非AI核心工具
    ]
    tools_to_remove_lower = set(t.lower() for t in ai_ml_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)


    target_category = "Communication"
    # 删除17个工具
    communication_tools_to_remove = [
        "getBs",  # 名字含糊，无用
        "whin",  # 拼写不明，无意义
        "Dudu",  # 无明确用途
        "Punto 61",  # 不明工具名称
        "Barbaraaa",  # 无意义名称
        "Revista Verde",  # 非通信工具
        "HoG",  # 拼写无意义
        "bcolimited",  # 无用名称
        "bitikas1",  # 无意义名称
        # "mobile-recharge-plans-api-tariff-Plans-free",  # 不属于通信API
        "test apideno",  # 明显测试
        "SawyerTest",  # 测试工具
        "Another Rapid Test",  # 测试工具
        "https://i.imgur.com/JM9TESV.jpg/",  # 链接，不是API
        "Weather_dataSet",  # 天气数据，非通信
        "ISS Location",  # 空间站位置，非通信
        # "Flask"  # Web框架，非通信API
    ]
    tools_to_remove_lower = set(t.lower() for t in communication_tools_to_remove)
    clean_category_tools(TO_CLEAN_FILE, target_category, tools_to_remove_lower)

    # 共整理37个类别
    # ✅ 已从类别 'Data' 中删除 118 个工具。
    # ✅ 已从类别 'Movies' 中删除 9 个工具。
    # ✅ 已从类别 'Video_Images' 中删除 15 个工具。
    # ✅ 已从类别 'Financial' 中删除 12 个工具。
    # ✅ 已从类别 'Media' 中删除 10 个工具。
    # ✅ 已从类别 'eCommerce' 中删除 17 个工具。
    # ✅ 已从类别 'Sports' 中删除 19 个工具。
    # ✅ 已从类别 'Email' 中删除 16 个工具。
    # ✅ 已从类别 'Mapping' 中删除 7 个工具。
    # ✅ 已从类别 'Finance' 中删除 34 个工具。
    # ✅ 已从类别 'Health_and_Fitness' 中删除 4 个工具。
    # ✅ 已从类别 'Travel' 中删除 16 个工具。
    # ✅ 已从类别 'Database' 中删除 47 个工具。
    # ✅ 已从类别 'Education' 中删除 28 个工具。
    # ✅ 已从类别 'Science' 中删除 7 个工具。
    # ✅ 已从类别 'Monitoring' 中删除 7 个工具。
    # ✅ 已从类别 'Translation' 中删除 8 个工具。
    # ✅ 已从类别 'Tools' 中删除 38 个工具。
    # ✅ 已从类别 'Text_Analysis' 中删除 20 个工具。
    # ✅ 已从类别 'Advertising' 中删除 35 个工具。
    # ✅ 已从类别 'Weather' 中删除 16 个工具。
    # ✅ 已从类别 'News_Media' 中删除 13 个工具。
    # ✅ 已从类别 'Business_Software' 中删除 18 个工具。
    # ✅ 已从类别 'Gaming' 中删除 18 个工具。
    # ✅ 已从类别 'Location' 中删除 22 个工具。
    # ✅ 已从类别 'Social' 中删除 24 个工具。
    # ✅ 已从类别 'Search' 中删除 19 个工具。
    # ✅ 已从类别 'Visual_Recognition' 中删除 6 个工具。
    # ✅ 已从类别 'Transportation' 中删除 5 个工具。
    # ✅ 已从类别 'Other' 中删除 45 个工具。
    # ✅ 已从类别 'Food' 中删除 20 个工具。
    # ✅ 已从类别 'Entertainment' 中删除 34 个工具。
    # ✅ 已从类别 'Commerce' 中删除 25 个工具。
    # ✅ 已从类别 'Music' 中删除 25 个工具。
    # ✅ 已从类别 'Business' 中删除 71 个工具。
    # ✅ 已从类别 'Artificial_Intelligence_Machine_Learning' 中删除 4 个工具。
    # ✅ 已从类别 'Communication' 中删除 15 个工具。

def filter_valid_entries():
    """
    将原记录中category_name、tool_name 和 api_name 不在tool_data中出现的记录从data_path中删除
    过滤 data_path 中的记录，仅保留那些在 api_dict_path 所指定结构中
    其 category_name、tool_name 和 api_name 都能匹配的记录。

    参数：
    - data_path: 原始数据集 JSON 文件路径
    - api_dict_path: 工具类别-工具名-API 名列表的 JSON 文件路径
    - output_path: 筛选后的输出 JSON 文件路径
    """
    # 1. 加载数据
    with open("/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/G3_query.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open("tool_data.json", 'r', encoding='utf-8') as f:
        tool_api_dict = json.load(f)

    # 2. 定义检查函数
    def is_valid_api_entry(entry):
        for api in entry.get('relevant APIs', []):
            category = api.get('category_name')
            tool = api.get('tool_name')
            api_name = api.get('api_name')
            
            if not (category and tool and api_name):
                return False
            if category not in tool_api_dict:
                return False
            if tool not in tool_api_dict[category]:
                return False
            if 'api_list_names' not in tool_api_dict[category][tool]:
                return False
            if api_name not in tool_api_dict[category][tool]['api_list_names']:
                return False
        return True

    # 3. 过滤数据
    cleaned_dataset = [entry for entry in dataset if is_valid_api_entry(entry)]

    # 4. 写入结果
    with open("/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/G3_query.json", 'w', encoding='utf-8') as f:
        json.dump(cleaned_dataset, f, ensure_ascii=False, indent=2)

    print(f"筛选后保留 {len(cleaned_dataset)} 条记录，已保存至G3_query.json")

def merge_and_deduplicate(key='query_id'):
    """
    将两个json文件合并成一个json文件，即将两个查询集合并成一个并去重
    """
    # 读取文件
    with open("/root/autodl-tmp/AnyTool/data/test_instruction/G2_category.json", 'r', encoding='utf-8') as f1:
        list1 = json.load(f1)
    with open("/root/autodl-tmp/AnyTool/data/test_instruction/G2_instruction.json", 'r', encoding='utf-8') as f2:
        list2 = json.load(f2)
    
    # 用字典去重（后出现的对象覆盖先出现的）
    unique_dict = {}
    for item in list1 + list2:
        unique_dict[item[key]] = item  # 相同ID的对象会被覆盖
    
    # 转换为列表
    merged = list(unique_dict.values())
    
    # 保存结果
    with open("G2_query.json", 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"去重后总元素数：{len(merged)}，重复ID已处理")

def replace_relevant_apis(json_path):
    """
    构建一个快速查找字典，以 (tool_name, api_name) 为键，映射到 API 的基本元信息。

    参数:
        api_list (list of dict): 包含多个 API 定义的列表，每个元素为一个字典，至少包含以下字段：
            - "category_name": 所属类别
            - "tool_name": 工具名称
            - "api_name": 接口名称

    返回:
        dict: 一个字典，键为 (tool_name, api_name) 的元组，值为对应 API 的简化信息字典，包含：
            - "category_name"
            - "tool_name"
            - "api_name"
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        api_list = entry.get("api_list", [])
        relevant_apis = entry.get("relevant APIs", [])

        # 构建查找表 {(tool_name, api_name): 对应简化字段}
        api_lookup = {
            (api["tool_name"], api["api_name"]): {
                "category_name": api["category_name"],
                "tool_name": api["tool_name"],
                "api_name": api["api_name"]
            }
            for api in api_list
        }

        # 替换 relevant APIs
        new_relevant_apis = []
        for tool_name, api_name in relevant_apis:
            key = (tool_name, api_name)
            if key in api_lookup:
                new_relevant_apis.append(api_lookup[key])
            else:
                print(f"⚠ 未找到匹配项: {key}")

        entry["relevant APIs"] = new_relevant_apis

    # 保存修改后的结果
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("✅ relevant APIs 字段已成功替换。")

def filter_unsolvable_queries(json_path):
    """
    使用 gpt() 接口判断每条 query 是否满足工具组合解决条件，
    筛除不满足条件的项，并将结果覆盖写回原始 JSON 文件。

    筛选规则基于 ToolBench 可解性标准：
    1. 缺少关键信息（如电话号码、真实 ID）；
    2. 含有伪造参数（如不存在的 URL）；
    3. 明确指定了某个 API；
    4. 查询不合理（如请求太广泛、模糊）；
    """
    # 读取原始数据
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = []
    dropped = 0

    for item in tqdm(data, desc="Filtering queries"):
        query = item.get("query", "").strip()
        if not query:
            dropped += 1
            continue
        prompt = FILTER_DATASET_PROMPT.replace("{query}", query)
        try:
            result = gpt_for_data(prompt).choices[0].message.content
            print(result)  # 接口函数，返回 "Yes" 或 "No"
            if isinstance(result, str) and result.strip().lower().startswith("yes"):
                filtered.append(item)
            else:
                dropped += 1
        except Exception as e:
            print(f"⚠ Error filtering query: {e}")
            dropped += 1
        # time.sleep(0.5)  # 限速，避免 API 限流

    # 写回清洗后的数据
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 筛选完成：共处理 {len(data)} 条，保留 {len(filtered)} 条，剔除 {dropped} 条。")

def merge_json_files():
    """
    将多个 JSON 文件（每个是一个 list）合并成一个大 list，并保存到输出文件中。

    参数:
        input_files (list[str]): 输入文件路径列表。
        output_file (str): 合并后保存的输出文件路径。
    """
    input_files = [
        '/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/G1_query.json',
        '/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/G2_query.json',
        '/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/G3_query.json']
    output_file = '/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/toolbench.json'
    merged_data = []

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在，跳过: {file_path}")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"⚠️ 文件不是 JSON 列表，跳过: {file_path}")
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON 解析失败，跳过: {file_path}，错误: {e}")
    
    # 保存合并结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(merged_data, out_f, indent=2, ensure_ascii=False)

    print(f"✅ 合并完成，总计条目数: {len(merged_data)}，保存至: {output_file}")

def extract_api_names():
    """
    读取原始工具 JSON 文件，提取每个工具的 API 路径，重构为新的结构并保存。
    """
    input_file_path = "/root/autodl-tmp/ToolCombine/ToolsTree0307/data/api_details.json"
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_data = {}

    for category, tools in data.items():
        output_data[category] = {}
        for tool_name, tool_info in tools.items():
            api_list = tool_info.get("api_list", [])
            # 提取 name 字段
            api_names = [api.get("name") for api in api_list if isinstance(api, dict) and "name" in api]
            output_data[category][tool_name] = {
                "api_list_names": api_names
            }

    output_file_path = "tool_data.json"
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 提取完成，结果保存至: {output_file_path}")


if __name__ == '__main__':
    # 第一步，删除一些不常见的类别
    # remove_specified_keys('tool_data_new_v2.json', None, 'tool_data_new_v2.json')
    # get_all_tools_list()
    # merge_info()
    # get_all_tools_list_no()
    # 第二步，删除没有工具描述的tool
    # clean_null_tool()
    # remove_deleted_tools_from_second_file()
    # 第三步，删除没有描述的api
    # remove_api()
    # clean_null_api()
    # count_apis_per_tool_from_file("./data/tool_data.json")

    # 第四步，手动删除不能见名知意的工具，测试工具等等

    # 第五步，将信息比较全的去除没有的api信息
    # filter_api_by_reference()

    # 第六步，将工具和api名前后的空格去除
    # clean_and_overwrite_tool_file()
    # clean_api_list_names()
    # count_json_elements("/root/autodl-tmp/AnyTool/atb_data/anytoolbench_old.json")
    # count_json_elements("/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/anytoolbench.json")
    # count_json_elements("/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/toolbench.json")
    # count_json_elements("/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/G1_query.json")
    # count_json_elements("/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/G2_query.json")
    # count_json_elements("/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/G3_query.json")
    # get_all_tools_list_no()

    # 统计api数量
    count_api_number()
    # filter_valid_entries()
    # merge_and_deduplicate()
    # replace_relevant_apis("/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/G3_query.json")
    # filter_valid_entries()
    # filter_unsolvable_queries("/root/autodl-tmp/ToolCombine/ToolsTree0307/query_dataset/G2_query.json")
    
    # merge_json_files()
    # extract_api_names()




