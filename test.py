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
        tools = list(data_tool[selected_category].keys())
        selected_tools = random.sample(tools, min(TOOL_COUNT_ONE_CATEGORY, len(tools)))

        tool_details = []  
        tool_details_with_apis = []  
        tool_entries_for_prompt = []
        tool_entries_full = []
        for tool in selected_tools:
            if len(data_tool[category][tool]["api_list_names"]) > 15:
                selected_tools.remove(tool)
                continue
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

        prompt = GENERATE_COMBINATION_SINGLE_CATEGORY_PROMPT.replace("{USER_INPUT}", str(tool_details))
        response = gpt_for_data(prompt)
        result = response.choices[0].message.content    
        print("\n", "🤖 GPT Response:\n", result, "\n")
        if result.strip().lower() == "none":
            print("❌ The tool combination is unreasonable, skip...")
            continue
        try:
            result = clean_and_load_json(result)
            if not isinstance(result, dict) or "queries" not in result or "apis" not in result:
                print("❌ Result malformed, skipping")
                continue
            if isinstance(result, str):
                result = json.loads(result)
        except Exception as e:
            print(f"⚠️ JSON parsing failed, skipped. Error: {e}")
            continue
        selected_tools = result.get("selected_tools")

        api_details = []
        for item in selected_tools:
            category = category
            tool = item['tool_name']
            apis = data[category][tool]['api_list']

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
        response = gpt_for_data(prompt)
        result = json.loads(response.choices[0].message.content)
        result = clean_and_load_json(result)
        print("\n", "🤖 GPT Response:\n", result, "\n")
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
            return None
        
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

text = """{'role': 'system', 'content': "\nYou are APIGPT, You have access to a database of apis. The database has the following categories: ['Commerce', 'Other', 'Mapping', 'Travel', 'Text_Analysis', 'Database', 'Video_Images', 'Energy', 'Communication', 'Weather', 'Food', 'Data', 'Medical', 'Science', 'Translation', 'Social', 'Media', 'Transportation', 'SMS', 'Movies', 'Location', 'Business_Software', 'Advertising', 'Entertainment', 'Sports', 'Music', 'Artificial_Intelligence_Machine_Learning', 'Education', 'Business', 'Search', 'Jobs', 'Health_and_Fitness', 'Logistics', 'Gaming', 'eCommerce', 'News_Media', 'Financial', 'Tools', 'Visual_Recognition', 'Email', 'Finance', 'Events'].\nYour goal is to find the relevant categories for a task and assign each to a new agent. \nYou can use the get_tools_in_category function to retrieve the available tools of a specific category. \nIf you are unsure about the functionality of some tools, you can use the get_tools_descriptions function to retrieve the details of these tools. This will help you understand the general functionality of each category.\nYou can use the create_agent_category_level function to assign a relevant category to the agent, which is called immediately when you think it needs to be assigned.\nWhen you finish the assignment, call the Finish function. \nYou can assign multiple categories, but each agent can only be assigned one category.\nYou can try other categories of tools that you haven't explored before. The query may be solved by tools in unexpected categories.\nAt each step, you need to give your thought to analyze the status now and what to do next, with the function calls to actually excute your step. All the thought is short, at most in 3 sentence.\nRemember, you do not need to answer the query, all you need is to find all possible relevant categories and assign them to agents.\n"}
{'role': 'user', 'content': 'Please determine relevant categories and assign them. Task description: Please retrieve the latest status for the U.S. flag along with any recent announcements regarding half-staff events for my review. Begin!'}
{'role': 'assistant', 'content': "To find relevant categories for retrieving the latest status of the U.S. flag and recent half-staff announcements, I should consider categories related to Government, News, and possibly Social or Communication for announcements. First, I will explore the News_Media category since it likely covers any announcements related to such topics.\n\nI'll retrieve the tools in the News_Media category first. \n\nLet's proceed with that."}
{'role': 'function', 'name': 'get_tools_in_category', 'param': '{"category_name":"News_Media"}', 'content': "['Sri Lanka News Api', 'Election2020 Smartable', 'Live Climate Change_v2', 'Live world futbol news', 'climate-change-api_v2', 'Klimawandel Live', 'News_v3', 'Indonesia News', 'Live Climate Change News_v2', 'Crypto News_v3', 'Latest IPL News', 'Crypto News Live_v2', 'News API_v2', 'Climate News API_v2', 'India Today Unofficial', 'Greek News in English', 'A.I. Smartable', 'Climate change_v31', 'LIVE TV API', 'AI News - Global', 'Movie Articles and News', 'LGBTQ World News Live', 'News Network', 'Crypto News_v11', 'Climate Change API_v5', 'Instant Cameroon News', 'pollution-news-api', 'Climate News Feed', 'Anime News Net', 'Climate Change News_v5', 'Energy Price News', 'dev-to-api', 'German Police and Crime News Live', 'Global Recession Live', 'Covid news_v2', 'Crypto News_v2', 'Newscatcher', 'BizToc', 'ALL Crypto News Feed', 'Reuters Business and Financial News', 'Kubric: The Comprehensive Movie News API', 'Cote Ivoire news', 'NFT API News', 'New climate', 'AI News_v2', 'Climate Change Live_v27', 'Climate Change Live_v3', 'Coronavirus Smartable', 'Arabic news API', 'Political Bias Database', 'Flixster', 'Climate Change API Tutorial', 'East China News', 'Climate Change API_v3', 'Google News', 'Moka News', 'Malaysia Kini', 'Webit News Search', 'NewsData', 'US Presidential Election', 'TRW - news', 'berita', 'Philippine News', 'H Crypto News', 'Targeted Keyword trend', 'Narco in Mexico', 'article-api', 'Fashion Industry News Data Collection', 'AR VR News', 'Google News API', 'Pakistan News API', 'News Sentiment', 'Pharma Industry Data Collection', 'The Onion Api', 'Top Stories of Kuensel', 'IEEE Spectrum API', 'Smart Curator']"}
{'role': 'assistant', 'content': "The News_Media category has a variety of tools that seem suitable for retrieving announcements about the U.S. flag and half-staff events. Tools like 'Google News API' and 'News API_v2' could provide the latest information desirable for this task.\n\nNext, I'll create an agent for the News_Media category to handle this. After that, I will also check the Government category to see if there are any additional relevant tools. \n\nLet's assign the News_Media category to an agent first."}
{'role': 'function', 'name': 'create_agent_category_level', 'param': '{"category":"News_Media"}', 'content': 'category: News_Media assigned.'}
{'role': 'function', 'name': 'get_tools_in_category', 'param': '{"category_name":"Other"}', 'content': "['IFC BIM Services', 'HauteCouture-API', 'uniswap-v2-api', 'ChurchApp', 'Ukraine war data', 'Sunnah Fasting', 'The South Asian Express', 'SurveyMethods', 'Ridet NC', 'Uniblock', 'flow study', 'Dog API', 'ClickMeter', 'hanime-python-api', 'Korean Baby Name Ranking', 'DogBreeds', 'Horoscope Astrology', 'ODAM', 'World of Quotes', 'Horoscope API', 'Felina API', 'chevereto', 'cat-facts', 'Ecoindex', 'Online Code Compiler', 'Shreehari', 'Inferno', 'Words to numbers', 'J2ACoin', 'Cardsearch', 'Quotes Villa', 'QR API', 'Agc', 'petFood', 'MyPolicyAPI', 'courses', 'Neko SFW', 'Borsa', 'httpbin-test', 'GetSource', 'Bucket List by API-Ninjas', 'favicon-finder', 'Term Analysis', 'Industrypedia Random Articles', 'Mexican Vehicles REPUVE', 'trainmyAPI27', 'Facts by API-Ninjas', 'Baby Names by API-Ninjas', 'ZomatoAPI-bhaskar', 'whois', 'Hafez', 'Calulator', 'Cors Proxy', 'Random Username Generate']"}
{'role': 'assistant', 'content': 'It appears the Government category does not have any directly relevant tools for retrieving the U.S. flag status or half-staff announcements. The News_Media agent should suffice for this task.\n\nNow, I will proceed to finalize the assignment since we have established the News_Media category agent to handle the request effectively.  \n\nCalling the Finish function now.'}
{'role': 'user', 'content': 'Please invoke the list of tools I provided for you. Do not just think about but not act.'}
{'role': 'function', 'name': 'get_tools_in_category', 'param': '{"category_name":"Government"}', 'content': 'Illegal category name'}
{'role': 'assistant', 'content': "It appears that my previous request to list tools for a category was not executed correctly. Let's focus on completing the task at hand. \n\nSince I already assigned the News_Media category to an agent, I'll finalize the process now to complete the assignment. \n\nThis ensures the task can be executed with the newly created agent. Calling the Finish function now."}
{'role': 'user', 'content': 'Please invoke the list of tools I provided for you. Do not just think about but not act.'}
{'role': 'assistant', 'content': "It seems I misunderstood your last request about re-evaluating the tools available in the News_Media category. However, I've already accessed the tools in that category earlier.\n\nTo proceed effectively, let's double-check the tools available under the News_Media category, as this will guide the assignment for the task regarding announcements about the U.S. flag and half-staff events. \n\nI'll retrieve the tool descriptions for some of the key tools listed under News_Media to evaluate their relevance. I'll start with the first 5 tools to get an idea of their functionality. \n\nLet's do that now."}
{'role': 'user', 'content': 'Please invoke the list of tools I provided for you. Do not just think about but not act.'}
{'role': 'function', 'name': 'get_tools_in_category', 'param': '{"category_name":"News_Media"}', 'content': "['Sri Lanka News Api', 'Election2020 Smartable', 'Live Climate Change_v2', 'Live world futbol news', 'climate-change-api_v2', 'Klimawandel Live', 'News_v3', 'Indonesia News', 'Live Climate Change News_v2', 'Crypto News_v3', 'Latest IPL News', 'Crypto News Live_v2', 'News API_v2', 'Climate News API_v2', 'India Today Unofficial', 'Greek News in English', 'A.I. Smartable', 'Climate change_v31', 'LIVE TV API', 'AI News - Global', 'Movie Articles and News', 'LGBTQ World News Live', 'News Network', 'Crypto News_v11', 'Climate Change API_v5', 'Instant Cameroon News', 'pollution-news-api', 'Climate News Feed', 'Anime News Net', 'Climate Change News_v5', 'Energy Price News', 'dev-to-api', 'German Police and Crime News Live', 'Global Recession Live', 'Covid news_v2', 'Crypto News_v2', 'Newscatcher', 'BizToc', 'ALL Crypto News Feed', 'Reuters Business and Financial News', 'Kubric: The Comprehensive Movie News API', 'Cote Ivoire news', 'NFT API News', 'New climate', 'AI News_v2', 'Climate Change Live_v27', 'Climate Change Live_v3', 'Coronavirus Smartable', 'Arabic news API', 'Political Bias Database', 'Flixster', 'Climate Change API Tutorial', 'East China News', 'Climate Change API_v3', 'Google News', 'Moka News', 'Malaysia Kini', 'Webit News Search', 'NewsData', 'US Presidential Election', 'TRW - news', 'berita', 'Philippine News', 'H Crypto News', 'Targeted Keyword trend', 'Narco in Mexico', 'article-api', 'Fashion Industry News Data Collection', 'AR VR News', 'Google News API', 'Pakistan News API', 'News Sentiment', 'Pharma Industry Data Collection', 'The Onion Api', 'Top Stories of Kuensel', 'IEEE Spectrum API', 'Smart Curator']"}
{'role': 'assistant', 'content': "I’ve retrieved the list of tools from the News_Media category again. To provide clarity, I'll move on to identify specific tools that could help in retrieving the latest announcements regarding the U.S. flag and half-staff events. \n\nThe tools that look particularly relevant include 'Google News API', 'News API_v2', and 'Newscatcher' among others. \n\nNext, I will get descriptions for these relevant tools to ensure their functionality aligns with the task requirements. I'll do this for the top three tools now: 'Google News API', 'News API_v2', and 'Newscatcher'. Let's proceed with that. "}
{'role': 'user', 'content': 'Please invoke the list of tools I provided for you. Do not just think about but not act.'}"""
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
token_cnt = len(enc.encode(str(text)))
print(token_cnt)