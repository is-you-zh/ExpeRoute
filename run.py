import os
import json
import logging
from termcolor import colored
import time
from copy import deepcopy
from collections import OrderedDict
from threading import Semaphore
from utils import *
from prompt import *
from models import *
from long_term_memory import *
from function_database import *
from arguments import parse_args
from lats import lats_search
from react import react_search
from dfs import dfs_search
from classifier import select_strategy
from value import evaluate_completion_score
import traceback

args = parse_args()
sem = Semaphore(16)  
multi_thread = True

if multi_thread:
    counter_lock = threading.Lock()
else:
    counter_lock = DoNothingContextManager()
   
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs(args.log, exist_ok=True)

class Agent(object):
    def __init__(self) -> None:
        self.failed_reason = None
        self.messages = []
        self.depth = 0
        self.index = 0
        self.finish_search = False
        self.sub_agents = []


class Main_Search_Agent(Agent):
    def __init__(self, query) -> None:
        super().__init__()
        self.categories = []
        self.query = query
        self.api_mapping = {
            "get_tools_in_category": get_tools_in_category,
            "get_tools_descriptions": get_tools_descriptions,
            "create_agent_category_level": self.create_agent_category_level,
            "Finish": Finish
        }
        self.functions = [
            get_tools_in_category_function,
            get_tools_descriptions_function,
            create_agent_category_level_function,
            finish_function
        ]
   
        self.messages = [
            {
            "role": "system",
            "content": META_AGENT_PROMPT.replace('{categories}', str(query_all_categories()))
            },
            {"role": "user",
             "content": f"Please determine relevant categories and assign them. Task description: {query} Begin!"
            }
        ]
        logger.info(f'Main_Search_Agent prompt:{self.messages}')

    def create_agent_category_level(self, category):
        global agents, index
        if category in self.categories:
            print(colored(f'category: {category} already assigned', 'green'))
            return f'category: {category} already assigned. Please do not call the same tool multiple times with the same parameters.'
        if not isinstance(category, str):
            return f'Error: category: {category} is not str'
        if category not in query_all_categories():
            return f'category: {category} not in database'
        self.categories.append(category)
        with counter_lock:
            agents.append(Category_Agent(self.query, category))
            index += 1
            agents[-1].depth = self.depth + 1
            agents[-1].index = index
        self.sub_agents.append(agents[-1])
        if multi_thread:
            thread = threading.Thread(target=agents[-1].category_search)
            sem.acquire()
            thread.start()
            with counter_lock:
                threads.append(thread)
            result = f'category: {category} assigned.'
        else:
            agents[-1].category_search()
            result = f'category: {category} assigned. The status of current found apis is: {status}'

        return result
     
    def assign_main(self):
        global total_tokens, stop, error_flag
        for i in range(10):
            logger.info(f'The {i+1} loop of assign_main:')
            if stop or total_tokens > 200000:
                if multi_thread:
                    sem.release()
                logger.info(f"stop: {stop}")
                logger.info(f"total_tokens: {total_tokens}")
            try:
                response = call_gpt(
                        messages=self.messages,
                        model=model_name,
                        functions=self.functions
                    )
            except:
                error_flag = True
                stop = True
                continue
            if isinstance(response, str):
                print('Response is str. response:', response)
                logger.info(f'Response is str. response: {response}')
                continue
            with counter_lock:
                total_tokens += response.usage.total_tokens            
            print('#'*100)
            print(colored(f'The {i+1} loop of assign_main:','yellow'))
            if(response.choices[0].message.content):
                print('Thought:', response.choices[0].message.content)
                message = {
                    "role": "assistant",
                    "content": response.choices[0].message.content if response.choices[0].message.content is not None else '',
                }
                self.messages.append(message)
                logger.info(message)
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                # print('tool call number', len(tool_calls))
                logger.info(f'tool call number: {len(tool_calls)}')
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    if function_name.lower() == 'finish':
                        self.messages.append(
                            {
                                "role": "function",
                                "name": function_name,
                                "content": 'Finished main search.',
                            })
                        self.finish_search = True
                        print(colored('Finished main search.', 'green'))
                        logger.info('Finished main search.')
                        if multi_thread:
                            sem.release()
                        return self.categories
                
                    if function_name not in self.api_mapping:
                        function_name = 'hullucinating_function_name'
                        function_call_result = "Function name error"
                        logger.info("hullucinating function name")
                    else:
                        try:
                            function_call_result = self.api_mapping[function_name](**json.loads(function_args))
                        except Exception as e:
                            print(e, function_name, function_args, file=open(f'{args.output_dir}/error.txt', 'a', encoding='utf-8'))
                            function_call_result = str(e)
                            
                    message = {
                                "role":"function",
                                "name": function_name,
                                "param": function_args,
                                "content": str(function_call_result),
                            }
                    self.messages.append(message)
                    print(json.dumps(message, ensure_ascii=False, indent=2))
                    logger.info(json.dumps(message, ensure_ascii=False, indent=2))   
            else:
                message = {
                    'role': "user",
                    'content': 'Please invoke the list of tools I provided for you. Do not just think about but not act.'
                }
                self.messages.append(message)
                logger.info(message)
                
        self.finish_search = True
        if multi_thread:
            sem.release()
        return self.categories

    def resume_search(self):
        if stop or total_tokens > 200000: 
            logger.info(f"stop: {stop}")
            logger.info(f"total_tokens: {total_tokens}")
            self.finish_search = True
            if multi_thread:
                sem.release()
            return self.categories
        if self.failed_reason is not None:
            self.messages.append({"role": "user", "content": REFIND_CATEGORY_PROMPT.replace('{failed_reason}', str(self.failed_reason))})
            self.failed_reason = None
        return self.assign_main()


class Category_Agent(Agent):
    def __init__(self, query, category=None) -> None:
        super().__init__()
        self.category = category
        self.tools = get_tools_in_category(self.category)
        self.query = query
        self.info = f'category: {self.category} assigned'
        self.api_mapping = {
            "query_all_categories": query_all_categories,
            "Finish": Finish,
            "get_tools_descriptions": get_tools_descriptions,
            "create_agent_tool_level": self.create_agent_tool_level,
        }
        self.functions = [
            get_tools_descriptions_function,
            finish_function,
            create_agent_tool_level_function
        ]
    
    def category_search(self):
        print(colored(f'assigning category: {self.category}', 'green'))
        logger.info(f'assigning category: {self.category}')
        if len(self.tools) > args.leaf_tool_number:
            self.messages = [
                {
                    "role": "system",
                    "content": CATEGORY_AGENT_PROMPT.replace('{category}', self.category)
                },
                {
                    "role": "user",
                    "content": f"Task description: {self.query}. All the tools: {self.tools}. Begin!"
                }
            ]
            logger.info(f"Category Agent prompt: {self.messages}")
        else:
            function_call_result = self.create_agent_tool_level(self.category, self.tools)
            return f'category: {self.category} assigned'
        global total_tokens, stop, error_flag, change_category
        for i in range(10):
            if change_category:
                change_category = False   
                print(colored(f'category: {self.category} assigned', 'green'))
                print(colored('category finish search', 'green'))
                logger.info(f"category: {self.category} assigned")
                logger.info("category finish search")
                self.messages.append(
                    {
                        "role": "function",
                        "name": 'Finish' ,
                        "content": 'Finished',
                    }
                )
                self.finish_search = True
                if multi_thread:
                    sem.release()
                    return f'category: {self.category} assigned.'
                else:
                    return f'category: {self.category} assigned. The status of current found apis is: {status}'
                        
            logger.info(f'The {i+1} loop of the {self.category} category_search function: ')
            if stop or total_tokens > 200000:
                logger.info(f"stop: {stop}")
                logger.info(f"total_tokens: {total_tokens}")
                if multi_thread:
                    sem.release()
                return f'category: {self.category} assigned'
            try:
                response = call_gpt(
                                messages=self.messages,
                                model=model_name,
                                functions=self.functions
                            )
            except:
                error_flag = True
                stop = True
                continue
            if isinstance(response, str):
                print('Response is str. response:', response)
                logger.info(f'Response is str. response: {response}')
                continue
            with counter_lock:
                total_tokens += response.usage.total_tokens
            print('#'*100)
            print(colored(f'The {i+1} loop of the {self.category} category_search function: ','yellow'))
            if(response.choices[0].message.content):
                print('Thought:', response.choices[0].message.content)
                message = {
                    "role": "assistant",
                    "content": response.choices[0].message.content if response.choices[0].message.content is not None else '',
                }
                self.messages.append(message)
                logger.info(message)
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                # print('tool call number', len(tool_calls))
                logger.info(f'tool call number: {len(tool_calls)}')
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    args_dict = json.loads(function_args)
                    tools = args_dict.get("tools", [])

                    if function_name.lower() == 'finish':
                        print(colored(f'category: {self.category} assigned', 'green'))
                        print(colored('category finish search', 'green'))
                        logger.info(f"category: {self.category} assigned")
                        logger.info("category finish search")
                        self.messages.append(
                            {
                                "role": "function",
                                "name": function_name,
                                "content": f'Finished category {self.category} search.'
                            }
                        )
                        self.finish_search = True
                        if multi_thread:
                            sem.release()
                            return f'category: {self.category} assigned.'
                        else:
                            return f'category: {self.category} assigned. The status of current found apis is: {status}'

                    elif function_name not in self.api_mapping:
                        function_name = 'hullucinating_function_name'
                        function_call_result = "Function name error"
                        logger.info("hullucinating_function_name")
                    elif function_name == "create_agent_tool_level":
                        print(colored('create_agent_tool_level', 'green')) 
                        logger.info("creating agent tool level")
                        try:
                            # print(tools)
                            logger.info(f"tools: {tools}")
                            function_call_result = self.create_agent_tool_level(**json.loads(function_args))
                        except Exception as e:
                            print(e, function_name, function_args, file=open(f'{args.output_dir}/error.txt', 'a', encoding='utf-8'))
                            function_call_result = str(e)
                    elif function_name == "get_tools_descriptions":
                        print(colored('getting tools descriptions', 'green')) 
                        logger.info("get tools descriptions")
                        function_call_result = get_tools_descriptions(**json.loads(function_args))
                    
                    message = {
                                "role":"function",
                                "name": function_name,
                                "param": function_args,
                                "content": str(function_call_result),
                            }
                    self.messages.append(message)
                    print(json.dumps(message, ensure_ascii=False, indent=2))
                    logger.info(json.dumps(message, ensure_ascii=False, indent=2))                   
            else:
                message = {
                    'role': "user",
                    'content': 'Please invoke the list of tools I provided for you. Do not just think about but not act.'
                }
                self.messages.append(message)
                logger.info(message)
                message = {
                    'role': "user",
                    'content': 'Please strictly follow the list of tools I provided for you to invoke. Do not just think about it but fail to act.'
                }
                self.messages.append(message)
                logger.info(message)   

        print(colored(f'category: {self.category} assigned', 'green'))
        logger.info(f'category: {self.category} assigned')
        self.finish_search = True
        if multi_thread:
            sem.release()
            return f'category: {self.category} assigned.'
        else:
            return f'category: {self.category} assigned. The status of current found apis is: {status}'
    
    def create_agent_tool_level(self, category: str, tools):
        if isinstance(tools, str):
            tools = eval(tools)
        illegal_tools = []
        for tool in tools:
            if tool not in self.tools:
                illegal_tools.append(tool)
        if len(illegal_tools) > 0:
            content = f'Illegal tools: {illegal_tools} is not in the {self.category} category. If you think that the tools you have assigned can already solve the requirements of the {self.category} part in the query, please call the Finish function to end the category_search and proceed with subsequent tasks.'
            print(colored(content, 'red'))
            logger.info(content)
            return content
        if len(tools) > args.leaf_tool_number:
            return f'Tool number should not exceed the max tool number of {args.leaf_tool_number}. Please assign again'
        
        global agents, index, threads
        with counter_lock:
            agents.append(Tool_Agent(self.query, category, tools))
            agents[-1].depth = self.depth + 1
            index += 1
            agents[-1].index = index
        self.sub_agents.append(agents[-1])
        if multi_thread:
            thread = threading.Thread(target=agents[-1].tool_search)
            sem.acquire()
            thread.start()
            with counter_lock:
                threads.append(thread)
            result = f'tools {tools} assigned.'
        else:
            agents[-1].tool_search()
            result = f'tools {tools} assigned. The status of current found apis is: {status}'

        return result

    def resume_search(self):
        global total_tokens, stop, error_flag
        if stop or total_tokens > 200000: 
            logger.info(f"stop: {stop}")
            logger.info(f"total_tokens: {total_tokens}")
            if multi_thread:
                sem.release()
            self.finish_search = True
            return f'category: {self.category} assigned'
        print(colored(f'assigning category: {self.category}', 'green'))
        logger.info(f'assigning category: {self.category}')
        if len(self.tools) <= args.leaf_tool_number:
            self.finish_search = True
            return f'category: {self.category} assigned'
        if self.failed_reason is not None:
            self.messages.append({"role": "user", "content": REFIND_TOOL_PROMPT.replace('{failed_reason}', str(self.failed_reason))})
            self.failed_reason = None
        for i in range(10):
            if stop or total_tokens > 200000:
                logger.info(f"stop: {stop}")
                logger.info(f"total_tokens: {total_tokens}")
                if multi_thread:
                    sem.release()
                return f'category: {self.category} assigned'
            try:
                response = call_gpt(
                                messages=self.messages,
                                model=model_name,
                                functions=self.functions
                            )
            except:
                error_flag = True
                stop = True
                continue
            with counter_lock:
                total_tokens += response.usage.total_tokens
            if isinstance(response, str):
                print('Response is str. response:', response)
                logger.info(f'Response is str. response: {response}')
                continue
            print('#'*100)
            if(response.choices[0].message.content):
                print('Thought:', response.choices[0].message.content)
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content if response.choices[0].message.content is not None else ''
                    }
                )
                logger.info(f'"role": "assistant", "content": {response.choices[0].message.content}')

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                # print('tool call number', len(tool_calls))
                logger.info(f"tool call number: {len(tool_calls)}")
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    if function_name.lower() == 'finish':
                        print(colored(f'category: {self.category} assigned', 'green'))
                        print(colored('category finish search', 'green'))
                        logger.info(f"category: {self.category} assigned")
                        logger.info("category finish search")
                        self.messages.append(
                            {
                                "role": "function",
                                "name": function_name,
                                "content": f'Finished category {self.category} search.'
                            })
                        self.finish_search = True
                        if multi_thread:
                            sem.release()
                            return f'category: {self.category} assigned.'
                        else:
                            return f'category: {self.category} assigned. The status of current found apis is: {status}'
                    elif function_name not in self.api_mapping:
                        function_name = 'hullucinating_function_name'
                        function_call_result = "Function name error"
                        logger.info(f"function response: hullucinating_function_name")
                    else:
                        try:
                            function_call_result = self.api_mapping[function_name](**json.loads(function_args))
                        except Exception as e:
                            print(e, function_name, function_args, file=open(f'{args.output_dir}/error.txt', 'a', encoding='utf-8'))
                            function_call_result = str(e)

                    message = {
                                "role":"function",
                                "name": function_name,
                                "param": function_args,
                                "content": str(function_call_result),
                            }
                    self.messages.append(message)
                    print(json.dumps(message, ensure_ascii=False, indent=2))
                    logger.info(json.dumps(message, ensure_ascii=False, indent=2))    
                    self.messages.append({'role': "user",
                                          'content': 'At each step,  you should call a function to actually excute your step.'})  
            else:
                message = {'role': "user",
                           'content': 'Please strictly follow the list of tools I provided for you to invoke. Do not just think about it but fail to act.'}
                self.messages.append(message)
                logger.info(message)

        print(colored(f'category: {self.category} assigned', 'green'))
        logger.info(f'category: {self.category} assigned')
        self.finish_search = True
        if multi_thread:
            sem.release()
            return f'category: {self.category} assigned.'
        else:
            return f'category: {self.category} assigned. The status of current found apis is: {status}' 


class Tool_Agent(Agent):
    def __init__(self, query, category=None, tools=None) -> None:
        super().__init__()
        global change_category
        change_category = False
        self.category = category
        if isinstance(tools, str):
            tools = eval(tools)
        self.tools = tools
        self.functions = [
            finish_function,
            add_apis_into_api_pool_function
        ]
        self.api_mapping = {
            "query_all_categories": query_all_categories,
            "get_tools_in_category": get_tools_in_category,
            "add_apis_into_api_pool": self.add_apis_into_api_pool,
            "Finish": Finish                
        }
        self.query = query

        if len(tools) > args.leaf_tool_number:
            logger.info(f"you should assign less than {args.leaf_tool_number} tools each time")
            return f"you should assign less than {args.leaf_tool_number} tools each time"
        else:
            tools_info = query_all_tool_info(category, tools)
            self.messages = [
                {
                    "role": "system",
                    "content": TOOL_AGENT_PROMPT.replace('{category}', str(category)).replace('{tools}', str(tools))
                },
                {
                    "role": "user",
                    "content": f"Task description: {self.query} All the tool description and the contained api_list as a dict: {tools_info}. Begin!"
                }
            ]
            logger.info(f"Tool Agent Prompt: {self.messages}")

    def remove_apis(self, api_list):
        """remove apis from the current available api list. required input to be list of dictionaries describing with the keys category_name, tool_name, api_name"""
        print(colored(f'removing apis: {api_list}', 'red'))
        logger.info(f'removing apis: {api_list}')
        if isinstance(api_list, str):
            api_list = eval(api_list)
        if not isinstance(api_list, list) or any('category_name' not in ele or 'tool_name' not in ele or 'api_name' not in ele for ele in api_list):
            return 'illegal input, input should be list, each element in the list should have category_name, tool_name, api_name'
        if not all([isinstance(ele['category_name'],str) and isinstance(ele['tool_name'],str) and isinstance(ele['api_name'],str) for ele in api_list]):
            return 'illegal input, category_name, tool_name, api_name should be string'
        origin_api_list = deepcopy(api_list)
        global global_api_list, global_api_list_detailed
        for api in api_list:
            tool_details = get_tool_description(self.category, api['tool_name'])
            api_details = get_api_details(**api)
            api['tool_description'] = tool_details['tool_description'] if isinstance(tool_details, dict) else ''
            api['api_description'] = api_details['description'] if 'description' in api_details else ''
            try:
                with counter_lock:
                    if api in global_api_list:
                        global_api_list.remove(api)
            except:
                pass

        for api in origin_api_list:
            for ele in global_api_list:
                if ele['category_name'] == api['category_name'] and ele['tool_name'] == api['tool_name'] and ele['api_name'] == api['api_name']:
                    with counter_lock:
                        global_api_list.remove(ele)
                    break
        return f'apis removed successfully. Current api number: {len(global_api_list)}. Max api number: {args.max_api_number}'

    def add_apis_into_api_pool(self, api_list):
        """add apis to the current available api list. required input to be list of dictionaries describing with the keys category_name, tool_name, api_name"""
        global change_category, global_api_list, global_api_list_detailed, stop, status, total_tokens
        print(colored(f'adding apis: {api_list}', 'red'))
        if len(global_api_list) + len(api_list) > args.max_api_number:
            message = f'API number exceeds the max API number of {args.max_api_number}, current API number: {len(global_api_list)}, number of APIs to be added: {len(api_list)}. Please reduce the APIs to be added.'
            logger.info(message)
            return message
        if isinstance(api_list, str):
            api_list = eval(api_list)
        if not isinstance(api_list, list) or any('category_name' not in ele or 'tool_name' not in ele or 'api_name' not in ele for ele in api_list):
            message = 'illegal input, input should be list, each element in the list should have category_name, tool_name, api_name'
            logger.info(message)
            return message
        if not all([isinstance(ele['category_name'],str) and isinstance(ele['tool_name'],str) and isinstance(ele['api_name'],str) for ele in api_list]):
            message = 'illegal input, category_name, tool_name, api_name should be string'
            logger.info(message)
            return message

        for api in api_list:
            tool_details = get_tool_description(self.category, api['tool_name'])
            if tool_details == 'tool name not found':
                continue
            if api not in global_api_list:
                global_api_list.append(deepcopy(api))
            api_details = get_api_details(**api)
            # 目前是只添加api的描述信息，没有工具的描述信息，可以加，也可以加api属于哪个工具即可
            api['api_description'] = api_details['description'] if 'description' in api_details else ''
            if api not in global_api_list_detailed:
                global_api_list_detailed.append(api)
        if not stop:
            logger.info("checking task solvable by function")
            solvable, reason, tokens = check_task_solvable_by_function_no_category(self.query, global_api_list_detailed, logger)
            self.messages.append(
                {
                    "role": "assistant",
                    "content": f"Status of the query is {solvable}, {reason}"
                }
            )
            total_tokens += tokens
            if solvable == 'Solvable':
                stop = True
                status = 'The current api list can solve the query.'
                return f'APIs added. Current API number: {len(global_api_list)}. Max API number: {args.max_api_number}'
            else:
                categoty_tools = get_tools_in_category(str(self.category))
                change_cate =[
                    {
                        "role": "assistant",
                        "content": IF_CHANGE_CATEGORY_PROMPT.replace('{category}', str(self.category)).replace('{query}', str(self.query)).replace('{reason}', str(reason)).replace('{global_api_list_detailed}', str(global_api_list_detailed)).replace('{categoty_tools}', str(categoty_tools))
                    }
                ]
                logger.info(f"Checking if change category: {change_cate}")
                response = call_gpt_no_func(messages=change_cate)
                content_str = response.choices[0].message.content
                if(content_str=='new'):
                    change_category = True
                    logger.info("change new category")
                else:
                    logger.info("continue old category")
                status = f'The current API list cannot solve the query due to the following reason: {reason}'
                if len(global_api_list) >= args.max_api_number:
                    stop = True
                return f'APIs added. Current API number: {len(global_api_list)}. Max API number: {args.max_api_number}, Status of the query is {solvable}, {reason}'
        return f'APIs added. Current API number: {len(global_api_list)}. Max API number: {args.max_api_number}, Status of the query is {solvable}, {reason}'

    def tool_search(self):
        global stop, total_tokens, error_flag, change_category
        print(colored(f'assigning tools: {self.tools} in category: {self.category}', 'blue'))
        logger.info(f"assigning tools: {self.tools} in category: {self.category}")
        for i in range(10):
            if change_category:
                function_name = 'finish'           
                self.messages.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": 'Finished',
                    }
                )
                print(f'tools {self.tools} assigned')
                logger.info(f'tools {self.tools} assigned')
                print(colored('tool finish search', 'green'))
                logger.info('tool finish search')
                if multi_thread:
                    sem.release()
                self.finish_search = True
                return f'tools {self.tools} assigned, The status of current found apis is: {status}'
            if self.finish_search:
                break
            if stop or total_tokens > 200000:
                logger.info(f"stop: {stop}")
                logger.info(f"total_tokens: {total_tokens}")
                print('#'*100)
                print(colored('stop', 'red'))
                return f'tools {self.tools} assigned'
            try:
                response = call_gpt(
                    messages=self.messages,
                    model=model_name,
                    functions=self.functions
                )
            except:
                error_flag = True
                stop = True
                continue
            if isinstance(response, str):
                continue
            with counter_lock:
                total_tokens += response.usage.total_tokens
            print('#'*100)
            print(colored(f'The {i+1} loop of tool_search:','yellow'))
            if(response.choices[0].message.content):
                print('Thought:', response.choices[0].message.content)
                message = {
                    "role": "assistant",
                    "content": response.choices[0].message.content if response.choices[0].message.content is not None else '',
                }
                self.messages.append(message)
                logger.info(message)

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                # print('tool call number', len(tool_calls))
                logger.info(f"tool call number: {len(tool_calls)}")
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
            
                    if function_name.lower() == 'finish':
                        print(f'tools {self.tools} assigned')
                        logger.info(f'tools {self.tools} assigned')
                        print(colored('tool finish search', 'green'))
                        logger.info('tool finish search')
                        self.finish_search = True
                        self.messages.append(
                                {
                                    "role":"function",
                                    "name": function_name,
                                    "content": 'Finished'
                                }
                            )   
                        if multi_thread:
                            sem.release()
                            return f'tools {self.tools} assigned'
                        else:
                            return f'tools {self.tools} assigned. The status of current found apis is: {status}'
                    elif function_name not in self.api_mapping:
                        function_name = 'hullucinating_function_name'
                        function_call_result = "Function name error"
                        logger.info("hullucinating_function_name")
                    elif function_name == 'add_apis_into_api_pool':
                        with counter_lock:
                            try:
                                function_call_result = self.api_mapping[function_name](**json.loads(function_args))
                            except Exception as e:
                                print(e, function_name, function_args, file=open(f'{args.output_dir}/error.txt', 'a', encoding='utf-8'))
                                function_call_result = 'input format error'
                    message = {
                                "role":"function",
                                "name": function_name,
                                "param": function_args,
                                "content": str(function_call_result),
                            }
                    self.messages.append(message)
                    print(json.dumps(message, ensure_ascii=False, indent=2))
                    logger.info(json.dumps(message, ensure_ascii=False, indent=2))  
            else:
                message = {
                    'role': "user",
                    'content': 'Please strictly follow the list of tools I provided for you to invoke. Do not just think about it but fail to act.'
                }
                self.messages.append(message)
                logger.info(message)        
        print(f'tools {self.tools} assigned')
        logger.info(f'tools {self.tools} assigned')
        self.finish_search = True
        if multi_thread:
            sem.release()
            return f'tools {self.tools} assigned'
        else:
            return f'tools {self.tools} assigned. The status of current found apis is: {status}'

    def resume_search(self):
        if stop or total_tokens > 200000: 
            logger.info(f"stop: {stop}")
            logger.info(f"total_tokens: {total_tokens}")
            if multi_thread:
                sem.release()
            self.finish_search = True
            print(f'tools {self.tools} assigned')
            return f'tools {self.tools} assigned'
        if self.failed_reason is not None:
            if len(self.tools) > leaf_tool_number:
                self.messages.append({"role": "user", "content": REFIND_TOOL_PROMPT.replace('{failed_reason}', str(self.failed_reason))})
            else:
                self.messages.append({"role": "user", "content": REFIND_API_PROMPT.replace('{failed_reason}', str(self.failed_reason))})
            self.failed_reason = None
        return self.tool_search()


def init_logger(task_id: str, log_dir: str):
    logger = logging.getLogger(f'task_{task_id}')
    logger.setLevel(logging.INFO)
    log_filename = f"{log_dir}/task_{task_id}.log"

    if not logger.hasHandlers():
        handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        formatter = logging.Formatter(
            fmt='[%(asctime)s][%(levelname)s][%(threadName)s][%(message)s]',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)    
    return logger

def save_answer(args, query_data, action_param_json):
    with open(args.answer_path, 'r', encoding='utf-8') as f:
        query_data_all = json.load(f)
    updated_dataset = []
    for item in query_data_all:
        if item.get("query_id") == query_data['query_id']:
            new_item = OrderedDict()
            for k, v in item.items():
                new_item[k] = v
                if k == "query_id":
                    new_item["final_answer"] = action_param_json["final_answer"]
            updated_dataset.append(new_item)
        else:
            updated_dataset.append(item)
    with open(args.query_path, 'w', encoding='utf-8') as f:
        json.dump(updated_dataset, f, ensure_ascii=False, indent=2)

    print(f"Final Answer: {action_param_json.get('final_answer')}")
    logger.info(f"Final Answer: {action_param_json.get('final_answer')}")

def api_retriever(query):
    global global_api_list, agents, threads, error_flag, retrieval_api_time
    print(colored('Retrieving API...', 'green'))
    t_s = time.time()
    runner = Main_Search_Agent(query)
    agents.append(runner)
    if multi_thread:
        thread = threading.Thread(target=runner.assign_main, args=())
        sem.acquire()
        thread.start()
        threads.append(thread)
    else:
        runner.assign_main()

    if multi_thread:
        for thread in threads:
            thread.join()
        threads.clear()
        if error_flag: 
            raise Exception('GPT Call Error')
    
    retrieval_api_time += (time.time() - t_s)
    tools_des = prune_api()
    
    return tools_des
    
def prune_api():
    global global_api_list, agents, global_api_list_detailed
    tools_des, to_prune = get_tools_des(global_api_list)
    if to_prune:
        print(colored(f'no tool description{to_prune}', 'red'))
        logger.info(f'no tool description{to_prune}')
        global_api_list = [cont for cont in global_api_list if cont["tool_name"] not in to_prune]
    
    print("The number of retrieval agents: "+str(len(agents)))
    # print("Total API list size: "+str(len(global_api_list)))
    print("Retrieval API list: "+str(global_api_list))
    logger.info(f"The number of retrieval agents: {str(len(agents))}\n"+ 
                f"Total API list size: {str(len(global_api_list))}\n"+
                'All API list detailed: ' + json.dumps(global_api_list_detailed, ensure_ascii=False, indent=2))   
    
    return tools_des
    
def run_lats(args, query_data, i, tools_des, experience):
    global global_api_list
    selected_api_list = deepcopy(global_api_list)
    t_s = time.time()
    state, tokens_len = lats_search(args, query_data, selected_api_list, i, tools_des, experience)
    
    with open("/root/autodl-tmp/ToolCombine/ToolsTree0307/answer_query/toolcombinebench_test.txt", 'a', encoding='utf-8') as f:
        print(f'solve time:\t{time.time() - t_s:.4f}s', file=f)
    result = "Unsucessful question" if state is None else state['action']
    logger.info(f'result:{result}')
    action_name = result.split('{')[0] if '{' in result else result
    if action_name.startswith('functions.'):
        action_name = action_name[10:]
    action_param = result[result.find('{'):].strip() if '{' in result else ""
    action_param_json = json.loads(action_param)

    if action_name.lower() == 'finish':
        if isinstance(action_param_json, dict):
            if action_param_json.get('final_answer'):
                save_answer(args, query_data, action_param_json)
        else:
            logger.info(f"The answer is not a dict: {action_param_json}")

    return action_name, action_param_json, tokens_len, state

def run_dfs(args, query_data, tools_des, i, experience):
    global global_api_list
    selected_api_list = deepcopy(global_api_list)
    t_s = time.time()
    solved, solve_data, tokens_len, output = dfs_search(args, query_data, selected_api_list, tools_des, i, experience)
    with open("/root/autodl-tmp/ToolCombine/ToolsTree0307/answer_query/toolcombinebench_test.txt", 'a', encoding='utf-8') as f:
        print(f'solve time:\t{time.time() - t_s:.4f}s', file=f)
    print("solved: ", solved, " result:", solve_data)
    action_param_json = solve_data
    if action_param_json['return_type'] == 'give_answer':
        save_answer(args, query_data, action_param_json)
    action_name = 'Finish'
    return action_name, action_param_json, tokens_len, output

def run_react(args, query_data, tools_des, i, experience):
    global global_api_list
    selected_api_list = deepcopy(global_api_list)
    t_s = time.time()
    output, solve_tokens = react_search(args, query_data, selected_api_list, tools_des, i, experience)
    with open("/root/autodl-tmp/ToolCombine/ToolsTree0307/answer_query/toolcombinebench_test.txt", 'a', encoding='utf-8') as f:
        print(f'solve time:\t{time.time() - t_s:.4f}s', file=f)
    action_match = re.search(r'Action:\s*(\w+)', output)
    action_input_match = re.search(r'Action Input:\s*({.*})', output, re.DOTALL)
    action_name = action_match.group(1).strip().lower() if action_match else None
    action_param = action_input_match.group(1).strip() if action_input_match else None
    action_param_json = {}
    try:
        action_param_json = json.loads(action_param)
    except Exception as e:
        logger.error(f"[run_react] Failed to parse action input JSON: {e}")
        logger.error(f"Raw action_param: {action_param}")
        return action_name, {} , solve_tokens, output 
    action_param_str = json.dumps(action_param_json, ensure_ascii=False, indent=2)
    result = f"{action_name}({action_param_str})"
    logger.info(f'result:{result}')
    if action_name.startswith('functions.'):
        action_name = action_name[10:]
    if action_param_json.get('final_answer'):
        save_answer(args, query_data, action_param_json)

    return action_name, action_param_json, solve_tokens, output
    
def reflection_and_resume_loop(new_exp, search_func):
    global agents, global_api_list, global_api_list_detailed, total_tokens, stop, failed_reason, query_data, query
    t_start = time.time()
    while not all([agent.finish_search for agent in agents]):
        logger.info('return_type: restart')
        logger.info(f"reason: {failed_reason}")
        print(colored("Reflection and resume loop", 'green'))
        max_depth = max([agent.depth for agent in agents])
        depth = max_depth
        while all([agent.finish_search for agent in agents if agent.depth == depth]) and depth >= 0:
            depth -= 1
        if depth < 0 or total_tokens > 200000:
            break
        stop = False
        removed_at_least_one = False
        agents_to_resume = [agent for agent in agents if not agent.finish_search and agent.depth == depth]
        apis_to_remove = [api for api in global_api_list if api['api_name'] in str(failed_reason)]
        for api in apis_to_remove:
            removed_at_least_one = True
            print(colored(f'removing api: {api}', 'red'))
            global_api_list.remove(api)
            global_api_list_detailed.remove(api)
    
        if removed_at_least_one:
            print(f'APIs removed successfully. Current API number: {len(global_api_list)}. Max API number: {args.max_api_number}')
            logger.info(f'APIs removed successfully. Current API number: {len(global_api_list)}. Max API number: {args.max_api_number}')

        print(colored('Refind Begin', 'red'))
        logger.info('Refind Begin')
        logger.info(f'The number of agents to resume: {len(agents_to_resume)}')
        print(colored(agents_to_resume, 'red'))
        logger.info(f'{agents_to_resume}')

        for agent in reversed(agents_to_resume):
            if agent.finish_search: continue
            agent.failed_reason = str(failed_reason)
            print(colored(('resuming', agent, agent.depth), 'red'))
            logger.info(f"resuming: {agent}, {agent.depth}")
            print(colored(('resuming', agent, agent.depth), 'red'), file=open(f'{args.output_dir}/resume.txt', 'a', encoding='utf-8'))
            agent.resume_search()
        if not stop:
            _, status, _ = check_task_solvable_by_function_no_category(query, global_api_list_detailed, logger)
            print(colored(f'status:{status}', 'red'))

        if stop or all([agent.finish_search for agent in agents]) and len(global_api_list) > 0:
            prune_api()
            # action_name, action_param_json = search_func(args, query_data, tools_des, i, new_exp)
            action_name, action_param_json, tokens_len, output = search_func(args, query_data, tools_des, i, new_exp)

            if action_name.lower() == 'finish':
                if action_param_json['return_type'] == 'give_answer':
                    update_success_cnt(action_param_json, case=1)
                    break
                elif action_param_json['return_type'] == 'give_up':
                    update_success_cnt(action_param_json, case=2)
                    break
                elif action_param_json['return_type'] == 'give_up_and_restart':
                    failed_reason = action_param_json['reason']
                    if all([agent.finish_search for agent in agents]):
                        update_success_cnt(action_param_json, case=2)
                else:
                    update_success_cnt(action_param_json, case=2)
                    break
            else:
                update_success_cnt(action_param_json, case=3)
                break

    print(f'reflection time:\t{time.time() - t_start:.4f}', file=open("/root/autodl-tmp/ToolCombine/ToolsTree0307/answer_query/toolcombinebench_test.txt", 'a', encoding='utf-8')) 
    return tokens_len, output

def update_success_cnt(action_param_json, case=2):
    global query_data, unsolvable_task_cnt, success_task_cnt, valid_task_cnt, total_cnt
    if case==1:
        success_task_cnt += 1
        log_msg = (f'Solved, id: {query_data["query_id"]} '
                f'unsolvable_cnt: {unsolvable_task_cnt} success_task_cnt: {success_task_cnt} '
                f'valid_task_cnt: {valid_task_cnt} total_cnt: {total_cnt} '
                f'success_task_cnt/valid_task_cnt: {success_task_cnt/valid_task_cnt}')
        with open(f'{args.output_dir}/success_cnt.txt', 'a', encoding='utf-8') as f:
            print(log_msg, file=f)      
    elif case==2:
        unsolvable_task_cnt += 1
        log_msg = (f'Unsolved, id: {query_data["query_id"]} reason: {action_param_json["reason"]} '
                f'unsolvable_cnt: {unsolvable_task_cnt} success_task_cnt: {success_task_cnt} '
                f'valid_task_cnt: {valid_task_cnt} total_cnt: {total_cnt} '
                f'success_task_cnt/valid_task_cnt: {success_task_cnt/valid_task_cnt}')
        with open(f'{args.output_dir}/success_cnt.txt', 'a', encoding='utf-8') as f:
            print(log_msg, file=f)
        print(colored(f'Unsolved, reason: {action_param_json["reason"]}','red'))
        logger.info(f"reason: {action_param_json['reason']}")
    elif case==3:
        unsolvable_task_cnt += 1
        log_msg = (f'Unsolved, id: {query_data["query_id"]} reason: The number of available steps is insufficient. '
                f'unsolvable_cnt: {unsolvable_task_cnt} success_task_cnt: {success_task_cnt} '
                f'valid_task_cnt: {valid_task_cnt} total_cnt: {total_cnt} '
                f'success_task_cnt/valid_task_cnt: {success_task_cnt/valid_task_cnt}')
        with open(f'{args.output_dir}/success_cnt.txt', 'a', encoding='utf-8') as f:
            print(log_msg, file=f)
        print(colored('Unsolved, the number of available steps is insufficient.','red'))
        logger.info('Unsolved, the number of available steps is insufficient.')
    elif case==4:
        unsolvable_task_cnt += 1
        log_msg = (f'Unsolved, id: {query_data["query_id"]} reason: The answer is not a dict. '
                f'unsolvable_cnt: {unsolvable_task_cnt} success_task_cnt: {success_task_cnt} '
                f'valid_task_cnt: {valid_task_cnt} total_cnt: {total_cnt} '
                f'success_task_cnt/valid_task_cnt: {success_task_cnt/valid_task_cnt}')
        with open(f'{args.output_dir}/success_cnt.txt', 'a', encoding='utf-8') as f:
            print(log_msg, file=f)
        print(colored('Unsolved, the answer is not a dict.','red'))
        logger.info(f"The answer is not a dict: {action_param_json}")
    elif case==5:
        unsolvable_task_cnt += 1
        ratio = 0 if valid_task_cnt == 0 else success_task_cnt / valid_task_cnt
        log_msg = (f'Unsolvable checked by human, id: {query_data["query_id"]} '
                    f'unsolvable_cnt: {unsolvable_task_cnt} success_task_cnt: {success_task_cnt} '
                    f'valid_task_cnt: {valid_task_cnt} total_cnt: {total_cnt} '
                    f'success_task_cnt/valid_task_cnt: {ratio:.4f}')
        with open(os.path.join(output_dir, 'success_cnt.txt'), 'a', encoding='utf-8') as f:
            print(log_msg, file=f)
        print(colored('Unsolvable checked by human', 'yellow'))
        logger.warning('Unsolvable checked by human')
    else:
        print(colored(f'Updated case error: {case}', 'red'))
   

if __name__ == '__main__':
    unsolvable_task_cnt = 0
    success_task_cnt = 0
    valid_task_cnt = 0
    total_cnt = args.task_end_index - args.task_start_index
    unsolvable_list = json.load(open('./data/abnormal_queries.json', 'r', encoding='utf-8'))
    with open(args.query_path, 'r', encoding='utf-8') as f:
        query_data_all = json.load(f)

    random.seed(42)
    random.shuffle(query_data_all)
    print(f"total tasks: {len(query_data_all)}")
    new_task_list = []
    for task in query_data_all:
        output_file_path = os.path.join(args.log,f"task_{task.get('query_id')}.log")
        if not os.path.exists(output_file_path):
            new_task_list.append(task)
    query_data_all = new_task_list
    print(f"undo tasks: {len(query_data_all)}") 

    for i, query_data in enumerate(query_data_all):
        agents = []
        global_api_list = []
        global_api_list_detailed = []  
        stop = False
        total_tokens = 0
        index = 0
        status = ''
        threads = []
        failed_reason = None
        error_flag = False
        experience = None
        change_category = False
        tools_des = []
        value_output = None
        retrieval_api_time = 0
        experience_hit = False

        print(colored(f"task {i}/{len(query_data_all)}  real_task_id_{query_data.get('query_id')}","blue"))
        logger = init_logger(query_data['query_id'], args.log)

        if query_data['query_id'] in unsolvable_list:
            update_success_cnt(query_data, case=5)
            continue

        valid_task_cnt += 1
        query = query_data['query']
        logger.info(f'Query: {query}')
        print(colored(f'Query: {query}', 'green'))

        total_time = time.time()
        t_s = time.time()
        experience, new_exp = get_similar_experiences(query)
        logger.info(f'Experience retrieval time: {time.time() - t_s:.2f}s')
        with open("/root/autodl-tmp/ToolCombine/ToolsTree0307/answer_query/toolcombinebench_test.txt", 'a', encoding='utf-8') as f:
            print(f'\nquery id:\t{query_data["query_id"]}', file=f)
            print(f'retrieval experience time:\t{time.time() - t_s:.4f}s', file=f)

        if experience:
            prompt_judge = JUDGE_USABILITY_PROMPT.replace('{current_query}', query).replace('{experience}', str(experience))
            # logger.info(f'Judge usability prompt: {prompt_judge}')
            response = gpt_for_data(model=model_name, prompt=prompt_judge, temperature=0).choices[0].message.content.strip().lower()
            logger.info(f'Judge response: {response}')
            if response == 'yes':
                print(colored(f'Experience {experience.get("experience_id")} accepted.', 'green'))
                exp_str = json.dumps(experience, ensure_ascii=False, indent=2)
                print(exp_str)
                logger.info(exp_str)
                try:
                    exp_tools = experience.get("tool_details")
                    exp_trajectory = experience.get("trajectory")
                    for api in exp_tools:
                        if api not in global_api_list:
                            global_api_list.append(deepcopy(api))
                        api_details = get_api_details(**api)
                        api['api_description'] = api_details['description'] if 'description' in api_details else ''
                        if api not in global_api_list_detailed:
                            global_api_list_detailed.append(api)

                    solvable, reason, sub_query, tokens = check_task_solvable_by_function_for_experience(query, global_api_list_detailed, logger)
                    total_tokens += tokens

                    if solvable == 'FullySolvable':
                        update_reuse_count_in_db(experience.get('experience_id'))
                        experience_hit = True
                        tools_des = prune_api()  

                    elif solvable == 'PartiallySolvable':
                        update_reuse_count_in_db(experience.get('experience_id'))
                        experience_hit = True
                        if(sub_query):
                            tools_des = api_retriever(sub_query)

                    else:
                        global_api_list = []
                        global_api_list_detailed = []                  
                        experience = None      
                    
                except Exception as e:
                    print(colored(e,'red'))
                    logger.info(f"{e}")
            else:
                print(colored('This experience is not applicable.', 'yellow'))
                experience = None

        if not experience:
            tools_des = api_retriever(query)

        algorithm = select_strategy(query, new_exp.subtasks, global_api_list, experience)
        print(colored(f"algorithm: {algorithm}",'blue'))
        logger.info(f"algorithm: {algorithm}")
        algorithm = 'react'

        if algorithm == 'lats':
            try:
                solve_tokens = 0
                new_exp_copy = None if experience else new_exp
                action_name, action_param_json, tokens_len, value_output = run_lats(args, query_data, i, tools_des, new_exp_copy) 
                solve_tokens += tokens_len
                if action_param_json.get('return_type') == 'give_answer':
                    update_success_cnt(action_param_json, case=1)
                else:
                    if experience_hit:
                        global_api_list = []
                        global_api_list_detailed = []
                        for agent in agents:
                            del agent
                        agents.clear()
                        stop = False
                        tools_des = api_retriever(query)  
                        action_name, action_param_json, tokens_len, value_output = run_lats(args, query_data, i, tools_des, new_exp)
                        solve_tokens += tokens_len
                        if action_name.lower() == 'finish':
                            if action_param_json['return_type'] == 'give_answer':
                                update_success_cnt(action_param_json, case=1)
                            elif action_param_json['return_type'] == 'give_up':
                                update_success_cnt(action_param_json, case=2)
                            elif action_param_json['return_type'] == 'give_up_and_restart':
                                failed_reason = action_param_json['reason']
                                tokens_len, value_output = reflection_and_resume_loop(new_exp, lats_search)
                                solve_tokens += tokens_len
                            else:
                                update_success_cnt(action_param_json, case=2)
                        else:
                            update_success_cnt(action_param_json, case=3)
                    else:
                        if action_param_json['return_type'] == 'give_up':
                            update_success_cnt(action_param_json, case=2)
                        elif action_param_json['return_type'] == 'give_up_and_restart':
                            failed_reason = action_param_json['reason']
                            tokens_len, value_output = reflection_and_resume_loop(new_exp, lats_search)
                            solve_tokens += tokens_len
                        else:
                            update_success_cnt(action_param_json, case=2)
            
            except Exception as e:
                traceback.print_exc()
                logger.info(f"{e}")


        elif algorithm == 'dfs':
            try:
                solve_tokens=0
                new_exp_copy = None if experience else new_exp
                _, action_param_json,tokens_len,value_output = run_dfs(args, query_data, tools_des, i, new_exp_copy)
                solve_tokens+=tokens_len
                if action_param_json.get('return_type') == 'give_answer':
                    update_success_cnt(action_param_json, case=1)
                else:
                    if experience_hit:
                        global_api_list = []
                        global_api_list_detailed = []
                        for agent in agents:
                            del agent
                        agents.clear()
                        stop = False
                        tools_des = api_retriever(query)  
                        _, action_param_json,tokens_len,value_output = run_dfs(args, query_data, tools_des, i, experience=new_exp)
                        solve_tokens+=tokens_len
                        if action_param_json['return_type'] == 'give_answer':
                            update_success_cnt(action_param_json, case=1)
                        elif action_param_json['return_type'] == 'give_up':
                            update_success_cnt(action_param_json, case=2)
                        elif action_param_json['return_type'] == 'give_up_and_restart':
                            failed_reason = action_param_json['reason']
                            tokens_len, value_output = reflection_and_resume_loop(new_exp, dfs_search)
                            solve_tokens+=tokens_len
                        else:
                            update_success_cnt(action_param_json, case=2)
                    else:
                        if action_param_json['return_type'] == 'give_up':
                            update_success_cnt(action_param_json, case=2)
                        elif action_param_json['return_type'] == 'give_up_and_restart':
                            failed_reason = action_param_json['reason']
                            tokens_len, value_output = reflection_and_resume_loop(new_exp, dfs_search)
                            solve_tokens+=tokens_len
                        else:
                            update_success_cnt(action_param_json, case=2)

            except Exception as e:
                traceback.print_exc()
                logger.info(f"{e}")


        elif algorithm == 'react':
            try:
                solve_tokens = 0
                new_exp_copy = None if experience else new_exp
                action_name, action_param_json, tokens_len, value_output = run_react(args, query_data, tools_des, i, new_exp_copy)
                solve_tokens += tokens_len
                if action_name.lower() == 'finish' and action_param_json.get('return_type') == 'give_answer':
                    update_success_cnt(action_param_json, case=1) 
                else:
                    if experience_hit:
                        global_api_list = []
                        global_api_list_detailed = []
                        for agent in agents:
                            del agent
                        agents.clear()
                        stop = False
                        tools_des = api_retriever(query)  
                        action_name, action_param_json, tokens_len, value_output = run_react(args, query_data, tools_des, i, experience=new_exp)
                        solve_tokens += tokens_len
                        if action_name == 'finish':
                            if isinstance(action_param_json, dict):
                                if action_param_json.get('return_type') == 'give_answer':
                                    update_success_cnt(action_param_json, case=1)
                                elif action_param_json['return_type'] == 'give_up':
                                    update_success_cnt(action_param_json, case=2)
                                elif action_param_json['return_type'] == 'give_up_and_restart':
                                    failed_reason = action_param_json['reason']
                                    tokens_len, value_output = reflection_and_resume_loop(new_exp, react_search)
                                    # reflection_and_resume_loop也应该返回最后的工具名，然后拿到一起打分
                                    solve_tokens += tokens_len
                            else:
                                update_success_cnt(action_param_json, case=4)
                        else:
                            update_success_cnt(action_param_json, case=3)
                    else:
                        if action_param_json['return_type'] == 'give_up':
                            update_success_cnt(action_param_json, case=2)
                        elif action_param_json['return_type'] == 'give_up_and_restart':
                            failed_reason = action_param_json['reason']
                            tokens_len, value_output = reflection_and_resume_loop(new_exp, react_search)
                            # reflection_and_resume_loop也应该返回最后的工具名，然后拿到一起打分
                            solve_tokens += tokens_len
                        else:
                            update_success_cnt(action_param_json, case=4)
            
            except Exception as e:
                traceback.print_exc()
                logger.info(f"{e}")


        else:
            raise Exception("Search algorithm option not valid")

        for agent in agents:
            del agent
        score = evaluate_completion_score(query, value_output)

        with open("/root/autodl-tmp/ToolCombine/ToolsTree0307/answer_query/toolcombinebench_test.txt", 'a', encoding='utf-8') as f:
            print(f'retrieval api time:\t{retrieval_api_time:.4f}s', file=f)
            print(f'total time:\t{time.time() - total_time:.4f}s', file=f)
            print(f'retrieval tokens:\t{total_tokens}', file=f)
            print(f'solve tokens:\t{solve_tokens}', file=f)
            print(f'total tokens:\t{total_tokens + solve_tokens}', file=f)
            print(f'score:\t{score}', file=f)

        print(colored("Next query.", 'green'))

