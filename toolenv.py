import json
import time
import gym
import os
from termcolor import colored
from copy import deepcopy
import requests
import re
from datetime import datetime
import time
from config import *
from function_database import finish_func


class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class ToolEnv(gym.Env):

  def __init__(self, query, api_list, tool_des, process_id=0):
    super().__init__()
    self.obs = "You can use the list of apis I gave you to solve your current problem.\n"  # current observation
    self.steps = 0  # current number of steps
    self.answer = None  # current answer from the agent
    self.observation_space = self.action_space = textSpace()
    self.search_time = 0
    self.num_searches = 0

    self.tool_root_dir = '/root/autodl-tmp/ToolCombine/tools'
    self.service_url = "http://0.0.0.0:8080/virtual"
    self.toolbench_key = toolbench_key
    self.max_observation_length = 4096
    self.observ_compress_method = 'truncate'
    self.converted_list = []
    # self.api_customization = args.api_customization
    self.process_id = process_id

    self.tool_names = []
    self.cate_names = []

    self.input_description = query
    self.functions = []
    self.api_name_reflect = {}
    
    self.data_dict = self.fetch_api_json(api_list)
    self.api2origin = {}
    origin_api_list = deepcopy(self.data_dict["api_list"])

    for k,api_json in enumerate(self.data_dict["api_list"]):
        standard_tool_name = tool_des[k][0]
        openai_function_json,cate_name, pure_api_name = self.api_json_to_openai_json(api_json,standard_tool_name)
        self.functions.append(openai_function_json)
        self.api_name_reflect[openai_function_json["name"]] = pure_api_name
        self.tool_names.append(standard_tool_name)
        self.cate_names.append(cate_name)
        self.api2origin[openai_function_json["name"]] = {'category_name': origin_api_list[k]['category_name'], 'tool_name': origin_api_list[k]['tool_name'], 'api_name': origin_api_list[k]['api_name']} 
    self.functions_names = [func["name"] for func in self.functions]
    self.functions.append(finish_func)
    self.converted_list = [self.convert_to_function_format(data) for data in self.functions]
    self.CALL_MAX_TIME = 3
    self.task_description = f'''You should use functions to help handle the real time user querys. Remember:
        1.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
        2.Do not use origin tool names, use only subfunctions' names.
        You have access of the following tools:\n'''
    
    unduplicated_reflection = {}
    for standardize_tool_name, tool_description in tool_des:
        unduplicated_reflection[standardize_tool_name] = tool_description

    for k,(standardize_tool_name, tool_des) in enumerate(unduplicated_reflection.items()):
        striped = tool_des[:512].replace('\n','').strip()
        if striped == "":
            striped = "None"
        self.task_description += f"{k+1}.{standardize_tool_name}: {striped}\n"

    self.success = 0


  def standardize(self, string):
      if not isinstance(string, str):
          print('*'*100)
          print(string)
      res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
      string = res.sub("_", string)
      string = re.sub(r"(_)\1+","_", string).lower()
      while True:
          if len(string) == 0:
              return string
          if string[0] == "_":
              string = string[1:]
          else:
              break
      while True:
          if len(string) == 0:
              return string
          if string[-1] == "_":
              string = string[:-1]
          else:
              break
      if string[0].isdigit():
          string = "get_" + string
      return string
  
  
  def fetch_api_json(self, api_list):
      data_dict = {"api_list":[]}
      for item in api_list:
          cate_name = item["category_name"]
          tool_name = self.standardize(item["tool_name"])
          api_name = self.change_name(self.standardize(item["api_name"]))
        
          try:
              tool_json = json.load(open(os.path.join(self.tool_root_dir, cate_name, tool_name + ".json"), "r"))
          except:
              print(self.tool_root_dir, cate_name, tool_name, "file is error")
              continue
          append_flag = False
          api_dict_names = []
          for api_dict in tool_json["api_list"]:
              api_dict_names.append(api_dict["name"])
              pure_api_name = self.change_name(self.standardize(api_dict["name"]))
              if pure_api_name != api_name:
                  continue
              api_json = {}
              api_json["category_name"] = cate_name
              api_json["tool_name"] = tool_json["tool_name"]
              api_json["api_name"] = api_dict["name"]
              api_json["api_description"] = api_dict["description"]
              api_json["required_parameters"] = api_dict["required_parameters"]
              api_json["optional_parameters"] = api_dict["optional_parameters"]
              data_dict["api_list"].append(api_json)
              append_flag = True
              break
          if not append_flag:
              print(api_name, api_dict_names)
      return data_dict
  
  
  def api_json_to_openai_json(self, api_json, standard_tool_name):
    description_max_length=256
    templete = {
        "name": "",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "required": [],
            "optional": [],
        }
    }
    
    map_type = {
        "NUMBER": "integer",
        "STRING": "string",
        "BOOLEAN": "boolean"
    }

    pure_api_name = self.change_name(self.standardize(api_json["api_name"]))
    templete["name"] = pure_api_name+ f"_for_{standard_tool_name}"
    # 这里是怕名字太长，我先删掉，后续需要再加上
    templete["name"] = templete["name"][-64:]

    # templete["description"] = f"This is the subfunction for tool \"{standard_tool_name}\", you can use this tool."
    
    # 检查是否为空字段
    if api_json["api_description"].strip() != "":
        tuncated_description = api_json['api_description'].strip().replace(api_json['api_name'],templete['name'])[:description_max_length]
        # templete["description"] = templete["description"] + f"The description of this function is: \"{tuncated_description}\""
        templete["description"] = f"{tuncated_description}"
    if "required_parameters" in api_json.keys() and len(api_json["required_parameters"]) > 0:
        for para in api_json["required_parameters"]:
            name = self.standardize(para["name"])
            name = self.change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else:
                param_type = "string"
            prompt = {
                "type":param_type,
                "description":para["description"][:description_max_length],
            }

            default_value = para['default']
            if len(str(default_value)) != 0:    
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                    "example_value": default_value
                }
            else:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length]
                }

            templete["parameters"]["properties"][name] = prompt
            templete["parameters"]["required"].append(name)
        for para in api_json["optional_parameters"]:
            name = self.standardize(para["name"])
            name = self.change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else:
                param_type = "string"

            default_value = para['default']
            if len(str(default_value)) != 0:    
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                    "example_value": default_value
                }
            else:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length]
                }

            templete["parameters"]["properties"][name] = prompt
            templete["parameters"]["optional"].append(name)

    return templete, api_json["category_name"],  pure_api_name
    

  def change_name(self, name):
      change_list = ["from", "class", "return", "false", "true", "id", "and"]
      if name in change_list:
          name = "is_" + name
      return name


  
  def step(self, action_name, action_input):
      obs, status, reward, done, info = self._step(action_name, action_input)
      if len(obs) > self.max_observation_length:
          obs = obs[:self.max_observation_length] + "..."
      return obs, status, reward, done, info


  def _step(self, action_name="", action_input=""):
    """Need to return an observation string and status code:
      0 means normal response
      1 means there is no corresponding api name
      2 means there is an error in the input
      3 represents the end of the generation and the final answer appears
      4 means that the model decides to pruning by itself
      5 represents api call timeout
      6 for 404
      7 means not subscribed
      8 represents unauthorized
      9 represents too many requests
      10 stands for rate limit
      11 message contains "error" field
      12 error sending request
    """
    reward = 0
    done = False
    self.steps += 1
    info = self._get_info
    
    if action_name == "Finish":
      print(colored(f"Finish, {action_input}",color="yellow"))
      timestamp = time.time()
      readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
      with open('output/finish.txt', 'a', encoding='utf-8') as f:
          print(action_input, readable_time, file=f)
      if isinstance(action_input, dict):
          answer = action_input
      else:
        try:
            answer = json.loads(action_input)
            if isinstance(answer, dict):
                answer = answer
            else:
                answer = json.loads(json.loads(action_input))
        except json.JSONDecodeError:
            # 有一种finish没有参数的情况要考虑到
            if action_input.count('}') - action_input.count('{') == 1:
                # 找到所有右括号的位置
                brace_indices = [m.start() for m in re.finditer(r'\}', action_input)]
                if len(brace_indices) >= 2:
                    # 删除倒数第二个 }
                    idx_to_remove = brace_indices[-2]
                    fixed_input = action_input[:idx_to_remove] + action_input[idx_to_remove + 1:]
                    try:
                        answer = json.loads(fixed_input)
                    except json.JSONDecodeError:
                        print("⚠ JSON 修复失败：", action_input)
                        print("Error: answer is not a valid JSON string")
                        answer = {}
            else:
                print("Error: answer is not a valid JSON string")
                answer = {}  # 或者根据需求处理错误情况
      done = True
      if "return_type" not in answer:
          answer["return_type"] = "give_answer"
      if answer["return_type"] == "give_answer" and answer["final_answer"] !='':
          reward = 1    
          self.answer = answer["final_answer"]
      elif answer["return_type"] == "give_up_and_restart":
          if answer["reason"] != '':
              self.answer = answer["reason"]
      elif answer["return_type"] == "give_up":
          if answer["reason"] != '' or answer["final_answer"] != '':
              if answer["reason"] != '':
                self.answer = answer["reason"]
              else:
                self.answer = answer["final_answer"]
      else:
          print("Return type error")
      self.obs = f"Episode finished, reward = {reward}\n"
      json_data = {}
      try:
          json_data = json.loads(action_input,strict=False)
          if 'reason' in json_data.keys():
              reason = json_data["reason"]
              print(json_data, file=open('output/reason.txt','a'))
      except Exception as e:    
          action_input = str(action_input)
          match = re.search(r'"return_type"\s*:\s*"([^"]+)"', action_input)
          if match:
              json_data["return_type"] = match.group(1).strip()
          match = re.search(r'"final_answer"\s*:\s*"([^"]+)"', action_input)
          if match:
              json_data["final_answer"] = match.group(1).strip()
          match = re.search(r'"reason"\s*:\s*"([^"]+)"', action_input)
          if match:
              json_data["reason"] = match.group(1).strip()
              with open('output/reason.txt', 'a', encoding='utf-8') as f:
                  print(json_data["reason"], file=f)

      if "return_type" not in json_data.keys():
          timestamp = time.time()
          readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
          print("query_id:", self.process_id, " time:", readable_time, json_data.keys(), file=open('output/error_return_type.txt','a'))
          return "{error:\"must have \"return_type\"\"}", 2, reward, done ,self._get_info()
      if json_data["return_type"] == "give_up_and_restart":
          return "{\"response\":\"chose to give up and restart\"}", 4, reward, done ,self._get_info()
      elif json_data["return_type"] == "give_up":
          if "reason" not in json_data.keys():
              return "{error:\"must have \"reason\"\"}", 2, reward, done ,self._get_info()
          return "{\"response\":\"chose to give up\"}", 3, reward, done ,self._get_info()
      elif json_data["return_type"] == "give_answer":
          if "final_answer" not in json_data.keys():
              return "{error:\"must have \"final_answer\"\"}", 2, reward, done ,self._get_info() 
          self.success = 1
          return "{\"response\":\"successfully giving the final answer.\"}", 3, reward, done ,self._get_info()
      else:
          return "{error:\"\"return_type\" is not a valid choice\"}", 2, reward, done ,self._get_info()
    else:
      for k, function in enumerate(self.functions):
        if function["name"].endswith(action_name):
          pure_api_name = self.api_name_reflect[function["name"]]
          payload = {
              "category": self.cate_names[k],
              "tool_name": self.tool_names[k],
              "api_name": pure_api_name,
              "tool_input": action_input,
              "strip": "",
              "toolbench_key": self.toolbench_key
          }
          print(colored(f"query to {self.cate_names[k]}-->{self.tool_names[k]}-->{pure_api_name}{action_input}",color="yellow"))
          headers = {"toolbench_key": self.toolbench_key}

          try:
            #   print("requesting")
              response = requests.post(self.service_url, json=payload, headers=headers, timeout=60)
              print(response.text)
          except:
              os.makedirs('output', exist_ok=True)
              print(payload, file=open('output/timeout.txt','a'))
              return json.dumps({"error": "connection timeout", "response": ""}), 13, reward, done ,self._get_info()
          if response.status_code != 200:
              return json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}), 12, reward, done ,self._get_info()
          try:
              response = response.json()
          except:
              print(response)
              return json.dumps({"error": f"request invalid, data error", "response": ""}), 12, reward, done ,self._get_info()
          if response["error"] == "API not working error...":
              status_code = 6
          elif response["error"] == "Unauthorized error...":
              status_code = 7
          elif response["error"] == "Unsubscribed error...":
              status_code = 8
          elif response["error"] == "Too many requests error...":
              status_code = 9
          elif response["error"] == "Rate limit per minute error...":
              print("Reach api calling limit per minute, sleeping...")
              time.sleep(10)
              status_code = 10
          elif response["error"] == "Message error...":
              status_code = 11
          else:
              status_code = 0
          return json.dumps(response), status_code, reward, done ,self._get_info()
      return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1, reward, done ,self._get_info()
    
    # elif action.startswith("think[") and action.endswith("]"):
    # self.obs = "Nice thought."
    # else:
    # self.obs = "Invalid action: {}".format(action)


  def check_success(self):
      return self.success


  def to_json(self):
      return {}


  def restart(self):
      pass


  def get_score(self):
      return 0.0
    
    
  def get_time_info(self):
    speed = self.search_time / self.num_searches if self.num_searches else 0
    return {
        "call_speed": speed,
        "call_time": self.search_time,
        "num_calls": self.num_searches,
    }
  

  def _get_obs(self):
    return self.obs


  def _get_info(self):
    return {"steps": self.steps, "answer": self.answer}


  def reset(self, seed=None, return_info=False, options=None):
    self.obs = ("You can use the list of apis I gave you to solve your current problem.\n")
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None
    self.steps = 0
    self.answer = None
    observation = self._get_obs()
    info = self._get_info()
    return (observation, info) if return_info else observation
  

  def convert_to_function_format(self, original_data):
    return {
        "type": "function",
        "function": {
            "name": original_data["name"],
            "description": original_data["description"],
            "parameters": original_data["parameters"]
        }
    }
  
  
