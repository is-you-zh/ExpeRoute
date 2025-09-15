from arguments import parse_args
args = parse_args()
leaf_tool_number = args.leaf_tool_number


reflection_prompt = '''
You are an advanced reasoning agent capable of improving based on self-reflection.  You will be given a previous reasoning trial in which you were provided with access to multiple tools, each with specific functionality, and a question to solve.  
Here is a list of tools and questions you can use:
{functions}
Question:
{question}
You may not be able to answer this question because::
1.You failed to correctly identify and combine the appropriate tools.
2.You have made incorrect assumptions about the parameters of the api.
3.You exceeded your set number of reasoning steps without finding a solution.
4.You provided an incorrect final answer.
Diagnose a possible reason for the failure, considering issues such as incorrect tool selection, suboptimal tool combinations, or insufficient reasoning steps.  Then, devise a new, concise, high-level plan that aims to mitigate the identified failure.  Your plan should outline strategies for better tool selection, efficient API usage, and improved reasoning steps.
Previous trial:
{trajectory}
Reflection:
'''


cot_prompt_feedback = '''
You are an advanced reasoning agent capable of solving complex tasks by using multiple tools(functions) in combination. Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps. 
You can utilize a variety of tools(functions), each serving specific purposes. 
Thought steps allow you to reason about the current situation, while Action steps involve invoking tools(functions) to retrieve or process information. Observation provides the result of the executed Action, which informs subsequent reasoning.
Here are the types of Action you can take:
{tool_des}
After each observation, provide the next Thought and next Action. 
You have attempted to answer the following question before and failed possibly because:
1.You failed to correctly identify and combine the appropriate tools.
2.You made incorrect assumptions about the inputs or outputs of the tools(functions).
3.You exceeded your set number of reasoning steps without finding a solution.
4.You provided an incorrect final answer.

The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.
{trajectories}
When providing the thought and action for the current trial, that into account these failed trajectories and make sure not to repeat the same mistakes and incorrect answers. 
{input}
'''


cot_prompt_feedback_short = '''
You are an advanced reasoning agent capable of solving complex tasks by using multiple APIs in combination. Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps. You can utilize a variety of APIs, each serving specific purposes. Thought steps allow you to reason about the current situation, while Action steps involve invoking APIs to retrieve or process information. Observation provides the result of the executed Action, which informs subsequent reasoning.
If you think you cannot finish the task with the current tools, you can call Finish function to restart and update api pool.
If you have already called an API before, there is no need to call it again.
Here are the types of Action you can take:
tools:
{tool_des}
After each observation, provide the next Thought and next Action. Here are some examples:


You have attempted to answer the following question before and failed because:
1.You failed to correctly identify and combine the appropriate tools.
2.You made incorrect assumptions about the inputs or outputs of the APIs.
3.You exceeded your set number of reasoning steps without finding a solution.
4.You provided an incorrect final answer.

The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.

{trajectories}
When providing the thought and action for the current trial, that into account these failed trajectories and make sure not to repeat the same mistakes and incorrect answers. 

{input}
'''


cot_prompt = '''
You are an advanced reasoning agent capable of solving complex tasks by using multiple apis in combination. Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps.
You can use many apis to do the following task. Note the description of each tool. Here are the apis you can use:
{functions}
After each observation, offer the next Thought and the next Action.
Thought can reason about the current situation and think about the next action, while Action steps involve invoking apis to retrieve or process information, Observation provides the result of the executed Action.
Note: You can only use one api at a time, output only the name of the api, if there are parameters, you can output api_name{{parameter}}
Remember, if you have used an API before, do not call it again with the same parameters, because the result of your repeated call will still be the same. You can try other APIs.
If the current tools cannot complete the task, call Finish function to restart and give the reason.
trajectory:
{input}
'''


cot_prompt_with_tool = '''
You are an advanced reasoning agent capable of solving complex tasks by using multiple apis in combination. Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps. If the tool doesn't solve the problem, you can try to answer it based on your own knowledge.
You can use many apis to do the following task. Note the description of each api.
After each observation, **offer the next Thought and the next Action**.
Thought can reason about the current situation and think about the next action, while Action steps involve invoking apis to retrieve or process information, Observation provides the result of the executed Action.

Important:
1. **Use only one api at a time**, output only the name of the api, if there are parameters, you can output api_name{{parameter}}
The parameters need to be in dictionary format, for example, {{"location":"beijing"}}
2. If you have already called an API with specific parameters, avoid calling it again with the same ones, as the result will not change.
3. If the tool does not return a direct answer, but clearly indicates that the tool can solve the user's problem, or it requires further actions from the user, this part of the task can also be considered completed.
4. If the current tools cannot complete the task, call Finish function to restart and give the reason.
5. If the user's query does not specify any parameters, to complete the task, you can decide to generate parameters to continue execution.

trajectory:
{input}
'''

# cot_prompt_with_tool = '''
# You are an advanced reasoning agent capable of solving complex tasks by using multiple apis in combination. Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps. If the tool doesn't solve the problem, you can try to answer it based on your own knowledge.
# You can use many apis to do the following task. Note the description of each api.
# After each observation, **offer the next Thought and the next Action**.
# Thought can reason about the current situation and think about the next action, while Action steps involve invoking apis to retrieve or process information, Observation provides the result of the executed Action.

# Important:
# 1. **Use only one api at a time**, output only the name of the api, if there are parameters, you can output api_name{{parameter}}
# The parameters need to be in dictionary format, for example, {{"location":"beijing"}}
# 2. If you have already called an API with specific parameters, avoid calling it again with the same ones, as the result will not change.
# 3. If the tool does not return a direct answer, but clearly indicates that the tool can solve the user's problem, or it requires further actions from the user, this part of the task can also be considered completed.
# 4. If you can solve part of the problem with your own knowledge, then that part doesn't require a tool call.
# 5. If the current tools cannot complete the task, call Finish function to restart and give the reason.

# trajectory:
# {input}
# '''
# For example:
# Thought 1: To create a fun and diverse quiz, I'll start by retrieving a trivia fact about a number to use as a question. The API v1_trivia_for_trivia_by_api_ninjas provides interesting number trivia.
# Action 1: v1_trivia_for_trivia_by_api_ninjas{{"number": "42"}}
# Thought 2: Great, I can turn this into a multiple-choice question. Now I'll fetch a historical event for the second question.
# Action 2: Functions.v1_historical_events_by_api_ninjas{{"date": "07-30"}}

cot_prompt_short = '''
Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps. You can utilize a variety of APIs, each serving specific purposes. Thought steps allow you to reason about the current situation, while Action steps involve invoking APIs to retrieve or process information. Observation provides the result of the executed Action, which informs subsequent reasoning.
Here are the types of Action you can take:
tools:
{tool_des}

{input}
'''


value_prompt_reasoning = '''
You are an advanced reasoning agent that can improve based on self refection. Analyze the trajectories of your previous solutions to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions.
Given a question and a trajectory, evaluate its correctness and you don't need to provide reasoning and analysis. Pay particular attention to the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Finally, you only need to output a number for the correctness score, which is an integer between 1 and 10.
{input}
'''


value_prompt_reasoning_together = '''
You are an advanced reasoning agent capable of evaluating strategies based on past experiences. Analyze the trajectories of your previous solutions to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions.
Given a question and multiple trajectories, evaluate its correctness and you don't need to provide reasoning and analysis. Pay particular attention to the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. 
Finally, you only need to output a set of numbers for the correctness score, which are some integers between 1 and 10. Such as[10, 9, 10, 7, 3]
Output as many numbers as there are child nodes.
{input}
'''


CHECK_SOLVABLE_BY_FUNCTION_PROMPT = """
Please evaluate whether the given query's {category}-related problems can be solved with the current available tools. Follow these rules:
1. If the `query` not contains any {category}-related content or you find that this type of tool is not the solution to the problems related to {category}: return "Irrelevant"
2. If the `query` provide invalid information (e.g. invalid email address or phone number), return "Unsolvable"
3. If the `query` needs more information to solve (e.g. the target restaurant name in a navigation task), return "Unsolvable"
4. If the existing tools cannot solve the problems related to "{category}" in the `query`, more tools are needed, return "Unsolvable"
5. If the current 'available_tools' are sufficient to solve the problems related to "{category}", return "Solvable".

Please return only two fields, "solvable" and "reason". 
Here are some examples: 
{"solvable": "Unsolvable", "reason": "The query requests booking a restaurant table but provides an invalid phone number for contact."}
{"solvable": "Solvable", "reason": "For the Movies category problem in this query (getting showtimes for 'The Dark Knight'), the current tools can handle this request."}
"""


CHECK_SOLVABLE_BY_FUNCTION_PROMPT_NO_CATEGORY = """
Please check whether the given task solvable with following rules:
1. If the `query` provide invalid information (e.g. invalid email address or phone number), return "Unsolvable"
2. If the `query` needs more information to solve (e.g. the target restaurant name in a navigation task), return "Unsolvable"
3. If the `query` is not enough for existing tools and requires more other category tools to solve, return "Unsolvable".
4. If the currently `available_tools` are enough to solve the entire query, return "Solvable"

Please return only two fields, "solvable" and "reason". 
Here are some examples: 
{"solvable": "Unsolvable", "reason": "The query involves retrieving detailed movie information and current timezone information. The tools available can provide movie information, but there isn't a tool available to get current timezone information for Los Angeles."}
{"solvable": "Solvable", "reason": "The query involves retrieving detailed movie information and current timezone information. The available tools provide sufficient resources to gather detailed information about 'The Incredible Hulk' movie and to find the current timezone for Los Angeles, USA. Now I can finish the entire query now."}
"""


CHECK_SOLVABLE_BY_FUNCTION_PROMPT_FOR_EXPERIENCE = """
Your responsibility is to assess whether the given `available_tools` can handle the `query`. 
Please evaluate with these three possible results:

1. If the available APIs, based on their names and descriptions, provide all the necessary functions to complete the task, the query can be considered fully solvable, return:
{
    "solvable": "FullySolvable",
    "reason": "Explain why the tools are sufficient."
}

2. If the available tools can partially resolve the query, return:
{
    "solvable": "PartiallySolvable",
    "reason": "Explain which parts can be covered and which parts are not covered.",
    "uncovered_subqueries": "Write **specific, human-readable subqueries** for the uncovered parts. For example: 'Get the timezone information for Los Angeles.'"
}

3. If the available tools cannot help at all, return:
{
    "solvable": "Unsolvable",
    "reason": "Explain why no current tools are applicable."
}

Important:
- Uncovered_subqueries should **merge all unmet needs into a single, clear natural language query**.
- Never omit any unmet requirement from the original query.
- If a tool's functional category covers the user's need and no explicit limitations are stated, assume the tool is capable of fulfilling that need.

Here are some examples:
---
Example 1:
{
    "solvable": "FullySolvable",
    "reason": "The tools can fully handle the request to get weather information and restaurant recommendations in Paris."
}

Example 2:
{
    "solvable": "PartiallySolvable",
    "reason": "The tools can provide restaurant recommendations but cannot fetch the current weather information.",
    "uncovered_subqueries": "Get the current weather in Paris."
}

Example 3:
{
    "solvable": "Unsolvable",
    "reason": "The query involves translating an ancient script, and no tools are available for that task."
}
"""


META_AGENT_PROMPT = """
You are APIGPT, You have access to a database of apis. The database has the following categories: {categories}.
Your goal is to find the relevant categories for a task and assign each to a new agent. 
You can use the get_tools_in_category function to retrieve the available tools of a specific category. 
If you are unsure about the functionality of some tools, you can use the get_tools_descriptions function to retrieve the details of these tools. This will help you understand the general functionality of each category.
You can use the create_agent_category_level function to assign a relevant category to the agent, which is called immediately when you think it needs to be assigned.
When you finish the assignment, call the Finish function. 
You can assign multiple categories, but each agent can only be assigned one category.
You can try other categories of tools that you haven't explored before. The query may be solved by tools in unexpected categories.
At each step, you need to give your thought to analyze the status now and what to do next, with the function calls to actually excute your step. All the thought is short, at most in 3 sentence.
Remember, you do not need to answer the query, all you need is to find all possible relevant categories and assign them to agents.
"""


CATEGORY_AGENT_PROMPT = """
You are APIGPT, You have access to a database of apis. Each category has many tools. Each tool has many apis.
Now, you should help the user find the relevant tools in '{category}' category for a task.
If you are unsure about the functioinality of some tools, you can use the get_tools_descriptions function to retrieve the details of these tools. Sometimes it's those tools whose functions are not immediately obvious that can find the right solutions.
Then you can use the create_agent_tool_level function to assign a subset of relevant tools to a agent. You should assign similar tools to the same agent and no more than {leaf_tool_number} tools to each agent.
You can assign multiple subsets to different agents. 
Remember, you do not need to answer the query but you need to assign all possible tools. 
If you think that the tools you have assigned can already solve the requirements of the {category} part in the query, please call the Finish function.
From the messages on your history, if the tool has already been assigned, do not repeat the assignment.
You need to give your thought to analyze the status now and what to do next, with the function calls to actually excute your step.
All the thought is short, at most in 3 sentence.
""".replace('{leaf_tool_number}', str(leaf_tool_number))


TOOL_AGENT_PROMPT = """
You are APIGPT, You have access to a database of apis. Each category has many tools. Each tool has many apis.
You will be given all the tool description and the contained api list and their details.
Now, you should help the user find the relevant apis in the tools {tools} of category '{category}' for a task. 
After determining the API names, call add_apis_into_api_pool to include them in the final API list.
If you think you have explored all the possible apis or you think there are no relevant apis in these tools, call the Finish function.
In the middle step, you may be provided with feedback on these apis.
At each step, you should all call a function within \"Finish, add_apis_into_api_pool\" to actually execute your step.
You need to give your thought to analyze the status now and what to do next, with the function calls to actually excute your step.
All the thought is short, at most in 3 sentence.
"""


REFIND_CATEGORY_PROMPT = """
Current APIs failed to solve the query and the result is: {{failed_reason}}. 
Please assign more unexplored categories to the agents.
"""


REFIND_TOOL_PROMPT = """
Current APIs failed to solve the query. The result is: {{failed_reason}}. 
Please assign more unexplored tools to the agents.
"""


REFIND_API_PROMPT = """
Current APIs failed to solve the query. The result is: {{failed_reason}}. 
You need to analyze the result, and find more apis.
It is possible that the tools do not have the relevant apis. In this case, you should call the Finish function. Do not make up the tool names or api names.
"""


INTENT_DECOMPOSER_PROMPT = """
You are an intelligent assistant. Your task is to understand the user's request and transform it into a structured output containing a **task list (task_list)** and a **task scene label (task_scene)**.

Please strictly follow the output format (JSON):
{
  "task_list": "Subtask 1 + Subtask 2 + Subtask 3 (if any)",
  "task_scene": "Task scene label"
}

The goal is to help the system identify the user's **core intentions** and extract clear subtasks. Each subtask represents a specific type of information or action the user explicitly requests, so the system can reuse this in future experience retrieval.

**Important rules to follow:**
1. Only extract **explicitly requested** subtasks. Ignore background details. **Do NOT** add any unrequested content, even if it seems reasonable.
2. Each subtask should be **specific and actionable**. Avoid vague or general terms (for example, replace “check transportation” with “check train tickets” or “check flight details”).
3. Connect subtasks using “+”. Keep the granularity consistent and the expressions clear and unambiguous.
4. **Do NOT** include any **parameters** (such as location names, person names, dates, etc.) in the subtask descriptions. Focus only on the type of action or information the user wants.
5. Output must be in valid JSON format. **Do NOT** include extra explanations or comments.

Below are reference examples:

[Input]: Please provide the CO2 emissions for electricity in Germany and all recent real estate transactions for the 10001 zip code. I'm researching how energy consumption correlates with property activities in that region.
[Output]:
{
  "task_list": "Check CO2 emissions for electricity + Get information on recent real estate transactions in the area by zip code",
  "task_scene": "Travel information query"
}

[Input]: Help me find Steve Jobs' biography and a high-resolution photo.
[Output]:
{
  "task_list": "Extract biography + Get person photo",
  "task_scene": "Person profile lookup"
}

[Input]: Please tell me the latest financial report of Company A and its shareholder structure.
[Output]:
{
  "task_list": "Check company financial report + Get company shareholder structure information",
  "task_scene": "Company research assistant"
}

[Input]: I want to fly from Paris (Charles de Gaulle Airport) to Tokyo (Haneda Airport). Please check the terminal information for both airports, recommend 3 hotels near Haneda, and tell me the next available train from Haneda to Ginza Station.
[Output]:
{
  "task_list": "Check airport terminal information + Recommend nearby hotels + Check train schedule",
  "task_scene": "Travel information query"
}

[Input]: I'm planning a trip to Turkey and need Istanbul postal code information. Can you provide the postal codes and regions for Istanbul Province (license plate 34)? Also, I want to know if there are any transportation agencies in Istanbul. Please look up their names and contact numbers.
[Output]:
{
  "task_list": "Check postal code information + Look up transportation agency names and contact numbers",
  "task_scene": "Travel information query"
}

Now, please decompose the following user request into structured intentions and generate the output:
[Input]: {{USER_INPUT}}
[Output]:
"""


GENERATE_COMBINATION_MULTI_CATEGORY_PROMPT = """
You are a tool combination planner.
I will provide a list of tools grouped by category. Each category contains several tools with descriptions.

Your task:
1. Carefully read all categories and tool descriptions.
2. From each category, select 1-2 of the most suitable tools, with **no more than 5 tools total** across all categories.
3. Ensure the selected tools together support a **realistic and meaningful cross-category user task**.

Strict Instructions:
- **Select at least one tool from each category.**
- **Do not exceed 2 tools per category.**
- **Do not exceed 5 tools in total.**
- Output **only** the selected tool list, no explanation or API info.

Output format (no extra text):
{
    "selected_tools": [
        { "category": "Weather", "tool": "WeatherInfoPro" },
        { "category": "Travel", "tool": "FlightFinder" }
    ]
}

Tool list:
{USER_INPUT}
"""



GENERATE_MUTIL_CATEGORY_PROMPT_VARIANTS = """
You are a dataset-generation assistant. Your task is to generate **a set of 3 semantically similar but stylistically different user queries** based on a realistic multi-tool, multi-category scenario.

You will be given:
- 2-6 tools from different categories
- Each tool includes its description and a list of APIs (with descriptions and required parameters)

Your workflow:
1. Carefully read the tool and API descriptions as well as the API parameters to understand what combined tasks can be accomplished and reflect them in the query.
2. For each tool, select **1-2 most appropriate API** that can realistically be combined with the others in a single user task. 
   - But the total number cannot exceed 5.
   - The selected APIs **must come from at least two different categories**.
3. Compose **three diverse but semantically equivalent user queries**, each of which would trigger the same set of selected APIs.
   - Each query should include a brief context, background, or motivation for the request (e.g., “I'm planning a trip…”).
   - Vary tone, sentence type (e.g., command, question, statement), and phrasing style across the queries.
   - Use realistic, human-like language that reflects plausible everyday usage.
   - Keep each query between **40 and 70 words**.
4. For each selected API:
   - **Embed required parameters naturally into the queries**.
   - Ensure all values are **realistic and match expected formats**.
   - If an API requires no parameters, use an empty object for `parameters`.

Return format:
{
  "apis": [
    { "category": "...", "tool": "...", "api": "...", "parameters": { ... } },
    ...
  ],
  "queries": [
    "First variation (e.g., polite question)...",
    "Second variation (e.g., concise command)...",
    "Third variation (e.g., longer formal statement)..."
  ]
}

Error examples:
 {
  "queries": [
    "I'm looking to get feedback on our customer service. Can you help me create a survey template for Net Promoter Score and summarize the results for our last campaign?"
  ],
  "apis": [
    { "category": "Business_Software", "tool": "NPS-Net Promoter Score", "api": "NPS Template", "parameters": { "tid": "template_id" } },
    { "category": "Business", "tool": "NPS-Net", "api": "NPS Organization", "parameters": { "oid": "organization_id", "start_date": "start_date", "end_date": "end_date" } }
  ]
} 
Reason: No actual ID was provided, so the description cannot be used as a parameter. In addition, the necessary information about "last activity" was not mentioned in the query.

If the tools cannot be logically combined or are too niche, return only: "None"

Tool + API list:
{USER_INPUT}
"""


GENERATE_SINGLE_TOOL_PROMPT_VARIANTS = """
You are a user simulation agent. Your task is to generate **three natural language queries** that express the **same underlying user intent**, using a single tool and its APIs.

You will receive:
- One tool and its description
- A list of its APIs, each with description and required parameters(if they have parameters)

Instructions:
1. **Carefully examine** the tool and API list.
2. Determine whether the tool is realistically useful in real-world scenarios.
   - If it is unrealistic, niche, or not generally useful, return ONLY: "None"
3. If usable, choose **2 to 5 APIs** that can be logically used together.
4. Read the parameters of each API carefully and **reflect them in the query**. 
For example, if you need to analyze an article, the article content should be reflected in the query; if you need to verify an email address, the specific email address should be included in the query.
5. Generate 3 user queries that share the same intent but differ in:
- Sentence type (question / statement / command...),
- Tone (formal / casual / neutral...),
- Wording and structure.

You can include a brief context, background, or motivation for the request (e.g., “I'm planning a trip…”). Each query should be realistic, fluent, and 30-50 words long.  
Avoid simple paraphrasing — make each query stylistically distinct.

Output format:
{
  "apis": [
    { "category": "...", "tool": "...", "api": "...", "parameters": { ... } },
    ...
  ],
  "queries": [
    "First natural query version...",
    "Second rephrased version with different tone...",
    "Third variant, possibly shorter/longer..."
  ]
}

Error examples:
 {
  "queries": [
    "I'm looking to get feedback on our customer service. Can you help me create a survey template for Net Promoter Score and summarize the results for our last campaign?"
  ],
  "apis": [
    { "category": "Business_Software", "tool": "NPS-Net Promoter Score", "api": "NPS Template", "parameters": { "tid": "template_id" } }
  ]
} 
Reason: No actual ID was provided, so the description cannot be used as a parameter. 

If the tool is not usable, just return: "None"

Tool + API list:
{USER_INPUT}
"""


GENERATE_COMBINATION_SINGLE_CATEGORY_PROMPT = """
You are a tool combination planner focused on a single category.

I will provide you with:
- One category name.
- A list of tools under this category.
- Each tool comes with a description of its functionality.

Your task:
1. Carefully read all tool descriptions within this category.
2. Select **2 to 5 tools** from the list that can realistically be **used together** to solve a meaningful and practical user task **within the same domain**.
3. The selected tools must be complementary, meaning they perform distinct but related sub-tasks within a broader domain-specific workflow.
4. Only create combinations that reflect a realistic, everyday use case. If the tools do not meaningfully complement each other, return "None".

**Important Notes:**
- Do **not** include APIs or parameter names.
- Avoid trivial or forced combinations. Only return valid, practical combinations.
- If no meaningful combination can be made, return exactly this string: `"None"`
- Avoid selecting tools that serve the exact same function or provide redundant outputs.

Output Format Example:
{
  "category": "Travel",
  "selected_tools": [
    { "tool_name": "Flight Search Expert" },
    { "tool_name": "HotelDealFinder" },
    { "tool_name": "LocalAttractionsGuide" }
  ]
}

Tool + API list:
{USER_INPUT}
"""


GENERATE_SINGLE_CATEGORY_PROMPT_VARIANTS = """
You are a user simulation assistant. 
Your task is to simulate three distinct but semantically equivalent user queries that involve a realistic combination of tools from a single category.

You will receive:
- A single tool category
- 2 to 5 tools under that category
- Each tool includes a description and a list of APIs (with descriptions and required parameters)

Your workflow:
1. Carefully read the tool and API descriptions to understand what combined tasks can be accomplished.
2. Choose a meaningful and realistic **user intent** that can be achieved by combining these tools.
  - From each tool, select **1-2 APIs** that support this intent.
  - But the total number of APIs should not exceed 5.
3. Read the parameters of each API carefully and **reflect them in the query**. 
For example, if you need to analyze an article, the article content should be reflected in the query; if you need to verify an email address, the specific email address should be included in the query.
4. Generate 3 user queries that share the same intent but differ in:
  - Sentence type (question / statement / command...),
  - Tone (formal / casual / neutral...),
  - Wording and structure.

You can include a brief context, background, or motivation for the request (e.g., “I'm planning a trip…”). Each query should be realistic, fluent, and 40-70 words long.

**Output format**:
{
  "queries": [
    "I'm job hunting in Berlin. Can you pull up data analyst roles, salary insights, and nearby apartment prices?",
    "What are the current openings for data analysts in Berlin, how much do they typically earn, and what’s the housing cost around those areas?",
    "Find me data analyst positions in Berlin, show the average pay range, and give me some info on rent prices nearby."
  ],
  "apis": [
    { "category": "Jobs", "tool": "JobSearchNow", "api": "search_jobs", "parameters": { "location": "Berlin", "position": "data analyst" } },
    { "category": "Jobs", "tool": "SalaryInsights", "api": "get_salary_range", "parameters": { "position": "data analyst", "city": "Berlin" } },
    { "category": "Jobs", "tool": "HousingScout", "api": "get_rent_prices", "parameters": { "city": "Berlin" } }
  ]
}

Error examples:
 {
  "queries": [
    "I'm looking to get feedback on our customer service. Can you help me create a survey template for Net Promoter Score and summarize the results for our last campaign?"
  ],
  "apis": [
    { "category": "Business_Software", "tool": "NPS-Net Promoter Score", "api": "NPS Template", "parameters": { "tid": "template_id" } },
    { "category": "Business_Software", "tool": "NPS-Net Promoter Score", "api": "NPS Organization", "parameters": { "oid": "organization_id", "start_date": "start_date", "end_date": "end_date" } }
  ]
} 
Reason: No actual ID was provided, so the description cannot be used as a parameter. In addition, the necessary information about "last activity" was not mentioned in the query.

If no meaningful combination is possible, return ONLY: "None"

Tool + API list:
{USER_INPUT}
"""


JUDGE_USABILITY_PROMPT = """
You are an intelligent task scheduling assessment expert. 
Your job is to determine whether a historical experience could help solve a new user query.

Current user query:
{current_query}
Historical experience:
{experience}

Please answer only "yes" or "no" (lowercase):

Criteria:
- If any part of the historical experience (scene, subtask intents) **matches the type of task or information** in the current query, answer "yes".
- If the tools used in the historical experience could be reused or adapted to solve (part of) the current query, answer "yes".
- Otherwise, answer "no".

"""


IF_CHANGE_CATEGORY_PROMPT = """
You're helping decide whether to continue searching for tools in {category} category and add it to the tool_list. 
{category} category contains the following tools: {category_tools}
User query: {query}
I can use the following tool_list: {global_api_list_detailed}
Below is why this tool_list cannot solve the query: {reason}
Based on this reason and the user's query, do you think it's worth continuing to look for tools in **this category**, or should we move on to a **different category**?

**Please reply with one word**:
- old → continue searching in this category
- new → switch to a different category
"""


USED_TOOL_SIGNATURE_PROMPT = """
Calling the same tool with the same parameters leads to redundant and unnecessary results. So don't call a function again **with the same parameters**.
"""


FILTER_DATASET_PROMPT = """
You are a evaluator of tool-using queries. Given the following user query, determine whether it is solvable using APIs.

Filter out non-solvable queries according to the following conditions
1. Queries lacking essential information, such as unspecified phone numbers or ambiguous references like “my friend”. These are inherently non-solvable since APIs require explicit input parameters.
2. Queries containing fake parameters, such as non-existent URLs.
3. Queries that specify a specific API are filtered out because they do not represent realistic scenarios. 
4. It must not be unreasonably vague or broad (e.g., "what's popular", "recommend me something cool").

Respond with only one word: "Yes" if it is solvable, or "No" if it is not. 

Query:
{query}
"""


FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = """You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: {task_description}"""


FORMAT_INSTRUCTIONS_USER_FUNCTION = """
{input_description}
Begin!
"""


FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ADAPTED = """You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer. 
If you feel you cannot solve the task or can only solve it partially, you should choose to give up and give your reason which should mention the names of the failed functions.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart" and give the reason.
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: {task_description}"""


DIVERSITY_PROMPT='''This is not the first time you try this task, all previous trails failed.
Before you generate my thought for this state, I will first show you your previous actions for this state, and then you must generate actions that is different from all of them. Here are some previous actions candidates:
{previous_candidate}
Remember you are now in the intermediate state of a trail, you will first analyze the now state and previous action candidates, then make actions that is different from all the previous.'''


LLM_PAIRWISE_RANK_SUBFIX_SYSTEM_PROMPT = '''
You are value-GPT, which is an expert of defining which trail is better, which trail is more close to solving the task. 
All candidate tries to solve this task with some funciton calls:
*******************************
{{TASK_DESCRIPTION}}
{task_description}
{{END_TASK_DESCRIPTION}}
*******************************
First, all candidate do the following things:
{intersect_trice}
After that, there are two candidates A and B, they do different things:
*******************************
{{CANDIDATE_A_START}}
{candidate_A}
{{CANDIDATE_A_END}}
*******************************
{{CANDIDATE_B_START}}
{candidate_B}
{{CANDIDATE_B_END}}
Which try do you think is more helpful to solving the task?
'''


LLM_PAIRWISE_RANK_USER_PROMPT = '''
Tell me which candidate is better in ONE Word: "A" or "B":'''


CHECK_API_PROMPT = """
You are an AI assistant that determines the nature of a given API output.

Given:
- The action name: {action_name}
- The action parameters: {action_param}
- The result: {obs}

Your task:

Classify the "result" into one of the following **three categories**:
1. **real_response** → A real-time API output that reflects specific user input or parameters. It gives actual returned content, such as search results, fact lookup results, or generated values **based on the input**.
2. **static_description** → A generic explanation of what the API does, its usage, or expected behavior. It does **not** depend on the input, and reads like documentation or help text.
3. **error_message** → 错误或空结果或者是指出参数范围的错误，An error or empty result, such as invalid/missing input, wrong parameters, empty reply, authentication failure, etc.

---

📌 Key distinction:

- If the response **includes specific content based on the user input**, it's **real_response**.
- If the response **could be shown even without any input**, it's **static_description**.

---

📌 Examples:

✅ static_description:
{"error":"","response": "This API allows users to search for weather information by location. It returns temperature, humidity, and forecast."} → **static_description**
{"error":"","response":"The API documentation provided does not contain any specific information regarding its functionality, parameters, or expected outputs. Additionally, there are no examples available to reference for generating a meaningful response. Please provide more detailed API documentation or examples to enable a more accurate and informative response."} → **static_description**

✅ error_message:
{"error":"","response": "Missing required parameter: 'location'."} → **error_message**
{"error":"","response":"[]"} → **error_message**

✅ real_response:
{"error":"","response":"[{'title': 'Meme Monday', 'url': 'https://dev.to/ben/meme-monday-53cl'}]"} → **real_response**
{"error":"","response":"['best productivity advice', 'best productivity systems']"} → **real_response**
{"error":"","response":"The drug 'Metformin' administered via the 'oral' route is primarily used to treat type 2 diabetes..."} → **real_response**
{"error":"","response":"For the input 'avocado', the API returns: 160 calories, 15g fat, 2g protein, 9g carbs..."} → **real_response**

---

Your response must be exactly one of the following:  
**real_response**, **static_description**, or **error_message**

Answer with only the label. Do not explain.
"""

