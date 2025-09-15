get_tools_in_category_function = {
    'name': 'get_tools_in_category',
    'description': 'get all tools in a specific category',
    'parameters': {
        'type': 'object',
        'properties': {
            'category_name': {'type': 'string'}
        },
        'required': ['category_name']
    }
}

get_tools_descriptions_function = {
    'name': 'get_tools_descriptions',
    'description': 'get the descriptions of some tools in a specific category. Require input to be list of tool names. You should query no more than 10 tools at a time.',
    'parameters': {
        'type': 'object',
        'properties': {
            'category_name':{'type':'string'}, 
            'tool_list': {
                'type': 'array', 
                'items': {'type': 'string'}
            }
        },
        'required': ['category_name', 'tool_list']
    }
}

create_agent_category_level_function = {
    'name': 'create_agent_category_level',
    'description': 'Assign a category to an agent',
    'parameters': {
        'type': 'object',
        'properties': {
            'category': {'type': 'string'}
        },
        'required': ['category']
    }
}

finish_function = {
    "name": "Finish",
    "description": "If you think you have finished, call this function.",
    "parameters": {
        "type": "object",
        'properties': {
    }
    }
}

create_agent_tool_level_function = {
    'name': 'create_agent_tool_level',
    'description': 'Assign a subset of tools in a category to an agent',
    'parameters': {
        'type': 'object',
        'properties': {
            'category': {'type': 'string'}, 
            'tools': {
                'type': 'array', 
                'items': {'type': 'string'}
            }
        },
        'required': ['category', 'tools']
    }
}


add_apis_into_api_pool_function = {
    'name': 'add_apis_into_api_pool',
    'description': 'add apis to the final api list. required input to be list of dictionaries describing with the keys category_name, tool_name, api_name',
    'parameters': {
        'type': 'object',
        'properties': {
            'api_list': {'type': 'null'}
        },
        'required': ['api_list']
    }
}                


finish_func = {
    "name": "Finish",
    "description": "1. If all the instructions in the user's input question have been completed, or the return of the tool expresses the meaning that it can be solved, call this function to provide the final_answer, which should include all the results the user needs.2. If the current tools cannot complete the task, call this function to restart and give the reason. The name of the failed function or the current unsolvable problem should be mentioned.3. If the task cannot be completed not because of the lack of tools, call this function to end and give the reason.  Remember: You must ALWAYS call this function at the end of your attempt and return the return_type.",
    "parameters": {
        "type": "object",
        "properties": {
            "return_type": {
                "type": "string",
                "enum": ["give_answer","give_up_and_restart", "give_up"],
            },
            "final_answer": {
                "type": "string",
                "description": "The final answer you want to give the user. It should not contain any sorry message. You should have this field if \"return_type\"==\"give_answer\"",
            },
            "reason": {
                "type": "string",
                "description": "The reason why you give up. You should mention the names of the failed functions. You should have this field if \"return_type\"==\"give_up\" or \"return_type\"==\"give_up_and_restart\"",
            }
        },
        "required": ["return_type"],
    }
  }