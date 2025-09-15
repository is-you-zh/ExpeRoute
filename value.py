from openai import OpenAI
import json
client = OpenAI(
        api_key="sk-Iw1MLBlIRuPsmFbtimgfi9eqj42ZyvpmisJIeWbN6rGJfFKa",
        base_url="https://xiaoai.plus/v1" 
    )

def evaluate_completion_score(query, system_response):
    prompt = f"""
You are an expert evaluator assessing how well a system response fulfills a user query. Based on the following criteria, output only a numeric score between 0.00 and 1.00 (e.g., 0.75). Do not provide any explanation.

[Evaluation Criteria]:
· 0.00: Completely irrelevant or useless response; fails to address the query.
- 0.10-0.69: Partially relevant; some key elements are addressed, but there are major omissions, logical errors, or broken steps.
- 0.70-0.99: Mostly complete; the main task is addressed, with only minor issues in detail, completeness, or precision.
- 1.00: Fully complete, accurate, and logically sound; no significant issues, no further clarification needed.

Notes:
- Ignore irrelevant or redundant content returned by the system (e.g., boilerplate or verbose API output).
- The primary evaluation criterion is the degree of task completion.

[Example]:
Query: Please provide the auditor details for ROAC number 87654321 and also return the company financial summary for company ID 54321. What is the temperature in Wuhan today?
Score 1.00
Response:
"Auditor info: Name: Carmen Vázquez Peña, ROAC: 87654321, Document: 33345107Y. Company financial summary for ID 54321: Revenue: $5M, Profit: $1M, Fiscal Year: 2023. Note: API version used is v1.17.0. The temperature in Wuhan is 35°C today."

Score 0.86
Response:
"Auditor info: Name: Carmen Vázquez Peña, ROAC: 87654321, Document: 33345107Y. Company financial summary for ID 54321: Revenue: $5M, Profit: $1M, Fiscal Year: 2023."

Score 0.65
Response:
"Auditor info: Name: Carmen Vázquez Peña. Company financial summary not available. Wuhan temp: 35°C."

Score 0.30
Response:
"The ROAC 87654321 is related to an auditor in Spain. Company financial data might be confidential."

Score 0.00
Response:
"Cats are great pets and love to jump around."

[User Query]:
{query}

[System Response]:
{system_response}
"""

    try:
        messages = [{"role": "user", "content": prompt}]
        # print(messages)
        # response = client.chat.completions.create(
        #     model="gpt-4-0125-preview", messages=messages, temperature=0.0,
        # )
        # print(messages)
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.0,
        )
        score_text = response.choices[0].message.content.strip()

        score = float(score_text)
    except Exception as e:
        print(f"Error in evaluation: {e}")
        score = -1  # Use -1 to indicate evaluation failure

    # Write to log file
    # with open(log_path, "a", encoding="utf-8") as log_file:
    #     log_file.write(f"Query: {query}\n")
    #     log_file.write(f"Response: {system_response}\n")
    #     log_file.write(f"Score: {score}\n")
    #     log_file.write("=" * 60 + "\n")

    return score

# query="I need to call the `api_auditores_roac_roac_sociedad_for_roac` function to get the detailed information for the auditor with ROAC number 87654321. The reason for calling this function is to retrieve the specific information related to this auditor. By passing the argument `{"roac": "87654321"}`, I can ensure that the API fetches the information for the correct ROAC number. This will help me provide the user with accurate details about the auditor they are interested in.".replace('"""', '\\"\\"\\"')

output="{\n  \"return_type\": \"give_answer\",\n  \"final_answer\": \"The detailed information for the auditor with ROAC number 87654321 is as follows: Documento: 33345107Y, CodigoROAC: 87654321, IdTipoDocumento: 1, Nombre: CARMEN, Apellidos: VAZQUEZ PE\u00d1A, RazonSocial: None, FechaAlta: 2014-09-10, FechaBaja: None. The current version of the AI service used for removing backgrounds from people photos is v1.17.0.\"\n}"

# print(evaluate_completion_score(query, output))