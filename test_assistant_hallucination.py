# Check Hallucination
from langchain.prompts                import ChatPromptTemplate
from langchain.chat_models            import ChatOpenAI
from langchain.schema.output_parser   import StrOutputParser

def create_eval_chain(context, agent_response):
  eval_system_prompt = """You are an assistant that evaluates \
  how well the quiz assistant
    creates quizzes for a user by looking at the set of \
    facts available to the assistant.
    Your primary concern is making sure that ONLY facts \
    available are used. Quizzes that contain facts outside
    the question bank are BAD quizzes and harmful to the student."""
  
  eval_user_message = """You are evaluating a generated quiz based on the question bank that the assistant uses to create the quiz.
  Here is the data:
    [BEGIN DATA]
    ************
    [Question Bank]: {context}
    ************
    [Quiz]: {agent_response}
    ************
    [END DATA]

## Examples of quiz questions
Subject: <subject>
   Categories: <category1>, <category2>
   Facts:
    - <fact 1>
    - <fact 2>

## Steps to make a decision
Compare the content of the submission with the question bank using the following steps

1. Review the question bank carefully. These are the only facts the quiz can reference
2. Compare the information in the quiz to the question bank.
3. Ignore differences in grammar or punctuation

Remember, the quizzes should only include information from the question bank.


## Additional rules
- Output an explanation of whether the quiz only references information in the context.
- Make the explanation brief only include a summary of your reasoning for the decsion.
- Include a clear "Yes" or "No" as the first paragraph.
- Reference facts from the quiz bank if the answer is yes

Separate the decision and the explanation. For example:

************
Decision: <Y>
************
Explanation: <Explanation>
************
"""
eval_prompt = ChatPromptTemplate.from_messages([
      ("system", eval_system_prompt),
      ("human", eval_user_message),
  ])
eval_prompt

  return eval_prompt | ChatOpenAI(
      model="gpt-3.5-turbo", 
      temperature=0) | \
    StrOutputParser()

def test_model_graded_eval_hallucination(quiz_bank):
  assistant = assistant_chain()
  quiz_request = "Write me a quiz about books."
  result = assistant.invoke({"question": quiz_request})
  print(result)
  eval_agent = create_eval_chain(quiz_bank, result)
  eval_response = eval_agent.invoke({"context": quiz_bank, "agent_response": result})
  print(eval_response)
  # Our test asks about a subject not in the context, so the agent should answer N
  assert eval_response == "N"


