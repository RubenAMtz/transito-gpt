INTENT_PROMPT = """
I want to understand the type of question a user asks, can you classify the following text into the categories I am going to show you?

'user_action_vs_law' is when the user wants to understand how his/her action compares to the law.
'officer_action_vs_law' is when the user wants to understand how the officer's action compares to the law.
'complaint': is when a user complains about either the law or an officer's action.

For example:

- me multaron por no traer el cinturon, es correcto?
question_classification = {
    'user_action_vs_law': True, 
    'officer_action_vs_law': True, 
    'complaint': False
}

- me detuvieron y el oficial se porto muy grosero
question_classification = {
    'user_action_vs_law': False, 
    'officer_action_vs_law': True, 
    'complaint': True
}

- es normal que te esposen por una infraccion de transito?'
question_classification = {
    'user_action_vs_law': True, 
    'officer_action_vs_law': True, 
    'complaint': False
}

- me cruce un semaforo y me llevaron al juzgado, tiene esa facultad un estatal?
question_classification = {
    'user_action_vs_law': True, 
    'officer_action_vs_law': True, 
    'complaint': False
}

- me puedo estacionar en sentido contrario?
question_classification = {
    'user_action_vs_law': True,
    'officer_action_vs_law': False,
    'complaint': False
}

- me multaron pero no me dijeron por que
question_classification = {
    'user_action_vs_law': False,
    'officer_action_vs_law': True,
    'complaint': False
}
{section_01}
"""