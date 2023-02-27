INTENT_PROMPT = """
I want to understand the intent in the question a user asks, so I can give the right answer. The intent can be one of the following:

'user_action_vs_law' is when the user wants to understand how his/her action compares to the law.
'officer_action_vs_law' is when the user wants to understand how the officer's action compares to the law.
'complaint': is when a user complains about either the law or an officer's action.

For example:

- me multaron por no traer el cinturon, es correcto?
intent = {
    'user_action_vs_law': True, 
    'officer_action_vs_law': True, 
    'complaint': False
}

- me detuvieron y el oficial se porto muy grosero
intent = {
    'user_action_vs_law': False, 
    'officer_action_vs_law': True, 
    'complaint': True
}

- es normal que te esposen/arresten/multen/detengan por una infraccion de transito?'
intent = {
    'user_action_vs_law': True, 
    'officer_action_vs_law': True, 
    'complaint': False
}

- me cruce un semaforo y me llevaron al juzgado, tiene esa facultad un estatal?
intent = {
    'user_action_vs_law': True, 
    'officer_action_vs_law': True, 
    'complaint': False
}

Here are some more examples:

- me puedo estacionar en sentido contrario?
intent = {
    'user_action_vs_law': True,
    'officer_action_vs_law': False,
    'complaint': False
}

- me multaron/arrestaron/detuvieron pero no me dijeron por que
intent = {
    'user_action_vs_law': False,
    'officer_action_vs_law': True,
    'complaint': False
}

- me pueden arrestar/multar/detener dentro de mi casa?
intent = {
    'user_action_vs_law': False,
    'officer_action_vs_law': True,
    'complaint': False
}
{section_01}
"""