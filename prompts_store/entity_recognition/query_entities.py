QUERY_ENTITY_RECOGNITION_PROMPT = """
Can you classify the following text into the entities I am going to show you?
For example:

- me multaron por no traer el cinturon, es correcto?
ner={
    'officer_action_or_consequence': 'me multaron',
    'user_action': 'no traer cinturon',
    'question': 'es correcto'
    'other': None
}

- me cruce un amarillo y no me multaron.
ner={
    'officer_action_or_consequence': None,
    'user_action': 'me cruce un amarillo',
    'question': None,
    'other': 'no me multaron'
}

- me puedo estacionar en sentido contrario?
ner={
    'officer_action_or_consequence': None,
    'user_action': 'estacionar en sentido contrario',
    'question': 'me puedo',
    'other': None
}

- me multaron pero no me dijeron por que
ner={
    'officer_action_or_consequence': 'me multaron',
    'user_action': None,
    'question': None,
    'other': 'no me dijeron por que'
{section_01}
"""