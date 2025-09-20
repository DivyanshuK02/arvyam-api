from app.selection_engine import selection_engine

ROMANTIC_PROMPT = "I am so sorry my love, I messed up completely"
PROFESSIONAL_PROMPT = "my apologies to the team for the delay"

def test_apology_romantic_routes_to_affection_support():
    """
    Tests that a romantic apology correctly resolves to the
    'Affection/Support' anchor and 'romantic' relationship context.
    """
    _, context, _ = selection_engine(prompt=ROMANTIC_PROMPT, context={})
    assert context.get("resolved_anchor") == "Affection/Support"
    assert context.get("relationship_context") == "romantic"


def test_apology_professional_routes_to_reconciliation_lanes():
    """
    Tests that a professional apology correctly sets the 
    sentiment_family to 'professional_repair'.
    """
    _, context, _ = selection_engine(prompt=PROFESSIONAL_PROMPT, context={})
    assert context.get("sentiment_family") == "professional_repair"
    assert context.get("relationship_context") == "professional"
