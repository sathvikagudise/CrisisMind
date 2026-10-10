import logging

logger = logging.getLogger("crisismind")

SEVERITY_RULES = {
    "Little or No Damage": {
        "priority": "Low",
        "infrastructure_impact": "Minor",
        "emergency_level": "Monitor Situation",
        "recommended_actions": [
            "Monitor situation",
            "Document damage",
            "Continue observation",
        ],
    },
    "Mild Damage": {
        "priority": "Medium",
        "infrastructure_impact": "Moderate",
        "emergency_level": "Field Assessment Required",
        "recommended_actions": [
            "Inspect roads",
            "Deploy local response teams",
            "Assess utilities",
        ],
    },
    "Severe Damage": {
        "priority": "High",
        "infrastructure_impact": "Major",
        "emergency_level": "Immediate Response",
        "recommended_actions": [
            "Deploy emergency response teams",
            "Coordinate with authorities",
            "Assess critical infrastructure",
            "Initiate evacuation if necessary",
        ],
    },
}


def decide(severity: str) -> dict:
    result = SEVERITY_RULES.get(severity)
    if result is None:
        logger.warning("Unknown severity '%s'; defaulting to Severe Damage rules", severity)
        result = SEVERITY_RULES["Severe Damage"]
    logger.info(
        "Decision: priority=%s, emergency=%s, impact=%s",
        result["priority"],
        result["emergency_level"],
        result["infrastructure_impact"],
    )
    return {
        "priority": result["priority"],
        "infrastructure_impact": result["infrastructure_impact"],
        "emergency_level": result["emergency_level"],
        "recommended_actions": list(result["recommended_actions"]),
    }
