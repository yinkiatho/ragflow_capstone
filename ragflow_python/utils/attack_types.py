from typing import List
from enum import Enum
from deepeval.vulnerability.misinformation import MisinformationType 
from deepeval.vulnerability.bias import BiasType # Vulnerability Type
from deepeval.vulnerability.robustness  import RobustnessType
from deepeval.vulnerability.illegal_activity import IllegalActivityType 
from deepeval.vulnerability.personal_safety import PersonalSafetyType 

from deepeval.vulnerability import BaseVulnerability
import ragflow_python.utils.logger as log

class LegalAttackType(Enum):
    '''
    NOT USABLE AT THE MOMENT
    '''
    # Hallucination
    FAKE_CASES = "Fake Legal Cases"
    FABRICATED_STATUTES = "Fabricated Laws"
    NONEXISTENT_PRECEDENTS = "Nonexistent Precedents"

    # AdversarialPrompting
    LEGAL_LOOPHOLES = "Tricking into Illegal Advice"
    ROLE_PLAYING = "Masquerading as a Lawyer"
    MULTI_TURN_ATTACKS = "Step-by-Step Manipulation"

    # ContradictionType
    CONFLICTING_PRECEDENTS = "Conflicting Case Law"
    AMBIGUOUS_STATUTES = "Ambiguous Legal Text"
    OVERGENERALIZATION = "Applying Laws Too Broadly"
    
    # Customized Legal Prompt Injections
    HIDDEN_INSTRUCTIONS = "Injecting Commands in Queries"
    HTML_TAG_EXPLOITS = "Hiding Prompts in Markup"
    OBFUSCATED_TEXT = "Using Encodings or Symbols"
    
    # Context Injection / Overloading
    MISLEADING_CONTEXT = "Fake Context Injection"
    PROMPT_OVERLOAD = "Too Much Information"
    MANIPULATIVE_SUMMARIES = "Framing the Issue"


class LegalAttack(BaseVulnerability):
    def __init__(self, types: List[LegalAttackType]):
        if not isinstance(types, list):
            raise TypeError("The 'types' attribute must be a list of LegalAttackType enums.")
        if not types:
            raise ValueError("The 'types' attribute cannot be an empty list.")
        if not all(isinstance(t, LegalAttackType) for t in types):
            raise TypeError("All items in the 'types' list must be of type LegalAttackType.")
        super().__init__(types=types)

    def get_name(self) -> str:
        return "Legal Attack"
    

enum_classes = {
    "MisinformationType": MisinformationType,
    "BiasType": BiasType,
    "RobustnessType": RobustnessType,
    "IllegalActivityType": IllegalActivityType,
    "PersonalSafetyType": PersonalSafetyType,
    "LegalAttackType": LegalAttackType,
}

def get_enum_value(enum_input, enum_classes: dict = enum_classes) -> str:
    """Extracts the enum value from a string like 'MisinformationType.UNSUPPORTED_CLAIMS'
    or directly from an enum instance."""
    
    # If the input is already an Enum instance, return its value directly
    if isinstance(enum_input, Enum):
        return enum_input.value

    # If the input is a string, process it normally
    if isinstance(enum_input, str):
        try:
            enum_class_name, enum_member_name = enum_input.split(".")
            enum_class = enum_classes.get(enum_class_name)  # Get the enum class dynamically
            
            if enum_class is None:
                raise ValueError(f"Unknown enum type: {enum_class_name}")
            
            return enum_class[enum_member_name].value  # Get the actual value
        except (KeyError, ValueError) as e:
            return f"Invalid Enum Entry: {enum_input} ({e})"
    logger.error(f"Failed to parse correct enum value")
    return f"Invalid Input Type: {type(enum_input)}"