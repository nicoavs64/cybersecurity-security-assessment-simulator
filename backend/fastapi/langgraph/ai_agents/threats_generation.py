from loguru import logger
from ..helpers.graph_state_classes import BusinessState, ThreatItemCollection
from ..helpers.model_config import fetch_model_from_ollama
#from ..helpers.output_validation import (
#    validate_generated_output,
#    create_threats_validation_prompt,
#    format_items_for_llm,
#)
from ..helpers.output_validation import format_items_for_llm
from ..prompts.threats_generation_prompt import threat_generator_prompt_message


def generate_threats(
    state: BusinessState, llm_model_name: str = "llama3.2"
) -> ThreatItemCollection | BusinessState:
    """
    Generates potential threats for a business based on business description, activities, and assets
    :param state: the business state object containing all the business information
    :param llm_model_name: the name of the model to be used for this task, which refers to a model on ollama model registry
    :return: the generated threats, or the previous state if there was an error
    """
    formatted_assets = format_items_for_llm(state["assets"])

    prompt = threat_generator_prompt_message.format(
        business_description=state["business_description"],
        business_activities=state["business_activity"],
        business_assets=formatted_assets,
    )

    try:
        llm_model = fetch_model_from_ollama(model_name=f"{llm_model_name}")
        llm_model_structured_output = llm_model.with_structured_output(
            ThreatItemCollection
        )
        logger.info(f"{llm_model_name} fetched successfully for threats generation")

        generated_threats = llm_model_structured_output.invoke(prompt)
        logger.info("Successfully generated threats.")
        return generated_threats

    except Exception as e:
        logger.error(
            f"There was an error while generating threats. Details below: \n{e}"
        )
        return state


def get_validated_threats(state: BusinessState) -> BusinessState | None:
    generated_threats = generate_threats(state)

    if not getattr(generated_threats, "threats", None):
        logger.error("Failed to generate threats.")
        return None

    return BusinessState(
        business_name=state["business_name"],
        business_location=state["business_location"],
        business_contact_info=state["business_contact_info"],
        business_activity=state["business_activity"],
        business_description=state["business_description"],
        assets=state["assets"],
        potential_threats=generated_threats,
    )

'''
def get_validated_threats(
    state: BusinessState, max_retries: int = 3
) -> BusinessState | None:
    """
    Calls threat generator and validator. Re-generates the threats if the output is not satisfactory
    :param state: previously generated business for which the threats will be generated
    :param max_retries: the max number of times business generator can be called to generate a new business if output is not satisfactory
    :return: the final business generator in a BusinessState format
    """
    attempt = 0
    are_threats_appropriate = None

    while attempt < max_retries:
        attempt += 1
        logger.info(f"Threats generation attempt {attempt}/{max_retries}")

        generated_threats = generate_threats(state)

        if not generated_threats.threats:
            logger.error("Failed to generate a threats.")
            break

        formatted_threats = format_items_for_llm(generated_threats)

        are_threats_appropriate = validate_generated_output(
            prompt=create_threats_validation_prompt(
                original_prompt=threat_generator_prompt_message,
                generated_threats=formatted_threats,
            )
        )

        if are_threats_appropriate.is_valid:
            logger.success("Generated sensible threats.")

            business_state_new_structure = BusinessState(
                business_name=state["business_name"],
                business_location=state["business_location"],
                business_contact_info=state["business_contact_info"],
                business_activity=state["business_activity"],
                business_description=state["business_description"],
                assets=state["assets"],
                potential_threats=generated_threats,
            )
            return business_state_new_structure

        else:
            logger.warning("Generated business is invalid. Retrying...")

    if not are_threats_appropriate or not are_threats_appropriate.is_valid:
        logger.error("Failed to generate a valid business after all retries.")

    return None
'''

if __name__ == "__main__":
    logger.info(
        "Not a runnable file. To run the business owner, please use api or test files"
    )
