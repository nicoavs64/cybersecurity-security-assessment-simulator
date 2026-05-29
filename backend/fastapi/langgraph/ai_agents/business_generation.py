from langchain_core.messages import HumanMessage
from loguru import logger

from ..prompts.business_generation_prompt import business_generation_prompt_message
from ..helpers.file_operations import retrieve_input_file
from ..helpers.model_config import fetch_model_from_ollama
from ..helpers.graph_state_classes import (
    BusinessState,
    BusinessOnlyState,
    AssetCollection,
    ThreatItemCollection,
)
#from ..helpers.output_validation import (
#    validate_generated_output,
#    create_business_validation_prompt,
#)


def generate_business(
    business_generation_prompt: str = business_generation_prompt_message,
    business_example_filename: str = "Business_ZenithPoint.txt",
    llm_model_name: str = "llama3.2",
) -> BusinessOnlyState | None:
    """
    Generates a business idea, using a prompt template, and an example
    :param business_generation_prompt:
    :param business_example_filename:
    :param llm_model_name:
    :return:
    """

    business_example_for_prompt_message = retrieve_input_file(
        f"{business_example_filename}"
    )

    business_generation_formatted_prompt = business_generation_prompt.format(
        example=business_example_for_prompt_message
    )

    try:
        ollama_llm = fetch_model_from_ollama(f"{llm_model_name}")
        ollama_llm_with_structured_output = ollama_llm.with_structured_output(
            BusinessOnlyState
        )
        logger.info(f"{llm_model_name} fetched successfully for business generation")

        ollama_llm_output = ollama_llm_with_structured_output.invoke(
            [HumanMessage(content=business_generation_formatted_prompt)]
        )
        return ollama_llm_output
    except Exception as e:
        logger.error(
            f"Failed to produce output with {llm_model_name}. Details below:\n", e
        )
        return


def get_validated_business() -> BusinessState | None:
    business = generate_business()

    if not business:
        logger.error("Failed to generate a business.")
        return None

    return BusinessState(
        business_name=business.business_name,
        business_location=business.business_location,
        business_contact_info=business.business_contact_info,
        business_activity=business.business_activity,
        business_description=business.business_description,
        assets=AssetCollection(assets=[]),
        potential_threats=ThreatItemCollection(threats=[]),
    )

'''
def get_validated_business(max_retries: int = 3) -> BusinessState | None:
    """
    Calls business generator and validator. Re-generates the business if the output is not satisfactory
    :param max_retries: the max number of times business generator can be called to generate a new business if output is not satisfactory
    :return: the final business generator in a BusinessState format
    """
    attempt = 0
    is_business_legit = None

    while attempt < max_retries:
        attempt += 1
        logger.info(f"Business generation attempt {attempt}/{max_retries}")

        business = generate_business()

        if not business:
            logger.error("Failed to generate a business.")
            break

        business_state = BusinessOnlyState(
            business_name=business.business_name,
            business_location=business.business_location,
            business_contact_info=business.business_contact_info,
            business_activity=business.business_activity,
            business_description=business.business_description,
        )

        is_business_legit = validate_generated_output(
            prompt=create_business_validation_prompt(
                original_prompt=business_generation_prompt_message,
                generated_business=business_state,
            )
        )

        if is_business_legit.is_valid:
            logger.success("Generated a valid business.")
            business_state_new_structure = BusinessState(
                business_name=business.business_name,
                business_location=business.business_location,
                business_contact_info=business.business_contact_info,
                business_activity=business.business_activity,
                business_description=business.business_description,
                assets=AssetCollection(assets=[]),
                potential_threats=ThreatItemCollection(threats=[]),
            )
            return business_state_new_structure

        else:
            logger.warning("Generated business is invalid. Retrying...")

    if not is_business_legit or not is_business_legit.is_valid:
        logger.error("Failed to generate a valid business after all retries.")

    return None
'''

if __name__ == "__main__":
    logger.info(
        "Not a runnable file. To run the business owner, please use api or test files"
    )
