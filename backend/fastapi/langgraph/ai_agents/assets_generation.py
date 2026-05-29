from loguru import logger
from ..helpers.graph_state_classes import (
    BusinessState,
    AssetCollection,
    ThreatItemCollection,
)
from ..helpers.file_operations import retrieve_input_file
from ..helpers.model_config import fetch_model_from_ollama
from ..prompts.assets_generation_prompt import asset_generator_prompt_message
#from ..helpers.output_validation import (
#    create_assets_validation_prompt,
#    validate_generated_output,
#    format_items_for_llm,
#)


def generate_assets(
    state: BusinessState,
    assets_example_filepath: str = "Assets_ZenithPoint.txt",
    llm_model_name: str = "llama3.2",
) -> AssetCollection | BusinessState:
    """
    Generates a list of assets based on a generated business's description using a prompt template, and an example
    :param state: the generated business's state. This is based on a Pydantic Base Model class with the necessary Fields to generate assets
    :param assets_example_filepath: path to the assets example file. Defaults to "Assets_ZenithPoint.txt"
    :param llm_model_name: name of the model to use for the given task, according to ollama model registry. Defaults to "llama3.2"
    :return: the generated assets from the llm in the form of AssetCollection
    """

    try:
        assets_example = retrieve_input_file(assets_example_filepath)
        prompt = asset_generator_prompt_message.format(
            business_description=state["business_description"],
            business_activity=state["business_activity"],
            assets_listing_example=assets_example,
        )

        model_ollama = fetch_model_from_ollama(llm_model_name).with_structured_output(
            AssetCollection
        )
        logger.info(f"{llm_model_name} fetched successfully for asset generation")

        response = model_ollama.invoke(prompt)
        return response

    except Exception as e:
        logger.error(
            f"Failed to produce output with {llm_model_name}. Details below:\n{e}"
        )
        return state
        
def get_validated_assets(state: BusinessState) -> BusinessState | None:
    generated_assets = generate_assets(state)

    if not getattr(generated_assets, "assets", None):
        logger.error("Failed to generate assets.")
        return None

    return BusinessState(
        business_name=state["business_name"],
        business_location=state["business_location"],
        business_contact_info=state["business_contact_info"],
        business_activity=state["business_activity"],
        business_description=state["business_description"],
        assets=generated_assets,
        potential_threats=ThreatItemCollection(threats=[]),
    )

'''
def get_validated_assets(
    state: BusinessState, max_retries: int = 3
) -> BusinessState | None:
    """
    Calls asset generator and validator. Re-generates the assets if the output is not satisfactory
    :param state: previously generated business for which the assets will be generated
    :param max_retries: the max number of times business generator can be called to generate a new business if output is not satisfactory
    :return: the final business generator in a BusinessState format
    """
    attempt = 0
    are_assets_appropriate = None

    while attempt < max_retries:
        attempt += 1
        logger.info(f"Assets generation attempt {attempt}/{max_retries}")

        generated_assets = generate_assets(state)

        if not generated_assets.assets:
            logger.error("Failed to generate assets.")
            break

        formatted_assets = format_items_for_llm(generated_assets)

        are_assets_appropriate = validate_generated_output(
            prompt=create_assets_validation_prompt(
                original_prompt=asset_generator_prompt_message,
                generated_assets=formatted_assets,
            )
        )

        if are_assets_appropriate.is_valid:
            logger.success("Generated sensible assets.")

            business_state_new_structure = BusinessState(
                business_name=state["business_name"],
                business_location=state["business_location"],
                business_contact_info=state["business_contact_info"],
                business_activity=state["business_activity"],
                business_description=state["business_description"],
                assets=generated_assets,
                potential_threats=ThreatItemCollection(threats=[]),
            )
            return business_state_new_structure

        else:
            logger.warning("Generated business is invalid. Retrying...")

    if not are_assets_appropriate or not are_assets_appropriate.is_valid:
        logger.error("Failed to generate a valid business after all retries.")

    return None
'''

if __name__ == "__main__":
    logger.info(
        "Not a runnable file. To run the business owner, please use api or test files"
    )

