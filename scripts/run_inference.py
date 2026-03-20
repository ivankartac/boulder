import json
import os
import random
from pathlib import Path

import yaml
from dotenv import load_dotenv
from boulder.inference import (
    ColorPrinter,
    build_chat_messages,
    extract_targets,
    get_prompt_templates,
    get_tools_for_prompt,
    load_config,
    load_datasets,
    parse_results,
)
from boulder.llm import OllamaClient, OpenRouterClient, create_tool_handler
from boulder.domains.db import AttractionDB, HotelDB, RestaurantDB, TrainDB
from boulder.response_parser import ResponseParser


load_dotenv()


if __name__ == "__main__":
    args = load_config()

    # Save resolved config for reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    config_snapshot = {k: v for k, v in vars(args).items() if k != "config_file"}
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False, sort_keys=False)

    if args.api == "openrouter":
        llm_client = OpenRouterClient()
    else:
        llm_client = OllamaClient(host=args.host)
        available_models = llm_client.list_models()
        for model in args.models:
            if model not in available_models:
                raise ValueError(f"Model {model} not found in Ollama models")

    options = {
        "num_ctx": args.num_ctx,
        "temperature": args.temperature,
        "num_predict": args.num_predict,
    }

    base_db_path = Path(__file__).resolve().parent.parent / "data" / "db"
    attraction_db = AttractionDB.from_json(f"{base_db_path}/attraction_db.json")
    hotel_db = HotelDB.from_json(f"{base_db_path}/hotel_db.json")
    restaurant_db = RestaurantDB.from_json(f"{base_db_path}/restaurant_db.json")
    train_db = TrainDB.from_json(f"{base_db_path}/train_db.json")

    datasets = load_datasets(args.dataset)
    selected_setups = get_prompt_templates(args.setup_ids)

    # Process each dataset
    for dataset_idx, dataset in enumerate(datasets):
        data = dataset["examples"]
        answer_type = dataset["answer_type"]
        task_name = dataset["task_name"]
        dataset_path = dataset["path"]
        dataset_domains = args.domains[dataset_idx] if args.domains else None

        if "frequency" in task_name:
            train_db = TrainDB.from_json(f"{base_db_path}/train_db-extended.json")

        print(f"\n{'='*100}")
        print(f"Processing dataset: {task_name} (answer_type: {answer_type})")
        print(f"{'='*100}\n")

        for model in args.models:
            model_name = model.split("/")[-1] if "/" in model else model
            output_path = os.path.join(args.output_dir, f"{task_name}-{model_name}.jsonl")

            with open(output_path, "w") as f:
                pass

            print(f"Writing results to: {output_path}")

            for setup_id, dialogue_template, is_dialogue in selected_setups:

                random.seed(args.seed)

                tools = get_tools_for_prompt(
                    setup_id, attraction_db, hotel_db, restaurant_db, train_db,
                    domains=dataset_domains,
                )
                tools_dict = {tool.name: tool for tool in tools}
                tool_schemas = [tool.get_tool_schema() for tool in tools]
                tool_handler = create_tool_handler(tools_dict)

                for example in data:
                    prompt, chat_messages, is_dialogue = build_chat_messages(
                        example, setup_id, dialogue_template, answer_type,
                        domains=dataset_domains,
                    )

                    printer = ColorPrinter()
                    printer.print_messages(prompt, chat_messages, is_dialogue)

                    chat_kwargs = dict(
                        model=model,
                        messages=chat_messages,
                        think=args.think,
                        options=options,
                    )
                    if setup_id not in ["baseline", "baseline-with-role"]:
                        chat_kwargs.update(tool_schemas=tool_schemas, tool_handler=tool_handler)

                    result = llm_client.chat(**chat_kwargs)
                    response, cot, tool_calls = result.content, result.reasoning, result.tool_calls_made
                    if cot:
                        printer.print_assistant(cot + "\n")
                    printer.print_assistant(response)
                    print("-" * 100)

                    targets, parser_enum = extract_targets(example, answer_type)

                    result = {
                        "id": example["id"],
                        "setup_id": setup_id,
                        "model": model,
                        "task_name": task_name,
                        "answer_type": answer_type,
                        "messages": chat_messages,
                        "cot": cot,
                        "response": response,
                        "target": targets,
                        "tool_calls": tool_calls,
                        "parser_enum": parser_enum,
                    }

                    with open(output_path, "a") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"All responses written to {output_path}")

            print("Parsing answers...")
            resp_parser = ResponseParser(model=args.parser_model, host=args.host)
            parse_results(output_path, answer_type, resp_parser)
            print(f"All results with parsed answers written to {output_path}")
