import argparse
import json
import os

from dotenv import load_dotenv

from boulder.dialogue_template_generator import generate_paraphrases

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if OPENROUTER_API_KEY is None:
    raise ValueError("OPENROUTER_API_KEY is not set")

parser = argparse.ArgumentParser(description="Generate dialogue template paraphrases using LLM")
parser.add_argument("-i", "--input", required=True, help="Input JSON file containing dialogue templates")
parser.add_argument("-o", "--output", help="Output JSON file (defaults to input file)")
parser.add_argument("-m", "--model", help="Model to use for generation")
parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Temperature for generation.")
parser.add_argument("-n", "--num-paraphrases", type=int, default=10, help="Number of paraphrases to generate")
parser.add_argument("-r", "--max-retries", type=int, default=3, help="Maximum number of retries for each paraphrase generation")
parser.add_argument("--api", default="openrouter", choices=["ollama", "openrouter"], help="API backend to use (default: openrouter)")
args = parser.parse_args()

output_path = args.output if args.output else args.input

with open(args.input, "r") as f:
    templates = json.load(f)

for template_name, template_data in templates.items():
    original_dialogue = template_data.get("original", [])

    if not original_dialogue:
        print(f"Skipping {template_name}: empty template")
        continue

    print(f"\n{'='*60}")
    print(f"Processing template: {template_name}")
    print(f"{'='*60}")

    paraphrases = generate_paraphrases(
        original_dialogue,
        model=args.model,
        temperature=args.temperature,
        api_key=OPENROUTER_API_KEY,
        api=args.api,
        num_paraphrases=args.num_paraphrases,
        max_retries=args.max_retries,
    )

    templates[template_name]["paraphrases"] = paraphrases
    print(f"Generated {len(paraphrases)} valid paraphrases for {template_name}")

with open(output_path, "w") as f:
    json.dump(templates, f, indent=4)

print(f"\n{'='*60}")
print(f"Completed! Updated {output_path}")
print(f"{'='*60}")
