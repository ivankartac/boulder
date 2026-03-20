from boulder.evaluation import EvaluationPipeline, load_eval_config


if __name__ == "__main__":
    config = load_eval_config()
    pipeline = EvaluationPipeline.from_config(config)
    pipeline.run()
