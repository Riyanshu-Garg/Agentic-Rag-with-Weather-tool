from langsmith.evaluation import EvaluationResult, run_evaluator, evaluate
from typing import Optional, Any, Dict
from evaluate import run_agent_graph



def _get_agent_output(run: Any) -> Optional[str]:
    """Robustly extract agent textual output from a LangSmith run object."""
    try:
        outs = getattr(run, "outputs", None) or run.get("outputs", {}) if isinstance(run, dict) else None
    except Exception:
        outs = None

    if isinstance(outs, dict):
        for k in ("output", "generated_answer", "answer", "result"):
            if k in outs and outs[k]:
                return str(outs[k])
    if isinstance(outs, str):
        return outs
    return None

def _get_reference_from_example(example: Dict) -> Optional[str]:
    """Try common places where a dataset might keep a reference answer."""
    if not isinstance(example, dict):
        return None

    if "output" in example and example["output"]:
        return str(example["output"])

    if "outputs" in example and isinstance(example["outputs"], dict):
        for maybe in ("reference", "answer", "target", "label"):
            if maybe in example["outputs"] and example["outputs"][maybe]:
                return str(example["outputs"][maybe])


    inp = example.get("input") if isinstance(example.get("input"), dict) else None
    if inp:
        if "reference" in inp and inp["reference"]:
            return str(inp["reference"])
        if "expected_answer" in inp and inp["expected_answer"]:
            return str(inp["expected_answer"])

    for v in example.values():
        if isinstance(v, str) and v.strip():
            return v

    return None

def _get_retrieved_docs_from_example(example: Dict):
    """Return list of retrieved_docs from example if present."""
    if not isinstance(example, dict):
        return []
    inp = example.get("input") if isinstance(example.get("input"), dict) else None
    if inp and isinstance(inp.get("retrieved_docs"), list):
        return inp.get("retrieved_docs")
    # sometimes top-level
    if isinstance(example.get("retrieved_docs"), list):
        return example.get("retrieved_docs")
    return []


@run_evaluator
def check_contains_reference(run, example) -> EvaluationResult:
    """
    Score = 1 if reference (case-insensitive) is contained in agent output; else 0.
    If either is missing returns score=0 with diagnostic comment.
    """
    agent_output = _get_agent_output(run)
    reference_output = _get_reference_from_example(example)

    if not agent_output:
        return EvaluationResult(
            key="contains_reference",
            score=0,
            comment="Agent output missing."
        )

    if not reference_output:
        return EvaluationResult(
            key="contains_reference",
            score=0,
            comment="Reference missing from dataset example."
        )

    if str(reference_output).lower().strip() in str(agent_output).lower():
        return EvaluationResult(
            key="contains_reference",
            score=1,
            comment="Reference found in agent output."
        )

    return EvaluationResult(
        key="contains_reference",
        score=0,
        comment=f"Reference not found. Reference snippet: '{reference_output[:120]}'"
    )

@run_evaluator
def check_retrieval_presence(run, example) -> EvaluationResult:
    """
    For RAG: checks if the example provided 'retrieved_docs' (non-empty).
    Scores 1 if retrieved docs exist, 0 otherwise.
    Useful to ensure retrieval step executed or dataset included docs.
    """
    retrieved = _get_retrieved_docs_from_example(example)
    if retrieved and len(retrieved) > 0:
        return EvaluationResult(
            key="retrieval_present",
            score=1,
            comment=f"{len(retrieved)} retrieved_docs present in example."
        )
    # If the run object itself records retrieved docs, check that too
    try:
        run_meta = getattr(run, "metadata", None) or (run.get("metadata") if isinstance(run, dict) else None)
        if run_meta and run_meta.get("retrieved_docs"):
            return EvaluationResult(
                key="retrieval_present",
                score=1,
                comment="Retrieved docs recorded in run metadata."
            )
    except Exception:
        pass

    return EvaluationResult(
        key="retrieval_present",
        score=0,
        comment="No retrieved_docs present in example or run."
    )


if __name__ == "__main__":

    predictor = run_agent_graph           
    dataset_name = "Neura_Dynamics_Assignment"  # <-- your LangSmith dataset name

    results = evaluate(
        predictor,
        data=dataset_name,
        description="Evaluation run with custom evaluators: contains_reference + retrieval_presence",
        evaluators=[check_contains_reference, check_retrieval_presence],
    )

    print("Custom evaluation results:")
    print(results)
