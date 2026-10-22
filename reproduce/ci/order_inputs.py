"""Read saved ORDER-DR predictions and their model identities."""
import hashlib
import json
import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def policy_selection(model, policy):
    selected = policy[model["policy_entry"]] if "policy_entry" in model else policy
    if model.get("policy_entry") == "ORDER_CODE__2101":
        if selected["fusion"] != [0.5, 0.5] or selected["members"] != model["member_ids"]:
            raise ValueError("ORDER member/fusion policy differs")
    return selected, selected[model.get("policy_selection", "selection")]


def validate_npz_identity(z, binding, array_sha):
    # Same source:ID convention accepted by archived supplement_ci.validate_rows.
    ids = [str(key).rsplit(":", 1)[-1] for key in z["image_key"]]
    grade = z["grade"]
    if len(ids) != int(binding["images"]) or len(set(ids)) != len(ids):
        raise ValueError("ORDER full-cohort identity/count mismatch")
    for values, key in [(ids, "ordered_original_id_sha256"), (grade, "grade_sha256"),
                        (z["row_index"], "row_index_sha256"),
                        (z["component_hash"], "component_sha256")]:
        if array_sha(values) != binding[key]:
            raise ValueError("ORDER original ordered identity differs: " + key)
    return ids, grade


def project_order(entry, arrays):
    """Literal archived projection: mean probabilities, argmax, fixed log-odds.

    MESS already stores fused score/pred; retain them. No clipping, cuts or refit.
    """
    if entry["projection"] == "ORDER_components":
        if entry["weights"] != [0.5, 0.5] or len(arrays) != 2:
            raise ValueError("Only the registered equal two-member fusion is allowed")
        probabilities = (arrays[0]["probabilities"] + arrays[1]["probabilities"]) / 2
        return probabilities.argmax(1), (np.log(probabilities[:, 2:].sum(1))
                                         - np.log(probabilities[:, :2].sum(1)))
    if entry["projection"] != "standard" or len(arrays) != 1:
        raise ValueError("Unexpected ORDER prediction projection")
    z = arrays[0]
    probabilities = z["probabilities"]
    pred = z["pred"] if "pred" in z else probabilities.argmax(1)
    score = (z["score"] if "score" in z else
             np.log(probabilities[:, 2:].sum(1)) - np.log(probabilities[:, :2].sum(1)))
    return pred, score


def load_order_prediction(entry, root, binding, array_sha, input_hashes):
    arrays = []
    for member in entry.get("members", [entry]):
        path, receipt_path = root / member["path"], root / member["receipt_path"]
        if sha(path) != member["sha256"] or sha(receipt_path) != member["receipt_sha256"]:
            raise ValueError("ORDER original prediction/receipt SHA256 differs")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("sha256", receipt.get("prediction_sha256")) != member["sha256"]
                or receipt.get("model_sha256", receipt.get("system_identity_sha256"))
                != member["model_sha256"]):
            raise ValueError("ORDER saved prediction checkpoint identity differs")
        with np.load(path, allow_pickle=False) as source:
            z = {key: source[key] for key in
                 ["image_key", "grade", "row_index", "component_hash", "probabilities", "pred", "score"]
                 if key in source.files}
        ids, grade = validate_npz_identity(z, binding, array_sha)
        p = z["probabilities"]
        if p.shape != (len(grade), 5) or not np.isfinite(p).all():
            raise ValueError("Invalid saved ORDER five-grade probabilities")
        arrays.append(z)
        input_hashes.extend([{"path": str(path), "sha256": member["sha256"]},
                             {"path": str(receipt_path), "sha256": member["receipt_sha256"]}])
    pred, score = project_order(entry, arrays)
    return ids, grade, pred, score
