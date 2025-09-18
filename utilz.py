import pandas as pd

HEALTHY = "Asymptomatic controls"
DISEASE = "Pancreatic diseases"
CANCER = "Pancreatic cancer"


def save_report(y_pred, y_test_encoded, dataset, le):
    y_true = y_test_encoded
    y_pred_s = pd.Series(y_pred, index=y_true.index, name="y_pred")

    eval_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_s})
    class_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Mapowanie klas:", class_map)

    results = {}

    for cls, cls_code in class_map.items():
        if len(class_map) ==2 and cls_code != 0:
            TP_idx = eval_df.index[(eval_df.y_true == cls_code) & (eval_df.y_pred == cls_code)]
            FP_idx = eval_df.index[(eval_df.y_true != cls_code) & (eval_df.y_pred == cls_code)]
            FN_idx = eval_df.index[(eval_df.y_true == cls_code) & (eval_df.y_pred != cls_code)]
            TN_idx = eval_df.index[(eval_df.y_true != cls_code) & (eval_df.y_pred != cls_code)]

            results[cls] = {
                "TP": TP_idx,
                "FP": FP_idx,
                "FN": FN_idx,
                "TN": TN_idx,
            }

            for key in ["TP", "FP", "FN", "TN"]:
                print(f"\n--- {key} samples metadata ---")
                for idx in results[cls][key]:
                    sample_meta = dataset.meta.loc[idx]
                    print(f"{key} - Sample ID: {idx}, Metadata:")
                    print(sample_meta["Group"], sample_meta["Sex"], sample_meta["Age"], sample_meta["Stage"])
                    print("---")

    return

