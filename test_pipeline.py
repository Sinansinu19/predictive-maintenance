from src.pipeline import run_pipeline

test_cases = [
    {
        "machine_id": "NORMAL-001",
        "type_": "M",
        "air_temp_k": 300.0,
        "process_temp_k": 310.0,
        "rot_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 100.0,
    },
    {
        "machine_id": "HIGH-WEAR-001",
        "type_": "L",
        "air_temp_k": 300.0,
        "process_temp_k": 320.0,
        "rot_speed_rpm": 1400.0,
        "torque_nm": 50.0,
        "tool_wear_min": 240.0,
    },
    {
        "machine_id": "HIGH-LOAD-001",
        "type_": "H",
        "air_temp_k": 305.0,
        "process_temp_k": 330.0,
        "rot_speed_rpm": 2500.0,
        "torque_nm": 180.0,
        "tool_wear_min": 200.0,
    },
]

for case in test_cases:
    result = run_pipeline(**case)

    print("\n" + "=" * 60)
    print("MACHINE:", result["machine_id"])
    print("FAILURE PROBABILITY:", result["failure_prediction"]["probability"])
    print("ANOMALY:", result["anomaly_detection"]["anomaly_flag"])
    print("FAILURE MODE:", result["failure_mode"]["prediction"])
    print("PRIORITY:", result["maintenance"]["priority"])
    print("RECOMMENDATION:", result["maintenance"]["recommendation"])